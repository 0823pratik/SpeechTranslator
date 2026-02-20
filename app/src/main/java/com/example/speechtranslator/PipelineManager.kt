package com.example.speechtranslator

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import android.util.Log
import kotlinx.coroutines.*
import kotlinx.coroutines.channels.Channel
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

@OptIn(
    kotlinx.coroutines.ExperimentalCoroutinesApi::class,
    kotlinx.coroutines.DelicateCoroutinesApi::class
)
class PipelineManager(private val context: Context) {

    companion object {
        private const val TAG                = "PipelineManager"
        private const val WHISPER_THREADS    = 4
        private const val LLAMA_THREADS      = 6
        private const val N_CTX              = 2048
        private const val MIN_SPEECH_SAMPLES = 3200   // ~200ms @ 16kHz
        private const val TTS_SAMPLE_RATE    = 22050

        // ── TTS sentence segmentation ──────────────────────────────────────
        // Flush to TTS immediately at hard sentence boundaries
        private val SENTENCE_END = setOf('.', '!', '?', '।', '\n', '،', '、', '，', '\u0964', '\u0965')
        // Flush at clause boundaries only if the buffer is long enough
        private val CLAUSE_END   = setOf(',', ';', ':')
        private const val CLAUSE_FLUSH_MIN = 50   // chars; raised from 40 to avoid tiny clips

        // ── Inter-sentence silence ─────────────────────────────────────────
        // Silence (ms) inserted between synthesised sentences during playback.
        // This prevents sentences from running together and sounds more natural.
        // Raise to 200ms if it still feels rushed; lower to 50ms if too slow.
        private const val INTER_SENTENCE_SILENCE_MS = 120L

        // ── Playback drain polling ─────────────────────────────────────────
        // How often (ms) we poll AudioTrack.playbackHeadPosition.
        // 8ms = ~1 audio frame at 22050Hz; fine-grained enough for accuracy.
        private const val PLAYBACK_POLL_MS = 8L
        // Safety margin added on top of calculated audio duration before timeout.
        private const val PLAYBACK_TIMEOUT_MARGIN_MS = 400L

        private val LANG_TO_MMS = mapOf(
            "en" to "eng", "hi" to "hin", "fr" to "fra",
            "es" to "spa", "de" to "deu", "ta" to "tam",
            "ar" to "ara", "zh" to "zho",
            "te" to "tel", "mr" to "mar"
        )

        init { System.loadLibrary("translator_native") }
    }

    // ── JNI ───────────────────────────────────────────────────────────────────
    private external fun nativeWhisperInit(path: String, threads: Int): Boolean
    private external fun nativeWhisperTranscribe(pcm: FloatArray, lang: String): String
    private external fun nativeWhisperFree()
    private external fun nativeLlamaInit(path: String, threads: Int, nCtx: Int): Boolean
    private external fun nativeLlamaTranslate(prompt: String, cb: TokenCallback)
    private external fun nativeLlamaFree()
    private external fun nativeGetBackendInfo(): String

    // ── Config & callbacks ────────────────────────────────────────────────────
    var sourceLanguageCode: String = ""
    var targetLanguageCode: String = ""
    var ttsEnabled:         Boolean = true

    var onTranscription:    ((String) -> Unit)? = null
    var onTranslationToken: ((String) -> Unit)? = null
    var onTranslationDone:  (() -> Unit)?       = null
    /**
     * Called when ALL TTS audio has finished playing.
     * MainActivity should re-enable the Record button here (not in onTranslationDone).
     * This prevents the microphone opening while the speaker is still active,
     * which was the root cause of the "TTS not synced" symptom.
     */
    var onTtsDone:          (() -> Unit)?       = null
    var onError:            ((String) -> Unit)? = null

    // ── Internal state ────────────────────────────────────────────────────────
    private val busy             = AtomicBoolean(false)
    private val capturingSpeech  = AtomicBoolean(false)
    private val pendingUtterance = AtomicReference<FloatArray?>(null)
    private val speechBuffer     = mutableListOf<FloatArray>()

    // Compute scope: Whisper + Llama inference + TTS synthesis
    private val computeScope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    // Playback scope: single dedicated thread — audio write never preempts compute
    private val playbackScope = CoroutineScope(
        newSingleThreadContext("tts-playback") + SupervisorJob()
    )

    private var initialized  = false
    private var ttsManager: MmsTtsManager? = null

    // ── Init / release ────────────────────────────────────────────────────────

    fun init(whisperPath: String, llamaPath: String, modelDir: String): Boolean {
        if (!File(whisperPath).exists()) { onError?.invoke("Whisper model not found"); return false }
        if (!File(llamaPath).exists())   { onError?.invoke("Llama model not found");   return false }

        if (!nativeWhisperInit(whisperPath, WHISPER_THREADS)) {
            onError?.invoke("Failed to load Whisper"); return false
        }
        if (!nativeLlamaInit(llamaPath, LLAMA_THREADS, N_CTX)) {
            onError?.invoke("Failed to load Llama"); return false
        }

        ttsManager = MmsTtsManager(context, modelDir)

        // Pre-warm TTS models for both languages in parallel so first utterance
        // has no cold-start synthesis delay.
        computeScope.launch(Dispatchers.IO) {
            val targetMms = LANG_TO_MMS[targetLanguageCode] ?: "hin"
            val sourceMms = LANG_TO_MMS[sourceLanguageCode] ?: "eng"
            coroutineScope {
                launch { ttsManager?.warmup(targetMms) }
                if (sourceMms != targetMms) launch { ttsManager?.warmup(sourceMms) }
            }
            Log.i(TAG, "TTS pre-warmed: $targetMms / $sourceMms")
        }

        Log.i(TAG, "Backend: ${nativeGetBackendInfo()}")
        initialized = true
        return true
    }

    fun release() {
        computeScope.cancel()
        playbackScope.cancel()
        if (initialized) {
            nativeWhisperFree()
            nativeLlamaFree()
            ttsManager?.release()
            initialized = false
        }
    }

    // ── Speech input ──────────────────────────────────────────────────────────

    fun onSpeechStart() {
        if (!initialized) return
        synchronized(speechBuffer) { speechBuffer.clear() }
        capturingSpeech.set(true)
        Log.d(TAG, "Speech capture started")
    }

    fun submitChunk(pcm: FloatArray) {
        if (!initialized || !capturingSpeech.get()) return
        synchronized(speechBuffer) { speechBuffer.add(pcm.copyOf()) }
    }

    fun onSpeechEnd() {
        if (!initialized) return
        capturingSpeech.set(false)

        val merged = synchronized(speechBuffer) {
            val out = speechBuffer.flatMap { it.toList() }.toFloatArray()
            speechBuffer.clear()
            out
        }

        Log.d(TAG, "Speech end: ${merged.size} samples (${merged.size / 16}ms)")

        if (merged.size < MIN_SPEECH_SAMPLES) {
            Log.d(TAG, "Too short, ignoring utterance")
            return
        }

        if (busy.compareAndSet(false, true)) {
            computeScope.launch { runPipeline(merged) }
        } else {
            // Queue latest utterance; drop older queued one if any
            val dropped = pendingUtterance.getAndSet(merged)
            if (dropped != null) Log.w(TAG, "Dropped queued utterance (pipeline busy)")
        }
    }

    // ── Core pipeline ─────────────────────────────────────────────────────────
    //
    //  Architecture (pipelined):
    //
    //  Llama token stream
    //      │ sentence boundary
    //      ▼
    //  synthChannel (UNLIMITED) ──► synthesisJob (Default):
    //                                   generateSamples() ──► audioChannel (capacity=2)
    //                                                              │
    //                                                   playbackJob (tts-playback):
    //                                                       playAudioStatic()  [blocking]
    //                                                       inter-sentence silence
    //
    //  Sentence N+1 is synthesised while sentence N is playing.
    //  Gap between sentences = INTER_SENTENCE_SILENCE_MS only.
    //
    //  onTtsDone fires after playbackJob completes — AFTER last audio drains.
    //  This is the correct signal for "safe to start next recording".

    private suspend fun runPipeline(pcm: FloatArray) {
        try {
            // ── 1. Transcribe ──────────────────────────────────────────────
            val transcribed = withContext(Dispatchers.Default) {
                nativeWhisperTranscribe(pcm, sourceLanguageCode)
            }.trim()

            if (transcribed.isBlank()) {
                Log.w(TAG, "Whisper returned empty result")
                onError?.invoke("No speech recognized")
                return
            }
            Log.i(TAG, "Whisper → \"$transcribed\"")
            onTranscription?.invoke(transcribed)

            // ── 2. Text-only path (TTS disabled) ──────────────────────────
            if (!ttsEnabled) {
                val sb = StringBuilder()
                withContext(Dispatchers.Default) {
                    nativeLlamaTranslate(buildPrompt(transcribed)) { token ->
                        sb.append(token)
                        onTranslationToken?.invoke(token)
                    }
                }
                Log.i(TAG, "Llama → \"${sb.trim()}\"")
                onTranslationDone?.invoke()
                onTtsDone?.invoke()
                return
            }

            // ── 3. Pipelined translate + synthesise + play ─────────────────
            val mmsCode = LANG_TO_MMS[targetLanguageCode.lowercase()] ?: "eng"

            // Channel: sentence strings → synthesis worker
            val synthChannel = Channel<String>(Channel.UNLIMITED)
            // Channel: audio float arrays → playback worker (1 ahead buffer)
            val audioChannel = Channel<FloatArray>(capacity = 2)

            // Worker A — synthesis on IO dispatcher (unbounded pool).
            // CRITICAL: must NOT use Dispatchers.Default here — Llama.translate()
            // blocks all Default threads, starving synthesis until translation finishes.
            // IO has an unbounded thread pool so synthesis runs concurrently with Llama.
            val synthesisJob = computeScope.launch(Dispatchers.IO) {
                for (sentence in synthChannel) {
                    val t0 = System.currentTimeMillis()
                    val samples = ttsManager?.generateSamples(sentence, mmsCode)
                        ?: FloatArray(0)
                    val genMs = System.currentTimeMillis() - t0
                    val durMs = if (samples.isNotEmpty())
                        samples.size.toLong() * 1000L / TTS_SAMPLE_RATE else 0L
                    Log.i(TAG, "TTS synthesis: ${samples.size} samples, " +
                            "duration=${durMs}ms, genTime=${genMs}ms, text=\"$sentence\"")
                    if (samples.isNotEmpty()) audioChannel.send(samples)
                }
                audioChannel.close()
            }

            // Worker B — playback (dedicated thread; never shares CPU with synthesis)
            val playbackJob = playbackScope.launch {
                var firstChunk = true
                for (samples in audioChannel) {
                    // Insert silence between sentences (not before the very first one)
                    if (!firstChunk) {
                        Thread.sleep(INTER_SENTENCE_SILENCE_MS)
                    }
                    firstChunk = false
                    playAudioStatic(samples)
                }
                // *** THIS is the correct place to fire onTtsDone ***
                // Audio has fully drained; it is now safe to open the microphone.
                Log.i(TAG, "All TTS playback complete")
                onTtsDone?.invoke()
            }

            // ── 4. Translate, flushing segments to synthChannel ────────────
            val fullTranslation = StringBuilder()
            val segmentBuffer   = StringBuilder()

            // Run Llama on IO — it's a blocking JNI call. Using Dispatchers.IO ensures
            // it doesn't occupy all Default threads, which would starve the synthesisJob
            // (also on IO) and prevent TTS from starting until translation is complete.
            withContext(Dispatchers.IO) {
                nativeLlamaTranslate(buildPrompt(transcribed)) { token ->
                    fullTranslation.append(token)
                    segmentBuffer.append(token)
                    onTranslationToken?.invoke(token)

                    val lastChar = token.lastOrNull() ?: return@nativeLlamaTranslate

                    val isSentenceEnd = lastChar in SENTENCE_END && segmentBuffer.length > 4
                    val isClauseEnd   = lastChar in CLAUSE_END
                            && segmentBuffer.length >= CLAUSE_FLUSH_MIN

                    if (isSentenceEnd || isClauseEnd) {
                        val segment = prepareForTts(segmentBuffer.toString())
                        segmentBuffer.clear()
                        if (segment.isNotBlank()) {
                            Log.d(TAG, "Flushing TTS segment: \"$segment\"")
                            synthChannel.trySend(segment)
                        }
                    }
                }
            }

            Log.i(TAG, "Llama → \"${fullTranslation.trim()}\"")
            onTranslationDone?.invoke()

            // Flush tail (text after the last sentence boundary)
            val tail = prepareForTts(segmentBuffer.toString())
            if (tail.isNotBlank()) {
                Log.d(TAG, "Flushing TTS tail: \"$tail\"")
                synthChannel.trySend(tail)
            }
            synthChannel.close()

            // Wait for synthesis to finish feeding audioChannel,
            // then wait for all audio to drain before returning.
            synthesisJob.join()
            playbackJob.join()

        } catch (e: CancellationException) {
            throw e  // don't swallow coroutine cancellation
        } catch (e: Exception) {
            Log.e(TAG, "Pipeline error: ${e.message}", e)
            onError?.invoke(e.message ?: "Unknown pipeline error")
        } finally {
            busy.set(false)
            // If a new utterance arrived while we were busy, run it now
            val queued = pendingUtterance.getAndSet(null)
            if (queued != null && busy.compareAndSet(false, true)) {
                computeScope.launch { runPipeline(queued) }
            }
        }
    }

    // ── TTS playback ──────────────────────────────────────────────────────────

    /**
     * Writes [samples] to an AudioTrack (MODE_STATIC) and blocks until the
     * hardware audio buffer has fully drained.
     *
     * Uses playbackHeadPosition polling instead of Thread.sleep(audioDuration)
     * so there is no fixed over/under-wait padding.
     */
    private fun playAudioStatic(samples: FloatArray) {
        val track = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setEncoding(AudioFormat.ENCODING_PCM_FLOAT)
                    .setSampleRate(TTS_SAMPLE_RATE)
                    .setChannelMask(AudioFormat.CHANNEL_OUT_MONO)
                    .build()
            )
            .setTransferMode(AudioTrack.MODE_STATIC)
            .setBufferSizeInBytes(samples.size * 4)  // 4 bytes per float
            .build()

        val t0 = System.currentTimeMillis()
        track.write(samples, 0, samples.size, AudioTrack.WRITE_BLOCKING)
        track.play()

        // Drain: poll head position until all frames are consumed.
        val audioDurationMs = samples.size.toLong() * 1000L / TTS_SAMPLE_RATE
        val deadline = System.currentTimeMillis() + audioDurationMs + PLAYBACK_TIMEOUT_MARGIN_MS

        while (track.playbackHeadPosition < samples.size) {
            if (System.currentTimeMillis() > deadline) {
                Log.w(TAG, "Playback drain timeout after ${audioDurationMs}ms")
                break
            }
            Thread.sleep(PLAYBACK_POLL_MS)
        }

        track.stop()
        track.release()
        Log.i(TAG, "TTS playback done (${System.currentTimeMillis() - t0}ms, " +
                "expected ${audioDurationMs}ms)")
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /**
     * Cleans a translation segment for TTS input.
     *
     * KEY FIX vs original: we no longer strip ALL punctuation.
     * Keeping commas, colons etc. as spaces helps VITS produce
     * natural pauses. Only remove chars that cause TTS errors.
     */
    private fun prepareForTts(text: String): String =
        text
            // Remove characters that cause TTS phonemisation errors
            .replace(Regex("[\"'()\\[\\]{}*_~`^]"), " ")
            // Sentence-ending punctuation → short pause (space is fine for VITS)
            .replace(Regex("[.!?।\u0964\u0965،、，]+\\s*"), " ")
            // Collapse whitespace
            .replace(Regex("\\s+"), " ")
            .trim()

    private fun getLanguageName(code: String) = when (code.lowercase()) {
        "en" -> "English";  "hi" -> "Hindi";   "fr" -> "French"
        "es" -> "Spanish";  "de" -> "German";  "ta" -> "Tamil"
        "zh" -> "Chinese";  "ar" -> "Arabic"
        "te" -> "Telugu";   "mr" -> "Marathi"
        else -> "English"
    }

    private fun buildPrompt(text: String): String {
        val src = getLanguageName(sourceLanguageCode)
        val tgt = getLanguageName(targetLanguageCode)
        return buildString {
            append("<start_of_turn>user\n")
            append("Translate the following $src text to $tgt. ")
            append("Output only the translated $tgt text, nothing else.\n\n")
            append("Text: \"$text\"\n")
            append("<end_of_turn>\n")
            append("<start_of_turn>model\n")
        }
    }
}