package com.example.speechtranslator

import android.content.Context
import android.util.Log
import com.k2fsa.sherpa.onnx.OfflineTts
import com.k2fsa.sherpa.onnx.OfflineTtsConfig
import com.k2fsa.sherpa.onnx.OfflineTtsModelConfig
import com.k2fsa.sherpa.onnx.OfflineTtsVitsModelConfig
import java.io.File

class MmsTtsManager(
    private val context: Context,
    private val modelDir: String
) {
    companion object {
        private const val TAG = "MmsTtsManager"
        private const val MAX_CACHED_MODELS = 2

        /**
         * Length scale per language — controls speaking rate.
         * >1.0 = slower (more natural), <1.0 = faster.
         *
         * MMS-TTS default cadence is brisk; these values slow it down
         * to feel closer to natural human speech pace.
         *
         * Tuning guide:
         *   - If TTS still sounds too fast → increase by 0.1
         *   - If TTS sounds too slow/dragged → decrease by 0.1
         *   - Range that typically sounds natural: 1.1 – 1.4
         */
        private val LENGTH_SCALE_BY_LANG = mapOf(
            "eng" to 1.20f,   // English  — default MMS is slightly rushed
            "hin" to 1.25f,   // Hindi    — MMS-hin runs noticeably fast
            "deu" to 1.15f,   // German   — long compound words need a bit more time
            "fra" to 1.20f,   // French
            "spa" to 1.15f,   // Spanish  — naturally faster cadence; less correction
            "tam" to 1.30f,   // Tamil    — MMS-tam is very fast by default
            "ara" to 1.25f,   // Arabic
            "zho" to 1.10f,   // Chinese  — tonal language; less stretching needed
            "tel" to 1.25f,   // Telugu
            "mar" to 1.25f,   // Marathi
        )

        // Noise scale: controls how expressive/varied the synthesis is.
        // 0.333 = smoother / more monotone, 0.667 (default) = more natural variation.
        private const val NOISE_SCALE   = 0.667f
        private const val NOISE_SCALE_W = 0.8f
    }

    // LRU cache: mmsCode → loaded OfflineTts instance
    private val modelCache = LinkedHashMap<String, OfflineTts>(
        MAX_CACHED_MODELS + 1, 0.75f, true  // accessOrder=true → LRU
    )
    private val cacheLock = Any()

    // ── Warmup ────────────────────────────────────────────────────────────────

    fun warmup(mmsCode: String) {
        Log.i(TAG, "Warming up TTS model: $mmsCode")
        getOrLoad(mmsCode)
    }

    // ── Primary API ───────────────────────────────────────────────────────────

    /**
     * Synthesises [text] for [mmsCode] (e.g. "eng", "hin").
     * Returns raw PCM float samples at 22 050 Hz mono.
     *
     * [speedOverride] lets the caller nudge speed on top of the language default.
     * 1.0 = use language default, 0.9 = 10% slower than default, etc.
     */
    fun generateSamples(
        text: String,
        mmsCode: String,
        speedOverride: Float = 0.5f
    ): FloatArray {
        if (text.isBlank()) return FloatArray(0)

        val normalized = normalizeTextForLang(text, mmsCode)
        if (normalized.isBlank()) return FloatArray(0)

        val tts = getOrLoad(mmsCode) ?: run {
            Log.w(TAG, "Model unavailable for $mmsCode — skipping TTS")
            return FloatArray(0)
        }

        // speed param in sherpa-onnx is the inverse of length_scale —
        // speed=1.0 means "use model's own length_scale", but we bake
        // length_scale into the config at load time, so speed=1.0 here
        // correctly applies our per-language rate.
        // speedOverride lets callers nudge further if desired.
        return try {
            tts.generate(normalized.trim(), sid = 0, speed = speedOverride)?.samples
                ?: FloatArray(0)
        } catch (e: Exception) {
            Log.e(TAG, "generate() failed for $mmsCode: ${e.message}")
            FloatArray(0)
        }
    }

    // ── Internal ──────────────────────────────────────────────────────────────

    private fun getOrLoad(mmsCode: String): OfflineTts? {
        synchronized(cacheLock) {
            modelCache[mmsCode]?.let { return it }

            if (modelCache.size >= MAX_CACHED_MODELS) {
                val lruKey = modelCache.keys.first()
                Log.i(TAG, "Evicting TTS model: $lruKey")
                modelCache.remove(lruKey)?.release()
            }

            val loaded = loadModel(mmsCode) ?: return null
            modelCache[mmsCode] = loaded
            return loaded
        }
    }

    private fun loadModel(mmsCode: String): OfflineTts? {
        val dir        = "$modelDir/$mmsCode"
        val modelFile  = File("$dir/model.onnx")
        val tokensFile = File("$dir/tokens.txt")

        if (!modelFile.exists()) {
            Log.e(TAG, "Missing model.onnx at: ${modelFile.absolutePath}")
            return null
        }
        if (!tokensFile.exists()) {
            Log.e(TAG, "Missing tokens.txt at: ${tokensFile.absolutePath}")
            return null
        }

        val lengthScale = LENGTH_SCALE_BY_LANG[mmsCode] ?: 1.20f
        Log.i(TAG, "Loading TTS model: $mmsCode  length_scale=$lengthScale")

        return try {
            val lexiconPath = "$dir/lexicon.txt"
            val config = OfflineTtsConfig(
                model = OfflineTtsModelConfig(
                    vits = OfflineTtsVitsModelConfig(
                        model       = modelFile.absolutePath,
                        tokens      = tokensFile.absolutePath,
                        lexicon     = if (File(lexiconPath).exists()) lexiconPath else "",
                        lengthScale = lengthScale,
                        noiseScale  = NOISE_SCALE,
                        noiseScaleW = NOISE_SCALE_W,
                    ),
                    numThreads = 4,
                    debug      = false
                )
            )
            OfflineTts(config = config).also {
                Log.i(TAG, "Loaded TTS model: $mmsCode")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load TTS model $mmsCode: ${e.message}")
            null
        }
    }

    /**
     * Per-language text normalisation before synthesis.
     * Prevents unknown characters from being silently skipped,
     * which causes the TTS to sound truncated or garbled.
     */
    private fun normalizeTextForLang(text: String, mmsCode: String): String {
        var s = text
        when (mmsCode) {
            "hin" -> {
                s = s.replace('\u093C', ' ')    // strip standalone nukta
                    .replace('\u095B', '\u091C') // ज़ → ज
                    .replace('\u095C', '\u0921') // ड़ → ड
                    .replace('\u095D', '\u0922') // ढ़ → ढ
                    .replace('\u0929', '\u0928') // ऩ → न
                    .replace('\u0931', '\u0930') // ऱ → र
                    .replace(Regex("\\s+"), " ")
            }
            "ara" -> {
                // Strip harakat (diacritics) — MMS-ara was trained without them
                s = s.replace(Regex("[\\u064B-\\u065F\\u0670]"), "")
            }
            "tam" -> {
                // Tamil: currently no known MMS-tam vocab issues
            }
            "eng" -> {
                // Expand common abbreviations that MMS-eng mispronounces
                s = s.replace(Regex("\\bDr\\."), "Doctor")
                    .replace(Regex("\\bMr\\."), "Mister")
                    .replace(Regex("\\bMrs\\."), "Missus")
                    .replace(Regex("\\bSt\\."), "Street")
                    .replace(Regex("\\b(\\d+)%"), "$1 percent")
            }
        }
        // Universal: collapse whitespace
        return s.replace(Regex("\\s+"), " ").trim()
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    fun release() {
        synchronized(cacheLock) {
            modelCache.values.forEach { it.release() }
            modelCache.clear()
        }
    }
}