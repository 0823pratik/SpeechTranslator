package com.example.speechtranslator

import android.Manifest
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import androidx.annotation.RequiresPermission
import kotlinx.coroutines.*

class AudioCapture(
    private val sampleRate: Int = 16_000,
    private val chunkMs:    Int = 500,
    private val pipeline:   PipelineManager
) {
    companion object { private const val TAG = "AudioCapture" }

    private var record: AudioRecord? = null
    private var job:    Job?         = null

    private val speechDetector = SpeechDetector().also {
        it.onSpeechStart = { pipeline.onSpeechStart() }
        it.onSpeechEnd   = { pipeline.onSpeechEnd()   }
    }

    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    fun start(scope: CoroutineScope) {
        if (record != null) return
        speechDetector.reset()

        val chunkSamples = sampleRate * chunkMs / 1000
        val minBuf = AudioRecord.getMinBufferSize(
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT
        )

        record = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION,
            sampleRate,
            AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT,
            maxOf(minBuf, chunkSamples * 2)
        )
        record!!.startRecording()
        Log.i(TAG, "Started @ ${sampleRate}Hz  chunkMs=${chunkMs}")

        job = scope.launch(Dispatchers.IO) {
            val buf = ShortArray(chunkSamples)
            while (isActive) {
                val read = record?.read(buf, 0, buf.size) ?: break
                if (read > 0) {
                    val pcm = FloatArray(read) { buf[it] / 32768f }
                    speechDetector.process(pcm)
                    pipeline.submitChunk(pcm)
                }
            }
        }
    }

    fun stop() {
        if (speechDetector.isSpeaking) {
            speechDetector.onSpeechEnd?.invoke()
        }
        job?.cancel();    job    = null
        record?.stop()
        record?.release(); record = null
        Log.i(TAG, "Stopped")
    }
}