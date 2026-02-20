package com.example.speechtranslator

import android.util.Log

class SpeechDetector(private val chunkMs: Int = 500) {
    companion object {
        private const val TAG = "SpeechDetector"
        private const val SPEECH_SNR_THRESHOLD  = 2.5f
        private const val SILENCE_SNR_THRESHOLD = 1.5f
        private const val SPEECH_FRAMES_TO_START = 1
        private const val SILENCE_FRAMES_TO_END = 2
        private const val MIN_SPEECH_FRAMES = 1
        private const val FLOOR_EMA_ALPHA = 0.05f
        private const val INITIAL_FLOOR   = 0.008f
        private const val MIN_FLOOR       = 0.003f
        private const val MAX_FLOOR       = 0.050f
    }

    var onSpeechStart: (() -> Unit)? = null
    var onSpeechEnd:   (() -> Unit)? = null

    var isSpeaking = false
        private set
    private var silenceFrames = 0
    private var loudFrames    = 0
    private var speechFrames  = 0
    private var noiseFloor    = INITIAL_FLOOR

    fun process(pcm: FloatArray) {
        val rms = computeRms(pcm)
        val snr = rms / noiseFloor.coerceAtLeast(MIN_FLOOR)
        if (snr < SILENCE_SNR_THRESHOLD) {
            noiseFloor = (FLOOR_EMA_ALPHA * rms + (1f - FLOOR_EMA_ALPHA) * noiseFloor)
                .coerceIn(MIN_FLOOR, MAX_FLOOR)
        }
        Log.d(TAG, "RMS=%.4f  floor=%.4f  SNR=%.1f".format(rms, noiseFloor, snr))
        if (!isSpeaking) {
            if (snr >= SPEECH_SNR_THRESHOLD) {
                loudFrames++
                if (loudFrames >= SPEECH_FRAMES_TO_START) {
                    isSpeaking    = true
                    silenceFrames = 0
                    speechFrames  = loudFrames
                    Log.i(TAG, "Speech START")
                    onSpeechStart?.invoke()
                }
            } else {
                loudFrames = 0
            }
        } else {
            if (snr >= SILENCE_SNR_THRESHOLD) {
                speechFrames++
                silenceFrames = 0
            } else {
                silenceFrames++
                if (silenceFrames >= SILENCE_FRAMES_TO_END) {
                    if (speechFrames >= MIN_SPEECH_FRAMES) {
                        isSpeaking   = false
                        loudFrames   = 0
                        speechFrames = 0
                        Log.i(TAG, "Speech END")
                        onSpeechEnd?.invoke()
                    } else {
                        isSpeaking    = false
                        loudFrames    = 0
                        speechFrames  = 0
                        silenceFrames = 0
                    }
                }
            }
        }
    }

    fun reset() {
        isSpeaking    = false
        silenceFrames = 0
        loudFrames    = 0
        speechFrames  = 0
    }

    val speaking: Boolean get() = isSpeaking
    val currentFloor: Float get() = noiseFloor

    private fun computeRms(pcm: FloatArray): Float {
        if (pcm.isEmpty()) return 0f
        var sum = 0.0
        for (s in pcm) sum += s * s
        return Math.sqrt(sum / pcm.size).toFloat()
    }
}
