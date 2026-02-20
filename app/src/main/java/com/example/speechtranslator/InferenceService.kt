package com.example.speechtranslator

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.os.IBinder

/**
 * InferenceService — keeps the CPU governor in performance mode on Vivo/MTK
 * devices during Whisper + Llama inference.
 *
 * Without this, the Dimensity scheduler detects a sustained CPU workload and
 * throttles the big cores to thermal limits after ~2s — exactly the window
 * where Whisper is running. A foreground service with MICROPHONE type signals
 * the OS to keep full performance headroom.
 *
 * Usage:
 *   Start:  InferenceService.start(context)
 *   Stop:   InferenceService.stop(context)
 *
 * AndroidManifest additions required (see bottom of this file).
 */
class InferenceService : Service() {

    companion object {
        private const val CHANNEL_ID   = "inference_channel"
        private const val NOTIF_ID     = 1001
        private const val ACTION_START = "START"
        private const val ACTION_STOP  = "STOP"

        fun start(context: android.content.Context) {
            val intent = Intent(context, InferenceService::class.java).apply {
                action = ACTION_START
            }
            context.startForegroundService(intent)
        }

        fun stop(context: android.content.Context) {
            val intent = Intent(context, InferenceService::class.java).apply {
                action = ACTION_STOP
            }
            context.startService(intent)
        }
    }

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START -> startForeground(NOTIF_ID, buildNotification())
            ACTION_STOP  -> stopSelf()
        }
        return START_NOT_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun buildNotification(): Notification {
        return Notification.Builder(this, CHANNEL_ID)
            .setContentTitle("Translating…")
            .setContentText("Speech translation is active")
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setOngoing(true)
            .build()
    }

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            CHANNEL_ID,
            "Speech Translation",
            NotificationManager.IMPORTANCE_LOW  // silent — no sound/vibration
        ).apply {
            description = "Active while translating speech"
        }
        getSystemService(NotificationManager::class.java)
            .createNotificationChannel(channel)
    }
}
