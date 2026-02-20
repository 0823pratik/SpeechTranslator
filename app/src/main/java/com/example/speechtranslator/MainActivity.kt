package com.example.speechtranslator
import kotlinx.coroutines.withContext
import kotlinx.coroutines.launch
import kotlinx.coroutines.Dispatchers

import android.Manifest
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.graphics.Color
import android.os.Bundle
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.google.android.material.button.MaterialButton

import java.io.File
import java.util.Locale
import android.app.AlertDialog
import kotlinx.coroutines.CoroutineScope

class MainActivity : AppCompatActivity() {

    companion object {
        private const val REQ_MIC       = 100
        private const val MODEL_DIR     = "models"
        private const val WHISPER_MODEL = "ggml-tiny-q8_0.bin"
        private const val LLAMA_MODEL   = "gemma-2-2b-it-Q4_K_M.gguf"
    }

    private lateinit var tvStatus:           TextView
    private lateinit var tvTranscription:    TextView
    private lateinit var tvTranslation:      TextView
    private lateinit var btnRecord:          MaterialButton
    private lateinit var btnSwapLanguages:   MaterialButton
    private lateinit var btnCopyTranslation: MaterialButton
    private lateinit var statusDot:          View
    private lateinit var spinnerSource:      Spinner
    private lateinit var spinnerTarget:      Spinner

    private lateinit var pipeline: PipelineManager
    private lateinit var tts:      MmsTtsManager
    private var recorder: AudioCapture? = null

    private var isRecording = false
    private val translationBuf = StringBuilder()

    private val languageMap = mapOf(
        "English"  to "en",
        "Hindi"    to "hi",
        "French"   to "fr",
        "Spanish"  to "es",
        "German"   to "de",
        "Tamil"    to "ta",
        "Arabic"   to "ar"
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        tvStatus           = findViewById(R.id.tvStatus)
        tvTranscription    = findViewById(R.id.tvTranscription)
        tvTranslation      = findViewById(R.id.tvTranslation)
        btnRecord          = findViewById(R.id.btnRecord)
        btnSwapLanguages   = findViewById(R.id.btnSwapLanguages)
        btnCopyTranslation = findViewById(R.id.btnCopyTranslation)
        statusDot          = findViewById(R.id.statusDot)
        spinnerSource      = findViewById(R.id.spinnerSourceLanguage)
        spinnerTarget      = findViewById(R.id.spinnerTargetLanguage)

        pipeline = PipelineManager(this)
        tts      = MmsTtsManager(this, "${getExternalFilesDir(null)?.absolutePath}/mms_tts")

        btnRecord.isEnabled = false
        tvStatus.text       = "Initializing…"
        setStatusDot("grey")

        bindPipelineCallbacks()
        setupLanguageSpinners()
        bindUi()
        checkMicPermission()
    }

    private fun bindPipelineCallbacks() {
        pipeline.onTranscription = { text ->
            runOnUiThread {
                tvTranscription.text = text
                translationBuf.clear()
                tvTranslation.text = ""
            }
        }
        pipeline.onTranslationToken = { token ->
            translationBuf.append(token)
            runOnUiThread { tvTranslation.text = translationBuf.toString() }
        }
        pipeline.onTranslationDone = {
            runOnUiThread {
                tvStatus.text = "Ready"
                setStatusDot("green")
                btnRecord.isEnabled = true
            }
        }
        pipeline.onError = { msg ->
            runOnUiThread {
                tvStatus.text = "Error"
                setStatusDot("red")
                Toast.makeText(this, "Error: $msg", Toast.LENGTH_LONG).show()
            }
        }
    }

    private fun setupLanguageSpinners() {
        val langNames = languageMap.keys.toList()
        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, langNames)

        spinnerSource.adapter = adapter
        spinnerTarget.adapter = adapter

        spinnerSource.setSelection(langNames.indexOf("Hindi"))
        spinnerTarget.setSelection(langNames.indexOf("English"))

        spinnerSource.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(p0: AdapterView<*>?, p1: View?, pos: Int, id: Long) {
                pipeline.sourceLanguageCode = languageMap[langNames[pos]] ?: "hi"
            }
            override fun onNothingSelected(p0: AdapterView<*>?) {}
        }

        spinnerTarget.onItemSelectedListener = object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(p0: AdapterView<*>?, p1: View?, pos: Int, id: Long) {
                val targetCode = languageMap[langNames[pos]] ?: "en"
                pipeline.targetLanguageCode = targetCode
            }
            override fun onNothingSelected(p0: AdapterView<*>?) {}
        }
    }

    private fun bindUi() {
        btnRecord.setOnClickListener {
            if (!isRecording) startRecording() else stopRecording()
        }

        btnSwapLanguages.setOnClickListener {
            val srcPos = spinnerSource.selectedItemPosition
            val tgtPos = spinnerTarget.selectedItemPosition
            spinnerSource.setSelection(tgtPos)
            spinnerTarget.setSelection(srcPos)
        }

        btnCopyTranslation.setOnClickListener {
            val text = tvTranslation.text.toString()
            if (text.isNotBlank()) {
                val clipboard = getSystemService(CLIPBOARD_SERVICE) as android.content.ClipboardManager
                clipboard.setPrimaryClip(android.content.ClipData.newPlainText("translation", text))
                Toast.makeText(this, "Copied!", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun ensureModelsPresent(onReady: () -> Unit) {
        val modelRoot = File(getExternalFilesDir(null), "mms_tts")
        val langs = listOf("eng", "hin", "fra", "spa", "deu", "tam", "ara")

        val missing = langs.filter { lang ->
            !File(modelRoot, "$lang/model.onnx").exists() ||
                    !File(modelRoot, "$lang/tokens.txt").exists()
        }

        if (missing.isEmpty()) {
            onReady()
        } else {
            tvStatus.text = "TTS models missing: ${missing.joinToString()}"
            setStatusDot("red")
        }
    }



    private fun loadModels() {
        val dir         = getExternalFilesDir(null)?.resolve(MODEL_DIR)
        val whisperPath = dir?.resolve(WHISPER_MODEL)?.absolutePath ?: ""
        val llamaPath   = dir?.resolve(LLAMA_MODEL)?.absolutePath   ?: ""

        if (!File(whisperPath).exists()) { toast("Missing: $WHISPER_MODEL"); return }
        if (!File(llamaPath).exists())   { toast("Missing: $LLAMA_MODEL");   return }

        btnRecord.isEnabled = false
        tvStatus.text = "Loading models…"
        setStatusDot("amber")

        ensureModelsPresent {
            lifecycleScope.launch(Dispatchers.IO) {
                val modelDir = "${getExternalFilesDir(null)?.absolutePath}/mms_tts"
                val ok = pipeline.init(whisperPath, llamaPath, modelDir)
                withContext(Dispatchers.Main) {
                    tvStatus.text       = if (ok) "Ready" else "Load failed"
                    btnRecord.isEnabled = ok
                    setStatusDot(if (ok) "green" else "red")
                }
            }
        }
    }

    private fun startRecording() {
        isRecording = true
        btnRecord.text = "Stop"
        tvStatus.text  = "Listening…"
        setStatusDot("amber")
        tvTranscription.text = ""
        tvTranslation.text   = ""

        recorder = AudioCapture(pipeline = pipeline)
        recorder!!.start(lifecycleScope)
    }

    private fun stopRecording() {
        isRecording = false
        recorder?.stop()
        recorder = null
        btnRecord.text = "Record"
        tvStatus.text  = "Processing…"
        setStatusDot("blue")
    }

    private fun checkMicPermission() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
            == PackageManager.PERMISSION_GRANTED
        ) loadModels()
        else ActivityCompat.requestPermissions(
            this, arrayOf(Manifest.permission.RECORD_AUDIO), REQ_MIC
        )
    }

    override fun onRequestPermissionsResult(
        code: Int, perms: Array<out String>, results: IntArray
    ) {
        super.onRequestPermissionsResult(code, perms, results)
        if (code == REQ_MIC && results.firstOrNull() == PackageManager.PERMISSION_GRANTED)
            loadModels()
        else
            toast("Microphone permission required")
    }

    override fun onDestroy() {
        super.onDestroy()
        if (isRecording) recorder?.stop()
        pipeline.release()
        tts.release()
    }

    private fun setStatusDot(state: String) {
        val color = when (state) {
            "green" -> "#3DD68C"
            "amber" -> "#F7C34F"
            "red"   -> "#F75F5F"
            "blue"  -> "#4F8EF7"
            else    -> "#8A94A6"
        }
        statusDot.backgroundTintList = ColorStateList.valueOf(Color.parseColor(color))
    }

    private fun toast(msg: String) =
        Toast.makeText(this, msg, Toast.LENGTH_LONG).show()
}
