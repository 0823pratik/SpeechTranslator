#include "whisper_bridge.h"
#include "whisper.h"
#include <android/log.h>
#include <string>
#include "ggml.h"


#define TAG  "WhisperBridge"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static whisper_context* g_ctx     = nullptr;
static int              g_threads = 4;

bool whisper_bridge_init(const char* model_path, int n_threads) {
    if (g_ctx) { whisper_free(g_ctx); g_ctx = nullptr; }
    g_threads = n_threads;

    whisper_context_params cp = whisper_context_default_params();
    cp.use_gpu = false;

    g_ctx = whisper_init_from_file_with_params(model_path, cp);
    if (!g_ctx) { LOGE("Failed to load: %s", model_path); return false; }
    LOGI("Whisper model loaded OK from %s", model_path);
    return true;
    // After whisper_init_from_file():
    LOGI("ggml CPU features: %s", ggml_cpu_has_sme() ? "SME=ON" : "SME=OFF");
    LOGI("ggml SVE: %s", ggml_cpu_has_sve() ? "SVE=ON" : "SVE=OFF");
    LOGI("ggml MATMUL_INT8: %s", ggml_cpu_has_matmul_int8() ? "MATMUL_INT8=ON" : "MATMUL_INT8=OFF");
}

std::string whisper_bridge_transcribe(const float* pcm, int n_samples, const char* lang) {
    if (!g_ctx) return "";

    whisper_full_params wp    = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wp.language               = lang;
    wp.translate              = false;
    wp.no_context             = false;
    wp.single_segment         = true;
    wp.print_realtime         = true;
    wp.print_progress         = false;
    wp.print_timestamps       = false;
    wp.suppress_blank         = true;
    // ✅ REMOVED: suppress_non_speech_tokens — not in this whisper.cpp version
    wp.n_threads              = g_threads;
    wp.audio_ctx              = 0;

    if (whisper_full(g_ctx, wp, pcm, n_samples) != 0) {
        LOGE("whisper_full() failed"); return "";
    }

    std::string out;
    int n = whisper_full_n_segments(g_ctx);
    for (int i = 0; i < n; ++i) {
        const char* seg = whisper_full_get_segment_text(g_ctx, i);
        if (seg) out += seg;
    }
    if (!out.empty() && out[0] == ' ') out = out.substr(1);
    return out;
}

void whisper_bridge_free() {
    if (g_ctx) { whisper_free(g_ctx); g_ctx = nullptr; }
}
