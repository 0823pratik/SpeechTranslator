#include "llama_bridge.h"
#include "llama.cpp/include/llama.h"
#include "llama.cpp/ggml/include/ggml-cpu.h"
#include <android/log.h>
#include <string>
#include <vector>
#include <functional>

#define TAG  "LlamaBridge"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO,  TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

static llama_model*   g_model   = nullptr;
static llama_context* g_ctx     = nullptr;
static int            g_threads = 4;

bool llama_bridge_init(const char* model_path, int n_threads, int n_ctx) {
    g_threads = n_threads;

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;

    g_model = llama_model_load_from_file(model_path, mp);
    if (!g_model) { LOGE("Failed to load: %s", model_path); return false; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = (uint32_t)n_ctx;
    cp.n_threads       = (uint32_t)n_threads;
    cp.n_threads_batch = (uint32_t)n_threads;


    g_ctx = llama_init_from_model(g_model, cp);
    if (!g_ctx) {
        LOGE("Failed to create context");
        llama_model_free(g_model); g_model = nullptr;
        return false;
    }

#if defined(__ARM_FEATURE_BF16) || defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
    const char* bf16 = "YES";
#else
    const char* bf16 = "NO";
#endif
    LOGI("Model loaded. SME=%s | I8MM=%s | BF16=%s",
         ggml_cpu_has_sme()         ? "YES" : "NO",
         ggml_cpu_has_matmul_int8() ? "YES" : "NO",
         bf16);
    return true;
}


void llama_bridge_translate(const std::string& prompt,
                            std::function<void(const std::string&)> on_token) {
    if (!g_ctx || !g_model) return;
    llama_memory_clear(llama_get_memory(g_ctx), true);
    const llama_vocab* vocab = llama_model_get_vocab(g_model);

    // Tokenize
    std::vector<llama_token> toks(prompt.size() + 64);
    int n = llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(),
                           toks.data(), (int)toks.size(), true, true);
    if (n < 0) {
        toks.resize(-n);
        n = llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(),
                           toks.data(), (int)toks.size(), true, true);
    }
    if (n <= 0) { LOGE("Tokenization failed"); return; }
    toks.resize(n);

    // ✅ REMOVED: llama_kv_cache_clear — not in this version, safe to skip
    
    // Prefill
    llama_batch batch = llama_batch_get_one(toks.data(), n);
    if (llama_decode(g_ctx, batch) != 0) {
        LOGE("Prefill failed"); return;
    }

    // Sampler chain
    auto* smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.90f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.60f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    char piece[256];
    for (int i = 0; i < 512; ++i) {
        llama_token tok = llama_sampler_sample(smpl, g_ctx, -1);
        if (llama_vocab_is_eog(vocab, tok)) break;

        int len = llama_token_to_piece(vocab, tok, piece, sizeof(piece), 0, true);
        if (len > 0) on_token(std::string(piece, len));

        llama_batch next = llama_batch_get_one(&tok, 1);
        if (llama_decode(g_ctx, next) != 0) break;
    }

    llama_sampler_free(smpl);
}

void llama_bridge_free() {
    if (g_ctx)   { llama_free(g_ctx);         g_ctx   = nullptr; }
    if (g_model) { llama_model_free(g_model);  g_model = nullptr; }
}
