#include <jni.h>
#include <string>
#include "whisper_bridge.h"
#include "llama_bridge.h"
#include "llama.cpp/ggml/include/ggml-cpu.h"

// ── Whisper ───────────────────────────────────────────────────────────────────

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_speechtranslator_PipelineManager_nativeWhisperInit(
        JNIEnv* env, jobject, jstring path_j, jint threads) {
    const char* p = env->GetStringUTFChars(path_j, nullptr);
    bool ok = whisper_bridge_init(p, (int)threads);
    env->ReleaseStringUTFChars(path_j, p);
    return (jboolean)ok;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_speechtranslator_PipelineManager_nativeWhisperTranscribe(
        JNIEnv* env, jobject, jfloatArray pcm_j, jstring lang_j) {
    jsize   len  = env->GetArrayLength(pcm_j);
    jfloat* pcm  = env->GetFloatArrayElements(pcm_j, nullptr);
    const char* lang = env->GetStringUTFChars(lang_j, nullptr);
    std::string result = whisper_bridge_transcribe(pcm, (int)len, lang);
    env->ReleaseFloatArrayElements(pcm_j, pcm, JNI_ABORT);
    env->ReleaseStringUTFChars(lang_j, lang);
    return env->NewStringUTF(result.c_str());
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_speechtranslator_PipelineManager_nativeWhisperFree(
        JNIEnv*, jobject) {
    whisper_bridge_free();
}

// ── Llama ─────────────────────────────────────────────────────────────────────

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_speechtranslator_PipelineManager_nativeLlamaInit(
        JNIEnv* env, jobject, jstring path_j, jint threads, jint n_ctx) {
    const char* p = env->GetStringUTFChars(path_j, nullptr);
    bool ok = llama_bridge_init(p, (int)threads, (int)n_ctx);
    env->ReleaseStringUTFChars(path_j, p);
    return (jboolean)ok;
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_speechtranslator_PipelineManager_nativeLlamaTranslate(
        JNIEnv* env, jobject, jstring prompt_j, jobject cb_obj) {
    const char* pc = env->GetStringUTFChars(prompt_j, nullptr);
    std::string prompt(pc);
    env->ReleaseStringUTFChars(prompt_j, pc);

    jclass    cls   = env->GetObjectClass(cb_obj);
    jmethodID onTok = env->GetMethodID(cls, "onToken", "(Ljava/lang/String;)V");

    llama_bridge_translate(prompt, [&](const std::string& tok) {
        jstring js = env->NewStringUTF(tok.c_str());
        env->CallVoidMethod(cb_obj, onTok, js);
        env->DeleteLocalRef(js);
    });
}

extern "C" JNIEXPORT void JNICALL
Java_com_example_speechtranslator_PipelineManager_nativeLlamaFree(
        JNIEnv*, jobject) {
    llama_bridge_free();
}

// ── Backend info ──────────────────────────────────────────────────────────────

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_speechtranslator_PipelineManager_nativeGetBackendInfo(
        JNIEnv* env, jobject) {

#if defined(__ARM_FEATURE_BF16) || defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
    const char* bf16 = "YES";
#else
    const char* bf16 = "NO";
#endif

    char buf[256];
    snprintf(buf, sizeof(buf),
             "SME=%s | NEON=%s | I8MM=%s | BF16=%s",
             ggml_cpu_has_sme()         ? "YES" : "NO",
#if defined(__ARM_NEON)
             "YES",
#else
            "NO",
#endif
             ggml_cpu_has_matmul_int8() ? "YES" : "NO",
             bf16
    );
    return env->NewStringUTF(buf);
}
