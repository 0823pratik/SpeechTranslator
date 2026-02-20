#pragma once
#include <string>

bool        whisper_bridge_init(const char* model_path, int n_threads);
std::string whisper_bridge_transcribe(const float* pcm, int n_samples, const char* lang);
void        whisper_bridge_free();
