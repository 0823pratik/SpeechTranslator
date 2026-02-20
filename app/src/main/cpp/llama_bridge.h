#pragma once
#include <string>
#include <functional>

bool llama_bridge_init(const char* model_path, int n_threads, int n_ctx);
void llama_bridge_translate(const std::string& prompt,
                            std::function<void(const std::string&)> on_token);
void llama_bridge_free();
