# SpeechTranslator

Offline multilingual speech-to-speech translator for Android using Whisper (STT), Llama (translation), and MMS/VITS (TTS). Runs entirely on-device with ARM/SME2 optimizations for low-latency real-time use. [github](https://github.com/Mozer/talk-llama-fast)

Optimized for Vivo X300 & ARMv9 hardware.

## Features

- **Real-time Pipeline**: Mic → VAD → Whisper transcribe → Llama translate → TTS playback (pipelined, no blocking).
- **Sentence Streaming**: Llama tokens → immediate TTS flush at boundaries (`.`, `।`, etc.) + inter-sentence silence for natural flow.
- **Hardware Acceleration**: SME2, NEON, I8MM, BF16 via llama.cpp/whisper.cpp.
- **Queuing**: Drops stale utterances if busy; processes latest.
- **Pre-warm TTS**: Zero cold-start delay.
- **Precise Drain**: Polls AudioTrack head position—no fixed sleep underruns.

## Architecture

```
Mic chunks → VAD (3200+ samples) → Whisper (16kHz float)
                          ↓
                    Llama prompt: "Translate [src] to [tgt]: \"[text]\""
                          ↓ (tokens stream)
Sentence flush (.!?।,) → MMS TTS (22kHz) → AudioTrack (MODE_STATIC, polled drain)
```


## Setup

### Prerequisites
- Android Studio + NDK (r26+ for SME2).
- Ubuntu/Linux for JNI builds (user preference). [user-information]

### 1. Clone & Models
```
git clone <your-repo> SpeechTranslator
cd app/src/main/cpp
# Download:
# - ggml-tiny-q8_0.bin (or multilingual)
# - gemma-2-2b-it-Q4_K_M.gguf (or your Q4_K_M)
# - mms-*/tts_*.pt (MMS models to assets/models/)
```

### 2. JNI Build
```
# CMakeLists.txt handles whisper.cpp/llama.cpp submodules
ndk-build  # Or Android Studio "Build Native"
# Install libtranslator_native.so to jniLibs/arm64-v8a
```

### 3. Android Studio
```
# Update paths in MainActivity/PipelineManager:
pipeline.init("whisper-base.bin", "llama.gguf", "models/")
# Run on ARM64 device (Vivo X300 recommended)
```

## Usage

1. Set source/target lang (e.g., "hi" → "en").
2. Tap Record → Speak → Auto VAD ends → Pipeline runs.
3. Transcription → Translation streams → TTS plays.
4. Record re-enables post-drain (no mic/speaker overlap).



## Optimization Tips

- **SME2**: Auto-detected; enable in CMake (`-march=armv9-a+sme2`).
- **Quantization**: Use Q4_K_M / Q5_K for 3B Llama.
- **Ctx Flush**: `nativeLlamaClear()` post-generation prevents hallucinations. [prior]
- **Tuning**:
  ```kotlin
  INTER_SENTENCE_SILENCE_MS = 120L  // Adjust pauses
  CLAUSE_FLUSH_MIN = 50  // Avoid tiny TTS clips
  ```

**Logs**: `adb logcat -s PipelineManager WhisperBridge LlamaBridge`

## Files

```
app/
├── src/main/
│   ├── kotlin/.../PipelineManager.kt     # Core (Whisper→Llama→TTS)
│   ├── kotlin/.../MainActivity.kt        # UI + AudioCapture wiring
│   ├── cpp/
│   │   ├── translator_native.cpp         # JNI wrapper
│   │   ├── whisper_bridge.cpp/.h         # whisper.cpp bindings
│   │   ├── llama_bridge.cpp/.h           # llama.cpp bindings + KV clear
│   │   └── CMakeLists.txt                # NDK build
│   └── assets/models/                    # MMS TTS models
└── jniLibs/arm64-v8a/libtranslator_native.so
```

