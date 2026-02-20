cmake_minimum_required(VERSION 3.22.1)
project(translator_native CXX C)

if(ANDROID)
    # Required for Android 15+ 16kb page size alignment
    add_link_options(-Wl,-z,max-page-size=16384)
endif()

# ── Dimensity 9400 / Cortex-X925 ARM feature flags ───────────────────────────
if(ANDROID_ABI STREQUAL "arm64-v8a")
    # Enable SME2, BF16, and i8mm
    set(ARM_FLAGS "-march=armv9.2-a+sme2+i8mm+bf16+dotprod")
    add_compile_options(${ARM_FLAGS})
    add_link_options(${ARM_FLAGS})
endif()

# ── llama.cpp ─────────────────────────────────────────────────────────────────
# Force the GGML backend to compile the KleidiAI SME2 micro-kernels
set(GGML_SME              ON  CACHE BOOL   "" FORCE)
set(GGML_CPU_ARM_ARCH     "armv9.2-a+sme2+i8mm+bf16" CACHE STRING "" FORCE)
set(GGML_NATIVE           OFF CACHE BOOL   "" FORCE)
set(LLAMA_BUILD_TESTS     OFF CACHE BOOL   "" FORCE)
set(LLAMA_BUILD_EXAMPLES  OFF CACHE BOOL   "" FORCE)
set(LLAMA_BUILD_SERVER    OFF CACHE BOOL   "" FORCE)
add_subdirectory(llama.cpp ${CMAKE_CURRENT_BINARY_DIR}/llama_build)

# ── whisper.cpp ───────────────────────────────────────────────────────────────
set(WHISPER_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(whisper.cpp ${CMAKE_CURRENT_BINARY_DIR}/whisper_build)

# ── JNI bridge ────────────────────────────────────────────────────────────────
add_library(translator_native SHARED
    pipeline_jni.cpp
    whisper_bridge.cpp
    llama_bridge.cpp
)

target_include_directories(translator_native PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(translator_native
    whisper
    llama
    log
)
