plugins {
    id("com.android.application") version "8.7.3"
    id("org.jetbrains.kotlin.android") version "2.1.10"
}

android {
    namespace = "com.example.speechtranslator"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.example.speechtranslator"
        minSdk = 28
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"

        ndk {
            abiFilters += "arm64-v8a"
        }

        externalNativeBuild {
            cmake {
                cppFlags += "-std=c++17 -march=armv9.2-a+sme+sme2+i8mm+bf16+dotprod -O3"
                cFlags   += "-march=armv9.2-a+sme+sme2+i8mm+bf16+dotprod -O3"
                arguments += listOf(
                    "-DANDROID_STL=c++_shared",
                    "-DGGML_KLEIDIAI=ON",
                    "-DGGML_NEON=ON",
                    "-DGGML_OPENMP=OFF",
                    "-DGGML_LLAMAFILE=OFF",
                    "-DGGML_SME=ON",
                    "-DGGML_NATIVE=OFF",
                    "-DGGML_CPU_ARM_ARCH=armv9.2-a+sme+sme2+i8mm+bf16"
                )
            }
        }
    }   // ← closes defaultConfig

    // ── android-level: locates CMakeLists.txt ────────────────────────────
    externalNativeBuild {
        cmake {
            path("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    buildFeatures { viewBinding = true }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions { jvmTarget = "17" }

    buildTypes {
        getByName("release") {
            signingConfig = signingConfigs.getByName("debug")
        }
    }
}   // ← closes android

dependencies {
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
    implementation("androidx.coordinatorlayout:coordinatorlayout:1.2.0")
}
