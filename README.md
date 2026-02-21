# OpenCL Audio Effects Engine (CLFX)

A high-performance, hardened, and multiplatform-safe OpenCL audio processing engine for WAV files. Optimized for modern GPUs (including Intel Iris Xe) with vectorization, local memory caching, and zero-copy unified memory support.

## 🚀 Key Features

- **High-Performance Audio Effects**:
  - **Gain**: Adjustable signal multiplier with automatic clipping.
  - **Echo/Delay**: Parallelized one-shot reflection.
  - **Low-pass Filter**: Smoothed 3-tap FIR filter utilizing `__local` memory caching.
  - **Bitcrush**: Pre-computed bit-depth reduction for a classic "lo-fi" sound.
- **Hardware-Accelerated**: Full OpenCL 3.0 backend with `float4` vectorization.
- **Hardened Host Code**: Robust error handling, resource management, and WAV validation.
- **Multiplatform Safe**: Dynamically queries hardware limits at runtime to ensure compatibility across different GPUs and CPUs.

## 🛠️ Performance Optimizations

1. **Vectorization**: Uses `float4` types to process 4 samples simultaneously per work-item, maximizing SIMD hardware utilization.
2. **Unified Memory (Zero-Copy)**: Automatically detects integrated GPUs (like Intel Iris Xe) and uses `CL_MEM_USE_HOST_PTR` to eliminate unnecessary data copies between CPU and GPU.
3. **Local Memory Caching**: The low-pass filter captures a "tile" of samples into fast `__local` memory, reducing global memory bandwidth requirements.
4. **Adaptive Scaling**: Queries `CL_DEVICE_MAX_WORK_GROUP_SIZE` and passes it to the kernel via compiler flags (`-DTILE_SIZE=n`) to ensure the code never exceeds hardware constraints.
5. **CPU Precomputation**: Offloads complex math (like bitcrush level power calculations) to the CPU, keeping the GPU kernel lean and fast.

## 📦 Prerequisites

- **OpenCL Drivers**: Ensure your GPU/CPU drivers support OpenCL 3.0 or later.
- **C Compiler**: GCC (e.g., via Cygwin or MinGW) or MSVC.
- **OpenCL SDK**: Headers and libraries are required (a minimal `include` directory is provided in this repo for convenience).

## 🔨 Building

To compile the engine using GCC on Windows:

```bash
gcc -Iinclude main.c -o clfx.exe C:\Windows\System32\OpenCL.dll -lm -std=c99 -D_POSIX_C_SOURCE=200112L
```

## 🎯 Usage

Run the executable with an input WAV file (uncompressed PCM, 16-bit) and specify the effect:

```bash
./clfx.exe <input.wav> <output.wav> <effect> [param1] [param2]
```

### Effects & Parameters:

| Effect | Parameter 1 | Parameter 2 | Example |
| :--- | :--- | :--- | :--- |
| `gain` | Multiplier | - | `gain 1.5` |
| `echo` | Delay (samples) | Decay (0.0 - 1.0) | `echo 4410 0.5` |
| `lowpass` | Strength (0.0 - 1.0) | - | `lowpass 0.8` |
| `bitcrush`| Bits (1 - 16) | - | `bitcrush 8` |

## 🛡️ Hardening & Safety

- **Resource Safety**: Uses a `goto cleanup` pattern in `main.c` to guarantee that all OpenCL objects and file handles are released even on error.
- **Validation**: Validates WAV headers and audio formats before processing.
- **Probe Logic**: Includes `#ifndef TILE_SIZE` fallbacks in `kernel.cl` for safe initial hardware-probing builds.
- **Mapping Safety**: Implements explicit `clEnqueueUnmapMemObject` for consistent command queue synchronization.

---
Developed for high-performance audio experimentation.
