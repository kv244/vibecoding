# OpenCL Audio Effects Engine (CLFX) v1.0.2

[![Build & Test](https://github.com/kv244/vibecoding/actions/workflows/build.yml/badge.svg)](https://github.com/kv244/vibecoding/actions/workflows/build.yml)

A high-performance, hardened, and multiplatform-safe OpenCL audio processing engine for WAV files. Optimized for modern GPUs (including Intel Iris Xe) with vectorization, local memory caching, and zero-copy unified memory support.

## 🚀 Key Features

- **High-Performance Audio Effects**:
  - **Gain**: Adjustable signal multiplier with automatic clipping.
  - **Echo/Delay**: Parallelized one-shot reflection.
  - **Low-pass Filter**: Smoothed 3-tap FIR filter utilizing `__local` memory caching.
  - **Bitcrush**: Pre-computed bit-depth reduction for a classic "lo-fi" sound.
- **Hardware-Accelerated**: Full OpenCL 3.0 backend with `float4` vectorization.
- **Multiplatform Safe**: Dynamically queries hardware limits at runtime to ensure compatibility across different GPUs and CPUs (supports Windows, Linux, and Raspberry Pi 5).
- **Hardened Security**:
  - **Path Jailing**: Prevents directory traversal attacks via `os.path.realpath` containment.
  - **Type Safety**: Strictly validates all numeric parameters as floats in both frontend and backend.
  - **Execution Timeouts**: Proactive subprocess monitoring (120s limit) to prevent server hangs.

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

To compile the engine using GCC on Windows (Cygwin or MinGW):

```bash
# Ensure you have the OpenCL.dll path correct for your system
# This command uses the provides /include folder and links against the system OpenCL
gcc -Iinclude main.c -o clfx.exe C:\Windows\System32\OpenCL.dll -lm -std=c99 -D_POSIX_C_SOURCE=200112L
```

## 🧪 Generating Test Audio

You can use the following Python snippet to generate a stereo test WAV file (440Hz sine in Left, 880Hz in Right):

```python
import wave, struct, math

def generate_test_wav(filename='test_input.wav', duration_sec=1.0):
    sample_rate = 44100
    f = wave.open(filename, 'wb')
    f.setnchannels(2)
    f.setsampwidth(2)
    f.setframerate(sample_rate)
    
    num_samples = int(sample_rate * duration_sec)
    for i in range(num_samples):
        # Left: 440Hz, Right: 880Hz
        val_l = int(32767 * math.sin(2 * math.pi * 440 * i / sample_rate))
        val_r = int(32767 * math.sin(2 * math.pi * 880 * i / sample_rate))
        f.writeframesraw(struct.pack('<hh', val_l, val_r))
    f.close()
    print(f"Generated {filename}")

if __name__ == "__main__":
    generate_test_wav()
```

## 🎯 Usage

Run the executable with an input WAV file (uncompressed PCM, 16-bit) and specify the effect:

```bash
./clfx.exe <input.wav> <output.wav> <effect> [param1] [param2]
```

### Effects & Parameters:

## 📊 Visualization

CLFX provides two ways to visualize your audio waveforms:

### 1. Instant CLI Visualization
Use the `--visualize` flag after any effect to see a peak-amplitude ASCII bar chart in your terminal:
```bash
./clfx.exe input.wav output.wav gain 1.5 --visualize
```

### 2. High-Resolution Python Plotting
For detailed analysis, use the provided `visualize.py` script. It generates a high-quality `waveform.png` if `matplotlib` is installed, or falls back to a precise ASCII view.
```bash
python visualize.py output.wav
```

## 🖥️ Graphical Dashboard (GUI)

CLFX includes a professional-grade web dashboard for building effect chains visually. It features a modern dark-mode interface with glassmorphism, real-time parameter sliders, and automated waveform analysis.

### To Launch:
1. Ensure you have **Flask** installed:
   ```bash
   pip install flask
   ```
2. Navigate to the GUI directory and start the server:
   ```bash
   cd gui
   python server.py
   ```
3. Open your browser to `http://localhost:5000`.

## 🤖 CI/CD

Every push to `master` automatically triggers a full build and functional test suite via **GitHub Actions** on Ubuntu:
1. **Install**: Installs OpenCL runtime (`pocl-opencl-icd`)
2. **Compile**: Builds `clfx` from source using GCC
3. **Probe**: Runs `clfx --info` to verify the binary can detect platform/device info
4. **Generate**: Creates a synthetic stereo 440Hz WAV input using Python
5. **Test Core Effects**: Gain, Lowpass, Echo
6. **Test Advanced Effects**: Bitcrush, Spectral Freeze, Noise Gate
7. **Validate**: Confirms all output WAV files were written successfully

### Features:
- **Drag-and-Drop FX Rack**: Reorder your signal chain visually to control the processing path instantly.
- **Hardware Status Bar**: Real-time display of the host OS and active OpenCL device (GPU/CPU) via automated system probing (`--info` flag).
- **Secure Upload Zone**: Drag-and-drop WAV uploads with automated collision-resistant naming (UUID) and a 100MB safety limit.
- **Accessibility & Offline Ready**: WCAG-compliant ARIA labels and semantic structure. Zero network dependencies (uses high-performance system fonts).
- **Visual Feedback**: Processed audio waveforms are automatically rendered for peak-amplitude analysis.
- **Dynamic Controls**: Every effect parameter can be adjusted via high-fidelity, type-safe sliders.

| Effect | Parameter 1 | Parameter 2 | Example |
| :--- | :--- | :--- | :--- |
| `gain` | Multiplier | - | `gain 1.5` |
| `echo` | Delay (samples) | Decay (0.0 - 1.0) | `echo 4410 0.5` |
| `lowpass` | Strength (0.0 - 1.0) | - | `lowpass 0.8` |
| `bitcrush`| Bits (1 - 16) | - | `bitcrush 8` |
| `tremolo` | Frequency (Hz) | Depth (0.0 - 1.0) | `tremolo 5.0 0.8` |
| `widening`| Width (>1.0) | - | `widening 1.5` |
| `pingpong`| Delay (samples) | Decay (0.0 - 1.0) | `pingpong 8820 0.5` |
| `chorus`  | - | - | `chorus` |
| `autowah` | - | - | `autowah` |
| `distortion`| Drive (>1.0) | - | `distortion 5.0` |
| `ringmod` | Carrier Freq (Hz)| - | `ringmod 440` |
| `pitch`   | Ratio (0.5 - 2.0)| - | `pitch 1.5` |
| `gate`    | Threshold (0.0 - 1.0)| Reduction (0.0 - 1.0) | `gate 0.1 0.0` |
| `pan`     | L/R Balance (-1 to 1)| - | `pan -0.5` |
| `eq`      | Center (0.0 - 1.0) | Gain | `eq 0.5 2.0` |
| `freeze`  | - | - | `freeze` |
| `convolve`| IR WAV Path | - | `convolve ir.wav` |
| `phase`   | Depth (0.0 - 1.0) | Rate (0.0 - 1.0) | `phase 0.5 0.2` |
| `compress`| Threshold (0.0 - 1.0)| Ratio (1 - 20) | `compress 0.5 4` |
| `reverb`  | Size (0.0 - 1.0)  | Mix (0.0 - 1.0) | `reverb 0.6 0.5` |
| `flange`  | Depth (0.0 - 1.0) | Feedback (0.0 - 1.0) | `flange 0.5 0.7` |

- **Mapping Safety**: Implements explicit `clEnqueueUnmapMemObject` for consistent command queue synchronization.

## ☁️ Deploying to Google Cloud (GCP)

CLFX includes a production-ready Dockerfile and a GitHub Actions workflow for zero-downtime, serverless deployment to **Google Cloud Run**. The container compiles the engine natively and uses `pocl` for CPU-based software OpenCL.

### Deployment Setup

1. **Enable GCP APIs**: Ensure you have enabled the **Cloud Run API**, **Artifact Registry API**, and **Cloud Build API**.
2. **Create the Artifact Registry**:
   ```bash
   gcloud artifacts repositories create clfx-repo \
     --repository-format=docker \
     --location=us-central1
   ```
3. **Configure GitHub Auth**:
   Create a Google Cloud Service Account with **Cloud Run Admin** and **Artifact Registry Writer** permissions. Generate a JSON key and save it as a repository secret named `GCP_CREDENTIALS` in GitHub Settings > Secrets and variables > Actions.

> [!NOTE]
> The `deploy-cloudrun` job in `.github/workflows/build.yml` is **commented out by default** to prevent the CI pipeline from failing when the repository is forked or when the required authentication secrets are not yet configured. Once your GCP project and secrets are ready, simply uncomment the job to enable automatic zero-downtime deployments.

## 🔬 Implementation Details

### Autowah (Modulated Filter)
The `autowah` effect uses a modulated 3-tap FIR filter approximation. It shifts between lowpass and highpass characteristics using an internal 0.6Hz LFO to create the sweep. High performance is maintained via `__local` memory caching.

### Advanced Spectral Processing
CLFX now supports frequency-domain processing through a self-contained **Radix-2 FFT/IFFT** implementation in the OpenCL kernel.
- **FFT Size**: Optimized for 1024-point blocks.
- **Spectral EQ**: Direct manipulation of frequency bins.
- **Spectral Freeze**: Transient smearing achieved through phase randomization.
- **Convolution**: Frequency-domain pointwise multiplication (efficiently handles 1024-sample blocks).

> [!TIP]
> Spectral effects force a `local_size` of 256 in `main.c` to guarantee full coverage of the 1024-point FFT window (256 work-items * 4 samples per item), ensuring maximum hardware utilization on modern GPUs.

---
**Version:** 1.0.1 | **Last Commit:** 2026-02-21
Developed for high-performance audio experimentation.
