# Vibecoding

A laboratory for **high-performance low-level engineering**, **multimodal AI reasoning**, and **aesthetic hardware experiences**. This project bridges the gap between "hardcore" performance (RISC-V assembly, NEON intrinsics, CUDA, double-buffered I/O) and the fluid, vision-driven "vibecoding" philosophy.

## 🛠 Tech Stack
* **Embedded & FPGA:** BeagleV-Fire (PolarFire SoC / RISC-V 64), Pimoroni Presto (RP2350 / Cortex-M33).
* **AI & Vision:** Google Gemini 2.0 Flash (Multimodal Reasoning), FFmpeg, pyttsx3.
* **Performance Engineering:** ARM Thumb & RISC-V Inline Assembly, CUDA (C++17), OpenCL (GPGPU), NEON SIMD, OpenMP, Numba (JIT), and DMA-driven I/O.
* **Graphics & UI:** Raylib (pyray), ncurses, PicoGraphics, and WS2812/SK6812 PIO Drivers.

---

## 🚀 Key Modules & Technical Elements

### 1. Multi-Backend 3D Renderer (`cube.py`)
A high-performance 3D engine that dynamically selects the fastest available processing path based on hardware capabilities.
* **CUDA Acceleration:** Offloads rotation matrix construction and Lambertian face shading to the GPU via PyTorch tensors.
* **CPU JIT Optimization:** Uses Numba (`@njit`) to compile vertex projection and normal calculations into machine code.
* **NumPy Integration:** Stores vertex coordinates and face indices in dense NumPy arrays for rapid memory hand-offs between CPU and GPU.
* **Z-Sorting & Shading:** Implements a manual Z-order sort on the CPU for depth testing and computes real-time face highlights using vector dot products.

### 2. BeagleV Hybrid GPGPU Encryption (`fileEncrypt.c` & `kernel.cl`)
A high-throughput encryption suite for the BeagleV-Fire that leverages both multi-core RISC-V processing and GPGPU acceleration.
* **Parallel Execution Engine:** `fileEncrypt.c` manages a `pthread` pool to segment files into 1MB chunks for concurrent processing on the PolarFire SoC.
* **OpenCL Hardware Offloading:** The system dynamically initializes `kernel.cl`, spawning threads to offload intensive XOR operations to the FPGA/GPU fabric.
* **Vectorized Throughput:** `kernel.cl` processes 16 bytes per thread using the `uchar16` vector type and `vload16`/`vstore16` primitives to maximize memory bandwidth.
* **RISC-V ASM Fallback:** If hardware acceleration is unavailable, the system utilizes a hand-optimized RISC-V 64 assembly XOR loop for low-latency processing.


### 3. AI Sidekick & Scene Monitor (`pygenai_mega.py`)
An advanced desk assistant using multimodal AI to monitor environments and interact with the user.
* **Reasoning Engine:** Uses Gemini 2.0 Flash with a strict JSON Schema to identify risks, specific objects, and interesting facts within a webcam frame.
* **Spatial & Temporal Tracking:** Maintains a persistent JSON history to answer contextual questions like "Where is my coffee?".
* **Proactive Alerts:** Integrated thread-safe TTS (pyttsx3) for audio alerts regarding identified hazards like fire, smoke, or intruders.


### 4. BeagleV Health & Fabric Dashboard (`beaglev_dashboard.c`)
An ncurses system monitor specifically tuned for the **PolarFire SoC** (RISC-V).
* **RISC-V ASM Optimizations:** Uses inline assembly for digit parsing and uptime calculation using `divu`/`remu` instructions to avoid M-extension overhead.
* **Fabric Bridge Monitoring:** Polls sysfs states for **FIC Bridges (br0-br3)** to diagnose AXI bus-fault risks in the FPGA fabric.
* **Security Scanner:** Background worker thread that cross-references `dpkg` status with a local SQLite3 CVE database.


### 5. VibeVault: Hardened Encryption (`fileEncryptRPI.c`, `encrypt.cu`)
High-security file protection with massive hardware acceleration.
* **ARM NEON SIMD:** Implements a **4-way interleaved AES-CTR kernel** using NEON intrinsics (`vaeseq_u8`, `vaesmcq_u8`) for high-speed encryption on ARM64.
* **Industrial GPU Crypto:** `encrypt.cu` leverages CUDA for **ChaCha20-Poly1305** and **Argon2id** key derivation with streaming support for large files.

---
## 🏗 High-Performance Crypto: OpenCL vs. CUDA

The repository contains two distinct hardware-accelerated encryption paths. Comparing the BeagleV-Fire's OpenCL implementation with the desktop-class CUDA implementation reveals the architectural trade-offs between **portability** and **platform-specific depth**.

### **1. Implementation Comparison**

| Feature | OpenCL Implementation (`fileEncrypt.c` / `kernel.cl`) | CUDA Implementation (`encrypt.cu`) |
| :--- | :--- | :--- |
| **Primary Goal** | **Hardware Portability:** Designed to run across RISC-V CPUs, GPUs, and FPGA fabrics. | **Deep Optimization:** Designed for industrial-strength throughput on NVIDIA hardware. |
| **Vectorization Strategy** | **Explicit:** Uses `uchar16` vector types and `vload16`/`vstore16` to manually saturate memory lanes. | **Implicit (SIMT):** Each thread handles a single path; optimizations focus on `__shared__` memory usage. |
| **Abstraction Level** | **Low-Level Boilerplate:** Requires manual management of platforms, devices, contexts, and command queues. | **Integrated:** Host and device code share a single file/compiler for seamless integration. |
| **Execution Path** | **Hybrid Fallback:** Includes a hand-optimized RISC-V 64 assembly loop if the OpenCL context fails. | **Streaming Pipelined:** Uses a pinned memory windowing system to process massive files without exceeding VRAM. |



### **2. Architectural Insights**

#### **The OpenCL Model (BeagleV-Fire)**
Your OpenCL implementation demonstrates the "write once, run anywhere" philosophy. Because OpenCL must remain hardware-agnostic, the kernel in `kernel.cl` uses **explicit vectorization** to ensure that the underlying hardware (whether an FPGA fabric or a GPU) uses its full SIMD width. The host code in `fileEncrypt.c` reflects the complexity of the OpenCL API, requiring significant management to compile the kernel at runtime and dispatch it to the BeagleV's fabric.

#### **The CUDA Model (NVIDIA)**
The `encrypt.cu` implementation focuses on **computational density**. By leveraging CUDA's tight coupling with NVIDIA's hardware, it implements complex, memory-hard algorithms like **Argon2id** and **ChaCha20-Poly1305** with significantly less boilerplate than OpenCL. It uses GPU **shared memory** to cache keys and nonces, dramatically reducing global memory latency compared to the standard XOR cycling seen in the OpenCL kernel.


