A laboratory for high-performance low-level engineering, multimodal AI reasoning, and aesthetic hardware experiences. This project bridges the gap between "hardcore" performance (RISC-V assembly, NEON intrinsics, CUDA, double-buffered I/O) and the fluid, vision-driven "vibecoding" philosophy.

🛠 Tech Stack
Embedded & FPGA: BeagleV-Fire (PolarFire SoC / RISC-V 64), Pimoroni Presto (RP2350 / Cortex-M33).

AI & Vision: Google Gemini 2.0 Flash (Multimodal Reasoning), FFmpeg, pyttsx3.

Performance Engineering: ARM Thumb & RISC-V Inline Assembly, CUDA (C++17), NEON SIMD, OpenMP, Numba (JIT), and DMA-driven I/O.

Graphics & UI: Raylib (pyray), ncurses, PicoGraphics, and WS2812/SK6812 PIO Drivers.

🚀 Key Modules & Technical Elements
1. Multi-Backend 3D Renderer (cube.py)
A high-performance 3D engine that dynamically selects the fastest available processing path based on hardware capabilities.

CUDA Acceleration: Offloads rotation matrix construction and Lambertian face shading to the GPU via PyTorch tensors.

CPU JIT Optimization: Uses Numba (@njit) to compile vertex projection and normal calculations into machine code, bypassing Python's interpreter overhead.

NumPy Integration: Stores vertex coordinates and face indices in dense NumPy arrays for rapid memory hand-offs between CPU and GPU.

Z-Sorting & Shading: Implements a manual Z-order sort on the CPU for depth testing and computes real-time face highlights using vector dot products.

2. AI Sidekick & Scene Monitor (pygenai_mega.py)
An advanced desk assistant using multimodal AI to monitor environments and interact with the user.

Reasoning Engine: Uses Gemini 2.0 Flash with a strict JSON Schema to identify risks, specific objects, and interesting facts within a webcam frame.

Spatial & Temporal Tracking: Maintains a persistent JSON history to answer contextual questions like "Where is my coffee?" or "What happened at 8 PM?".

Proactive Alerts: Integrated thread-safe TTS (pyttsx3) for audio alerts regarding identified hazards like fire, smoke, or intruders.

3. BeagleV Health & Fabric Dashboard (beaglev_dashboard.c)
An ncurses system monitor specifically tuned for the PolarFire SoC (RISC-V).

RISC-V ASM Optimizations: Uses inline assembly for digit parsing and uptime calculation using divu/remu instructions to avoid M-extension overhead.

Fabric Bridge Monitoring: Polls sysfs states for FIC Bridges (br0-br3) to diagnose AXI bus-fault risks in the FPGA fabric.

Security Scanner: Background worker thread that cross-references dpkg status with a local SQLite3 CVE database.

4. VibeVault: Hardened Encryption (fileEncryptRPI.c, encrypt.cu)
High-security file protection with massive hardware acceleration.

ARM NEON SIMD: Implements a 4-way interleaved AES-CTR kernel using NEON intrinsics (vaeseq_u8, vaesmcq_u8) for high-speed encryption on ARM64.

Industrial GPU Crypto: encrypt.cu leverages CUDA for ChaCha20-Poly1305 and Argon2id key derivation with streaming support for large files.

Dual-Core MicroPython: fileEncryptPresto.py uses double-buffering to overlap SD card I/O with 32-bit ARM Thumb assembly encryption kernels.

5. RP2350 Graphics & Transitions (image_gallery2_vibe.py)
Performance-tuned image gallery for the Pimoroni Presto (RP2350).

Hardware Interpolators: Accelerates "Mosaic" transitions using the RP2350 SIO Interpolators for high-speed 2D coordinate mapping.

PIO & DMA Aura: Ambient lighting is driven by PIO state machines and DMA loops, freeing CPU cores for JPEG decoding.

System Overclock: Automatically boosts CPU frequency to 250MHz for smooth image transitions.
