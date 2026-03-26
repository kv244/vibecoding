# vibecoding

A personal laboratory repository — high-performance systems engineering meets AI-assisted rapid prototyping.

## Structure

```
vibecoding/
├── clfx/          # OpenCL GPU-accelerated audio effects engine (CLFX v1.0.2)
└── experiments/   # Standalone vibe-coded experiments
    ├── gpu/       # CUDA / GPU compute
    ├── ai/        # Multimodal AI
    ├── embedded/  # BeagleV-Fire (RISC-V) & Pimoroni Presto (RP2350)
    └── creative/  # Graphics, music, interactive toys
```

## Projects

### [`clfx/`](./clfx) — CLFX Audio Engine
[![Build & Test](https://github.com/kv244/vibecoding/actions/workflows/build.yml/badge.svg)](https://github.com/kv244/vibecoding/actions/workflows/build.yml)

A high-performance OpenCL audio processing engine with 21 GPU-accelerated effects (gain, echo, reverb, convolution, pitch shift, chorus, compressor, and more). Includes a web dashboard GUI, Docker support, and Google Cloud Run deployment.

See [`clfx/README.md`](./clfx/README.md) for full documentation.

### [`experiments/gpu/`](./experiments/gpu)
| File | Description |
|---|---|
| `cube.py` | 3D rotating cube renderer with auto-selected backend (CUDA → Numba → NumPy) |
| `encrypt.cu` | GPU-accelerated file encryptor: ChaCha20-Poly1305 AEAD + Argon2id KDF, streaming chunks, directory mode |

See [`experiments/gpu/README.md`](./experiments/gpu/README.md) for build and usage instructions.

### [`experiments/ai/`](./experiments/ai)
| File | Description |
|---|---|
| `pygenai_mega.py` | Async Gemini 2.0 Flash webcam monitor with TTS, structured JSON output, and persistent scene history |

### [`experiments/embedded/`](./experiments/embedded)
| File | Description |
|---|---|
| `beaglev_dashboard.c` | ncurses system/CVE monitor for BeagleV-Fire (RISC-V inline ASM, pthread CVE scanner) |
| `image_gallery2_vibe.py` | MicroPython photo frame for Pimoroni Presto (hardware LFSR transitions, DMA LED aura) |

### [`experiments/creative/`](./experiments/creative)
| File | Description |
|---|---|
| `basic_interpreter.py` | Toy BASIC interpreter with 23+ commands and Matplotlib graphics |
| `harmonies.py` | Musical interval analysis — Lissajous curves, FFT, dyad playback |

---
*Bridging demanding performance work with AI-assisted creative development.*
