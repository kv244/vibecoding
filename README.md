# Vibecoding

A collection of high-performance animations, vintage interpreters, and AI experiments.

## 🚀 Projects

### 🧊 [cube.py](cube.py)
**3D Wireframe Cube Animation**
A high-performance implementation of a rotating 3D cube using **Pygame**.
- **What’s interesting:** This script implements a polymorphic backend system. It can run on pure **NumPy**, use **Numba JIT** for native execution speeds on the CPU, or leverage **PyTorch** for GPU-accelerated tensor operations. The perspective projection is fully vectorized for maximum efficiency.

### 🕹️ [basic_interpreter.py](basic_interpreter.py)
**Toy BASIC Interpreter with Graphics**
A feature-rich BASIC interpreter written in Python.
- **What’s interesting:** It’s not just a parser; it includes support for subroutines (`GOSUB`/`RETURN`), data handling (`DATA`/`READ`), and even sprite-based collision detection. It uses **Matplotlib** as a graphics engine, allowing you to `PLOT`, `DRAW`, and `CIRCLE` directly from BASIC code. It even supports loading JPEG images into the background!

### 🖼️ [image_gallery2_vibe.py](image_gallery2_vibe.py)
**MicroPython Photo Frame for Presto**
A professional-grade image gallery designed for the **Pimoroni Presto**.
- **What’s interesting:** This script features highly optimized MicroPython code, including custom transitions like **FizzleFade** (using a Linear Feedback Shift Register for pseudo-random pixel scattering), horizontal blinds, and mosaic transitions. It manages images via a circular doubly linked list and includes a "breathing" LED effect using pre-calculated sine tables to avoid floating-point overhead in real-time.

### 👁️ AI Webcam Experiments
A series of evolutional scripts integrating Google's **Gemini AI** with live webcam feeds:
- [**pygenai.py**](pygenai.py): The foundation. Simple frame capture via `ffmpeg` analyzed by Gemini Pro.
- [**pygenai2.py**](pygenai2.py): Adds intelligent model selection, automatically preferring the faster **Gemini Flash** variants if available.
- [**pygenai3.py**](pygenai3.py): The most advanced version. It features **dynamic hardware zoom control** (via `v4l2-ctl`), automatic timestamp overlays using **Pillow**, and detailed analytical prompts that ask Gemini to interpret how different zoom levels affect context and composition.

---

## 🛠️ Setup & Requirements

Depending on which experiment you're running, you may need:
- `pygame`, `numpy`, `numba`, `torch` (for the cube)
- `matplotlib`, `pillow` (for BASIC and Gemini scripts)
- `google-generativeai` (for Gemini experiments)
- `ffmpeg` (for webcam capture)

---
*Maintained with ❤️ and Vibecoding.*
