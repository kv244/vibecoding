# Vibecoding

A collection of high-performance animations, vintage interpreters, and AI experiments.

## 🚀 Projects

### 🧊 [cube.py](cube.py)
**3D Solid Cube Animation**
A high-performance implementation of a rotating 3D cube with **Adaptive Backend Selection**.
- **What’s interesting:** Now features a modular backend system that automatically selects the fastest available path: **PyTorch (CUDA GPU)**, **Numba (CPU JIT)**, or **Pure NumPy**. Includes **Solid Face Rendering** with **Flat Shading** (dynamic lighting) and **Painter's Algorithm** for depth sorting. Features **Interactive Keyboard Controls** (Arrows for speed/direction, Space to stop).
- ![cube-cuda](https://github.com/user-attachments/assets/f4e4b646-4b87-43b3-a12a-5aa1a933678c)


### 🕹️ [basic_interpreter.py](basic_interpreter.py)
**Toy BASIC Interpreter with Graphics**
A feature-rich BASIC interpreter written in Python.
- **What’s interesting:** It’s not just a parser; it includes support for subroutines (`GOSUB`/`RETURN`), data handling (`DATA`/`READ`), and even sprite-based collision detection. It uses **Matplotlib** as a graphics engine, allowing you to `PLOT`, `DRAW`, and `CIRCLE` directly from BASIC code. It even supports loading JPEG images into the background!

### 🖼️ [image_gallery2_vibe.py](image_gallery2_vibe.py)
**MicroPython Photo Frame for Presto (RP2350)**
A professional-grade image gallery designed for the **Pimoroni Presto**.
- **What’s interesting:** This script features highly optimized MicroPython code, including custom transitions like **FizzleFade**, horizontal blinds, and mosaic transitions. 
- **Hardware Acceleration:** Uses the RP2350's **PIO (Programmable I/O)** state machines to drive the ambient LED bar with nanosecond precision.
- **Multithreading:** Offloads LED animations and feedback to **Core 1**, ensuring perfectly smooth lighting even during heavy JPEG decoding on the main core. 
- **Smart Power**: Features auto-adaptive backlight control using the onboard light sensor.

### 👁️ AI Webcam Experiments
A series of evolutional scripts integrating Google's **Gemini AI** with live webcam feeds:
- [**pygenai.py**](pygenai.py): The foundation. Simple frame capture via `ffmpeg` analyzed by Gemini Pro.
- [**pygenai2.py**](pygenai2.py): Adds intelligent model selection, automatically preferring the faster **Gemini Flash** variants if available.
- [**pygenai3.py**](pygenai3.py): The **AI Reasoning Engine**. This is a sophisticated monitor featuring:
    - **Risk Assessment**: Detects hazards (fire, intruders) with severity levels and suggested actions.
    - **Audio Alerts**: Triggers real-time **Windows Audio Cues** (beeps) based on risk severity.
    - **Enhanced Visual Overlays**: Semi-transparent background boxes for clear insight readability.
    - **Fact of Interest Engine**: Identifies specific objects and explains them.
    - **Temporal Memory & Time/Location Awareness**: Remembers past events and is aware of the current local time and physical location (City/Region) for superior context (e.g., night-time visibility, weather).
    - **Adaptive Zoom**: Dynamic hardware zoom orchestration via the AI's recommendations.
    - **Structured Data**: Uses JSON schemas for high-reliability AI-to-Code communication.
- [**pygenai_mega.py**](pygenai_mega.py): The **Ultimate AI Sidekick / Desk Assistant**. A full evolution featuring:
    - **Persistent Memory**: Saves all events to a `monitor_history.json` for long-term recall.
    - **Interactive Query Mode**: An asynchronous CLI allows you to "talk" to the memory. (e.g., "Where did I leave my mug?").
    - **Voice Synthesis (TTS)**: The AI speaks its findings and responses aloud via a thread-safe `pyttsx3` manager.
    - **Spatial Object Tracking**: Tracks specific object positions and states over time.
    - **Proactive Automation**: Can trigger external webhooks/IFTTT events based on visual conditions.
    - **Robust & Resilient**: Features asynchronous I/O, non-blocking camera detection, and thread-safe audio concurrency.
    - **OCR & Document Analysis**: Automatically detects and extracts text from books, papers, or screens held up to the camera.

---

## 🛠️ Setup & Requirements

Depending on which experiment you're running, you may need:
- `pygame`, `numpy`, `numba`, `torch` (for the cube)
- `matplotlib`, `pillow` (for BASIC and Gemini scripts)
- `google-generativeai` (for Gemini experiments)
- `ffmpeg` (for webcam capture)

---
*Maintained with ❤️ and Vibecoding.*
