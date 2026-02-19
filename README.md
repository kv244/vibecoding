# vibecoding

Various AI-assisted coding experiments across embedded hardware, 3D graphics, language interpreters, cryptography, and AI computer vision. Every file in this repo was produced with AI pair-programming.

---

## Repository Structure

```
vibecoding/
├── cube.py                  # 3D rotating cube — Raylib + GPU/Numba/NumPy adaptive backend
├── basic_interpreter.py     # Full BASIC interpreter with graphics (Matplotlib)
├── image_gallery2_vibe.py   # MicroPython photo frame for Pimoroni Presto (RP2350)
├── encrypt.c                # RISC-V inline ASM XOR cipher — OpenMP parallel, shared lib
├── encrypt.py               # Python ctypes wrapper for encrypt.c / libbeagle_crypt.so
├── fileEncrypt.c            # RISC-V standalone file encryptor with inline ASM key cycling
├── fileEncrypt.py           # Pure Python equivalent of fileEncrypt.c
├── fileEncryptPresto.py     # MicroPython dual-core file encryptor for Pimoroni Presto
├── fileEncryptRPI.c         # ARM NEON SIMD file encryptor tuned for Raspberry Pi 5
├── beaglev_dashboard.c      # ncurses system monitor + CVE scanner for BeagleV-Fire
├── pygenai.py               # Gemini AI webcam monitor v1 — foundation
├── pygenai2.py              # Gemini AI webcam monitor v2 — smart model selection
├── pygenai3.py              # Gemini AI webcam monitor v3 — risk/reasoning engine
└── pygenai_mega.py          # Gemini AI webcam monitor v4 — full desk assistant
```

---

## File Breakdown

---

### `cube.py` — 3D Rotating Cube with Adaptive Compute Backend

- ![cube-cuda](https://github.com/user-attachments/assets/f4e4b646-4b87-43b3-a12a-5aa1a933678c)

**Purpose:** Real-time 3D solid-shaded cube rendered in a Raylib window, with automatic selection of the fastest available compute path at startup.

**Technical highlights:**

- **Three-tier backend detection.** On startup the script probes for PyTorch CUDA, then Numba JIT, then falls back to pure NumPy. The selected path is printed and shown in the window title (`3D Cube - Backend: CUDA / Numba / NumPy`). This means the same script runs optimally on a headless server with a GPU, a laptop, or a bare Python install with no extras.

- **Renderer: Raylib via `pyray`.** Unlike the original Pygame/Matplotlib approach, drawing goes through `pyray` (`init_window`, `begin_drawing`, `draw_triangle_fan`, `draw_line_v`). The render loop calls `draw_triangle_fan` with a list of `Vector2` projected points for each face, then draws edge outlines in a lighter shade of the same colour for depth definition.

- **Numba JIT hot path.** Three functions are decorated with `@njit`: `get_rotation_matrix_cpu`, `fast_project_cpu`, and `get_face_shading_cpu`. On first call Numba compiles these to native machine code. `get_face_shading_cpu` iterates faces in a tight `nopython` loop computing cross-product normals and dot products without any Python object overhead.

- **PyTorch GPU path.** When CUDA is available, vertices are moved to the GPU as `torch.float32` tensors at startup (`vertices_torch`, `faces_torch`, `light_dir_torch`). `get_rotation_matrix_torch` builds the rotation matrix from stacked `torch.cos`/`torch.sin` tensors and combines them with `torch.matmul`. `get_face_shading_torch` computes normals via `torch.cross` across all faces simultaneously (vectorised over the face dimension), then dots against the light vector with a single `torch.matmul`. Painter's Algorithm Z-sorting transfers only a tiny `(num_faces,)` float array back to CPU per frame.

- **Flat shading + Painter's Algorithm.** Each face gets a scalar intensity from `dot(normal, light_dir)`, clamped to a minimum of 0.1 to avoid fully black faces. Faces are sorted back-to-front by average Z before drawing. Face colour is `(100s, 150s, 255s)` where `s` is the shading scalar, giving a blue-toned lit appearance. Edge colour is the face colour brightened by 50 on each channel.

- **Interactive controls.** `KEY_UP`/`KEY_DOWN` increment/decrement `dax` (X angular velocity) by 0.005 rad/frame; `KEY_LEFT`/`KEY_RIGHT` do the same for `day`. `KEY_SPACE` zeros all three velocity components.

- **Dependency stack:** `pyray`, `numpy`. Optional: `numba`, `torch` (with CUDA).

---

### `basic_interpreter.py` — Toy BASIC Interpreter with Graphics

**Purpose:** A complete BASIC dialect interpreter running in Python, using Matplotlib as a graphics backend and supporting a REPL, file load/save, and sprite collision.

**Technical highlights:**

- **Compiled regex pre-processing.** Ten regexes are compiled at module level (`RE_LET`, `RE_DIM`, `RE_IF_EQ`, `RE_VAR_STR`, `RE_INKEY`, `RE_DATA_SPLIT`, `RE_OPS_COMBINED`, `RE_RENUM_GOTO`, `RE_RENUM_THEN`). The expression preprocessor splits the input at string literal boundaries so that transformations (e.g. `=` → `==`, `AND`/`OR`/`NOT` → Python equivalents) are never applied inside quoted strings.

- **Command object dispatch.** Each BASIC keyword maps to a command class instance stored in `self.cmds`. All have an `execute(interpreter, arg)` method. The main run loop does `cmd.execute(self, arg)` with no branching on keyword strings, making it easy to add new commands without touching the interpreter core.

- **Multi-dimensional arrays via `BasicArray`.** `DimCommand` creates a `BasicArray` with arbitrary dimensionality. The class pre-computes row-major multipliers at construction time (`multipliers[i] = multipliers[i+1] * bounds[i+1]`) so index-to-offset mapping is a single integer accumulation with no per-access division.

- **O(1) line jumps.** Before execution, `run()` builds `self.line_map = {line_number: index}`. `GOTO`/`GOSUB`/`IF` targets are resolved in O(1) via dict lookup rather than scanning the sorted line list.

- **Pre-parsed program list.** `run()` iterates the source once to build `self.program` (a list of `(command_object, arg_string)` pairs) and to collect all `DATA` values into `self.data_values`. The execution loop then works entirely off this pre-built list with no string parsing per tick.

- **`GOSUB`/`RETURN` call stack.** `GosubCommand.execute` pushes the next line number onto `interpreter.return_stack` before jumping. `ReturnCommand` pops it. Nested subroutines work correctly to arbitrary depth.

- **Matplotlib graphics.** `PLOT x,y` → `ax.plot([x],[y], 'k.', markersize=1)`. `DRAW x,y` → `ax.plot([last_x, x], [last_y, y], 'k-')`, updating `last_plot_pos`. `CIRCLE x,y,r` → `ax.add_patch(Circle(...))`. The canvas runs in non-blocking interactive mode (`plt.show(block=False)`) so the interpreter loop continues while the window is visible.

- **Sprite system with AABB collision.** `SpriteCommand` stores Matplotlib `Line2D` objects keyed by integer sprite ID. The built-in `COLLIDE(id1, id2)` function reads sprite positions via `get_data()` and returns 1 if the Euclidean distance is under 15 units.

- **`RENUM` with regex.** The `renum()` method remaps all line numbers to multiples of 10, then uses `RE_RENUM_GOTO` and `RE_RENUM_THEN` regex substitutions to update all `GOTO`, `GOSUB`, and `THEN` targets in the source, handling edge cases where a target is not found by leaving it unchanged.

- **`INKEY$` and `INPUT$`.** On Windows, `msvcrt.getch()` provides non-blocking single-key reads. On POSIX, `sys.stdin.read(1)` is used as a fallback.

- **Dependency stack:** `matplotlib`, `pillow`. Optional: `msvcrt` (Windows only, for `INKEY$`).

---

### `image_gallery2_vibe.py` — MicroPython Photo Frame for Pimoroni Presto (RP2350)

**Purpose:** A production-quality, touch-enabled JPEG slideshow for the Pimoroni Presto, using RP2350 hardware acceleration throughout for transitions, LED effects, and coordinate mapping.

**Technical highlights:**

- **PIO-driven WS2812 LED bar.** The `ws2812_aura` function is a `@rp2.asm_pio` program that bit-bangs the WS2812 protocol directly from a PIO state machine (`PIO_SM_ID = 4`, PIO1) at 8 MHz. Timing constants `T1=2, T2=5, T3=3` encode the standard WS2812 high/low bit pulse widths in PIO clock cycles.

- **Zero-CPU LED animation via DMA.** `start_background_aura()` programs DMA channel 11 to loop-transfer `self.breath_buffer` (a pre-computed 64-step sine-wave brightness animation, packed as `GGRRBB << 8` GRB words) directly into the PIO TX FIFO (`PIO_TX_FIFO`). The DMA control word encodes `DREQ_PIO1_TX0=8` as the data request signal and chains to itself for autonomous looping. The CPU is completely uninvolved once DMA starts — JPEG decoding and transitions run concurrently with the LED animation.

- **RP2350 SIO Interpolator for mosaic coordinate mapping.** `mosaic_transition()` configures Interpolator 0 (registers at `0xd0000000 + 0x80`) with lane masks to extract X and Y grid indices from a flat block index by writing to `INTERP0_ACCUM0` and reading back column and row from `INTERP0_PEEK0` / `INTERP0_PEEK1`. This replaces integer division and modulo for each block with a single hardware register read.

- **Cortex-M33 FPU and DSP inline assembly.** Two `@micropython.asm_thumb` functions expose hardware instructions: `asm_usat8_add` uses the `usat` instruction (unsigned saturate to 8 bits) for clamp-free brightness addition; `asm_f32_scale` uses `vmov`, `vcvt_f32_s32`, `vmul`, `vcvt_s32_f32` to perform a float multiply entirely in the FPU register file, used for HSV-to-RGB scaling.

- **Viper-compiled LFSR for FizzleFade.** `_calculate_lfsr()` is decorated `@micropython.viper`, compiling it to machine code with typed integers. The LFSR uses tap `0x20400` (an 18-bit maximal-length sequence), visiting every pixel coordinate exactly once before repeating — the same technique as Wolfenstein 3D's screen wipe. `fizzlefade()` runs in batches of 12,000 LFSR steps between `presto.update()` calls to remain responsive.

- **Double-buffered layer compositing.** Layer 0 holds the incoming image (decoded once via `jpegdec`). Layer 1 holds the outgoing image and UI overlays. Transitions work by painting Layer 1 transparent (pen index 0) in various patterns, revealing Layer 0 underneath. This avoids re-decoding and cuts transition CPU cost significantly.

- **Six transition modes chosen at random:** `fizzlefade`, `scroll_left`, `scroll_right`, `blinds`, `mosaic`, `curtain`. Each is a separate method on `VisualEffects`. `fast=True` uses larger step sizes for auto-advance transitions.

- **Auto-adaptive backlight.** If an LTR559 ambient light sensor is present on the I2C bus, `lux` is mapped linearly to backlight intensity (0.05 at dark, 1.0 at 100 lux+). Without the sensor, time-of-day fallback dims to 10% between 22:00 and 07:00. In dark mode, the DMA aura is also stopped.

- **SD card overclocked SPI.** The SD card SPI bus is initialised at 25 MHz (vs the standard 10–12 MHz), reducing JPEG load latency by approximately 30–40% on large files.

- **Hardware watchdog.** An 8-second `machine.WDT` is fed every main loop iteration. If the system hangs (e.g. during a corrupt JPEG decode), the watchdog resets the board.

- **Dependency stack:** MicroPython, Pimoroni Presto BSP (`presto`, `picographics`, `jpegdec`), optional `ltr559`.

---

### `encrypt.c` — RISC-V Parallel XOR Cipher (Shared Library)

**Purpose:** A minimal shared library (`libbeagle_crypt.so`) exposing a single function `beagle_crypt` that XORs an arbitrary byte buffer against a 64-bit key using RISC-V inline assembly, parallelised with OpenMP.

**Build:**
```bash
gcc -O3 -fopenmp -shared -fPIC encrypt.c -o libbeagle_crypt.so
```

**Technical highlights:**

- **OpenMP data parallelism.** The loop over 64-bit aligned blocks is annotated `#pragma omp parallel for schedule(static)`. With `schedule(static)`, OpenMP divides the block count evenly across available cores with no runtime scheduling overhead — appropriate for a uniform-cost loop body.

- **RISC-V inline ASM per block.** Each iteration emits three instructions: `ld t0, 0(ptr)` (load 64-bit word), `xor t0, t0, key` (XOR against the key register), `sd t0, 0(ptr)` (store back). The `"memory"` clobber informs the compiler that memory is modified through the pointer, preventing dead-store elimination. The `volatile` keyword on `asm volatile` prevents the entire block from being hoisted out of the loop.

- **64-bit granularity.** The function operates on `len_bytes / 8` 64-bit blocks. Any trailing bytes (partial block) are not processed — callers are expected to pad input to a multiple of 8, as the Python wrapper does.

- **Symbol cleanliness.** Compiled with plain `gcc` (not `g++`), so C name mangling is not applied and `ctypes.CDLL` can resolve `beagle_crypt` by its undecorated name.

---

### `encrypt.py` — Python ctypes Wrapper for libbeagle_crypt.so

**Purpose:** Demonstrates loading and calling the RISC-V shared library from Python, showing a complete encrypt-then-decrypt round trip.

**Technical highlights:**

- **`ctypes` ABI binding.** `crypt_lib.beagle_crypt.argtypes` is set to `[POINTER(c_uint64), c_size_t, c_uint64]`, matching the C prototype exactly. No `restype` is set (defaults to `c_int`) since the function returns `void` — this is benign but could be set to `None` for correctness.

- **Key derivation from ASCII string.** The 8-character `alpha_key` string is UTF-8 encoded, truncated or zero-padded to 8 bytes, then reinterpreted as a little-endian `uint64` via `int.from_bytes(..., byteorder='little')`.

- **8-byte aligned buffer.** The message is padded to the next multiple of 8 bytes with zero bytes. A `ctypes.c_uint64` array is allocated (`(c_uint64 * (padded_len // 8))()`), and `ctypes.memmove` copies the message bytes in. This ensures the buffer is properly aligned for the 64-bit load/store in the ASM.

- **XOR involution demo.** `beagle_crypt` is called twice with the same key — first to encrypt, second to decrypt — demonstrating that XOR is its own inverse without any separate decrypt function.

---

### `fileEncrypt.c` — RISC-V Standalone File Encryptor (OpenCL + ASM)

**Purpose:** A high-performance encryption tool for RISC-V (BeagleBoard-V Fire) featuring hybrid hardware acceleration.

**Technical highlights:**
- **OpenCL Acceleration**: Uses a vectorized OpenCL kernel (`kernel.cl`) with `uchar16` operations for massive parallel throughput on supported hardware.
- **Robust Fallback**: Automatically detects OpenCL support at runtime. If unavailable, it falls back to a hand-optimized **RISC-V 64-bit ASM** XOR loop.
- **Multithreaded Pipeline**: Distributes file segments across multiple CPU cores via `pthread`, ensuring high efficiency for large files.
- **Security & Large Files**: Processes data in 1MB chunks to support files of any size. Implements secure key file handling with `O_NOFOLLOW` and Restricted permissions.

---

### `fileEncrypt.py` — Pure Python File Encryptor

**Purpose:** A portable Python equivalent of `fileEncrypt.c`, producing the same `.enc` + `.ky` file pair using the `secrets` module for key generation.

**Technical highlights:**

- **`secrets.token_bytes(32)`** for key generation. Unlike `rand()` in the C version, `secrets` uses the OS CSPRNG (`/dev/urandom` on Linux), making this version cryptographically suitable for key material.

- **XOR cipher as a generator expression.** `xor_cipher` returns `bytes([b ^ key[i % len(key)] for i, b in enumerate(data)])`. Python's `%` operator handles key cycling; the list comprehension is straightforward and avoids any third-party dependency.

- **Command-line interface.** `sys.argv` is used directly — `argv[1]` is `encrypt` or `decrypt`, `argv[2]` is the filename. The decrypt path reconstructs `filename.enc` and `filename.ky` from the base filename, matching the C tool's conventions.

- **Dependency stack:** Python standard library only (`os`, `sys`, `secrets`).

---

### `fileEncryptPresto.py` — MicroPython Dual-Core Encryptor (RP2350)

**Purpose:** Encrypts files on the Presto's SD card using a multi-layered hardware strategy — combining dual-core processing, TRNG hardware, and an enhanced UI.

**Technical highlights:**
- **Dual-Core Ping-Pong Pipeline**: Utilizes a double-buffer system where Core 0 handles SD card I/O while Core 1 executes the Thumb-2 XOR kernel, overlapping compute and disk access.
- **Multi-Source Entropy Mixer**: Generates unique keys by hashing the **RP2350 Hardware TRNG** output with system uptime jitter and ADC noise for maximum robustness.
- **Performance UI**: Features a real-time **scrolling speed graph** and **ETA (Time Remaining)** calculation on the Presto's display.
- **ASM Tail Handling**: The fast Thumb-2 kernel processes word-aligned data, while Core 0 handles any trailing bytes (1-3) in Python, ensuring zero data loss and no alignment faults.
- **Memory Security**: Explicitly zeroes out data buffers in RAM after processing (Memory Cleansing).

---

### `fileEncryptRPI.c` — ARM NEON SIMD Encryptor (Raspberry Pi 5)

**Purpose:** A production-hardened tool tuned for the Cortex-A76, using 128-bit NEON SIMD and multi-core parallelism.

**Technical highlights:**
- **NEON SIMD Vectorization**: Uses `arm_neon.h` intrinsics to process 32 bytes per iteration, yielding extreme throughput.
- **Multithreaded Load Balancing**: Dynamically distributes 32-byte blocks across CPU cores to ensure perfect saturation and avoid remainder overhead.
- **I/O & Compute Overlap**: Implements a double-buffering system (ping-pong) to hide disk latency behind CPU execution.
- **Hardware Truth (Entropy)**: Pulls keys directly from the **Broadcom Hardware RNG** (`/dev/hwrng`) with a safe fallback to `/dev/urandom`.
- **Hardened Security**: Includes `O_NOFOLLOW` symlink protection, `0600` key file permissions, and explicit RAM cleansing.

---

### `beaglev_dashboard.c` — ncurses System Monitor + CVE Scanner for BeagleV-Fire

**Purpose:** A full-screen terminal dashboard for the BeagleV-Fire (RISC-V / Microchip PolarFire SoC) showing system health, FPGA fabric bridge states, network connectivity, and a live CVE vulnerability scan against installed Debian packages — all with RISC-V inline assembly in hot paths.

**Build:**
```bash
gcc -O2 -march=rv64gc -o beaglev_dashboard beaglev_dashboard.c \
    -lncurses -lsqlite3 -lpthread
```

**Technical highlights:**

- **Three ncurses windows.** `sys_win` (8×30): CPU temperature from `thermal_zone0` (raw millidegrees / 1000), 1-minute load average, thread count, root disk usage, uptime. `fb_win` (16×62): FPGA manager state + four FIC bridge states + Pi 5 TCP probe. `cve_win` (remaining height): scrollable CVE findings list.

- **RISC-V inline ASM — uptime division.** `update_system_info()` computes uptime hours and minutes using `divu`/`remu` instructions directly, guaranteeing the M-extension divide instructions are emitted regardless of `-O` level. The U54 application cores on PolarFire SoC implement the M extension as single-cycle operations.

- **RISC-V inline ASM — digit parsing.** `parse_digits()` implements the shift-add multiply trick: `acc*10 = (acc<<3) + (acc<<1)`, computing the multiply without the M-extension `mul` instruction. This keeps `ver_cmp_numeric()` correct on M-absent cores while still being tight on M-capable ones.

- **RISC-V inline ASM — branchless severity ranking.** In `update_cve_log()`, the severity string's first byte is loaded with `lbu`, then three `seqz` instructions produce boolean flags `is_C`, `is_H`, `is_M`. The rank is computed branchlessly as `rank = 3 - 3*is_C - 2*is_H - is_M`, mapping CRITICAL→0, HIGH→1, MEDIUM→2, LOW→3. A static `cp_table[4]` maps rank to colour pair in O(1) with no `strcmp` calls.

- **FIC bridge diagnostics.** A static table `fic_bridges[4]` maps each of the four PolarFire AXI FIC bridges (FIC0–FIC3) to their sysfs state files and address windows (0x60000000, 0x70000000, AXI-Lite, DMA). Each bridge is individually checked via `access()` + `read_sysfs()`. A disabled bridge is flagged `<-- bus fault risk` in red — directly exposing the root cause of `0x60000000` bus faults that occur when the fabric is "operating" but bridges are still gated.

- **Non-blocking TCP probe.** The Pi 5 reachability check uses `fcntl(sock, F_SETFL, O_NONBLOCK)` before `connect()`, which returns immediately with `EINPROGRESS`. A 300ms `select()` then checks the write fd-set; `getsockopt(SO_ERROR)` distinguishes success from `ECONNREFUSED`. `SO_SNDTIMEO` is explicitly not used because it does not reliably affect `connect()` on Linux.

- **SQLite CVE scanner on a background pthread.** `scanner_thread()` opens `cve_database.db` in WAL mode, creates the schema if absent, bulk-imports installed packages from `/var/lib/dpkg/status` in a single `BEGIN`/`COMMIT` transaction, then JOINs against the `cves` table ordering by severity (SQL `CASE` expression). Only packages where `installed_ver < fixed_ver` (via `ver_cmp_numeric`) are flagged. All findings are prepended to a mutex-protected singly-linked list. The UI thread reads the list under the same mutex with `g_scan.scan_done` as a completion flag.

- **Key bindings:** `q` quit, `r` rescan (joins previous thread first), `j`/`k` or arrow keys scroll the CVE list.

- **Dependency stack:** `ncurses`, `sqlite3`, `pthread`. Build target: `rv64gc`.

---

### `pygenai.py` — Gemini AI Webcam Monitor v1 (Foundation)

**Purpose:** The simplest version — captures a webcam frame with `ffmpeg` and sends it to Gemini for natural-language scene description.

**Technical highlights:**

- **`ffmpeg` subprocess for capture.** `subprocess.run` invokes `ffmpeg -f v4l2 -i /dev/video0 -frames:v 1 snapshot.jpg -y`. Using ffmpeg as a subprocess rather than `cv2.VideoCapture` avoids the OpenCV dependency entirely and works identically across V4L2 Linux devices.

- **`genai.list_models()`** is called at startup and all available model names are printed, useful for discovering what's available on the account before hardcoding a model string.

- **Model:** `models/gemini-pro-latest`. Frame is opened with `PIL.Image.open` and passed as the second element of the `generate_content` list alongside the prompt string `"What's happening?"`.

- **5-second polling loop** via `time.sleep(5)`.

- **Dependency stack:** `google-generativeai`, `Pillow`, `ffmpeg` (system binary).

---

### `pygenai2.py` — Gemini AI Webcam Monitor v2 (Smart Model Selection)

**Purpose:** Extends v1 with automatic model selection that prefers Gemini Flash variants for lower latency.

**Technical highlights:**

- **Runtime model selection.** `genai.list_models()` is enumerated and the first model whose name contains `"gemini-2.5-flash"` is selected. If none is found, the first model in the list is used as fallback. This means the script automatically adapts to new model releases without code changes.

- **`-loglevel error`** added to the ffmpeg command, suppressing verbose codec output that cluttered v1's terminal.

- **Prompt change:** `"Describe this webcam frame:"` instead of `"What's happening?"` — slightly more directive, encouraging a fuller description.

- **Dependency stack:** `google-generativeai`, `Pillow`, `ffmpeg`.

---

### `pygenai3.py` — Gemini AI Webcam Monitor v3 (AI Reasoning Engine)

**Purpose:** A major architectural upgrade. Transforms webcam analysis into a structured reasoning pipeline with JSON schema enforcement, tiered risk assessment, temporal memory, location awareness, adaptive zoom, and audio alerts.

**Technical highlights:**

- **JSON schema enforcement via `response_mime_type`.** The `GenerativeModel` is initialised with `generation_config={"response_mime_type": "application/json"}`. The prompt also includes the full `RESPONSE_SCHEMA` dict serialised with `json.dumps`. This instructs Gemini to return valid JSON matching the schema, which is then `json.loads()`'d directly — no regex scraping of free-form text.

- **Response schema fields:** `description` (string), `zoom_recommendation` (int 100–500), `risks` (array of `{hazard, severity, action}`), `interesting_facts` (array of `{subject, fact, confidence}`), `event_summary` (one-sentence delta since last frame).

- **Temporal memory with `collections.deque(maxlen=10)`.** `self.history` accumulates `event_summary` strings from the last 10 frames. Each new prompt includes these as `CONTEXT (Past Events)`, giving the model short-term awareness of what changed — e.g. "door was open, now closed".

- **Runtime geolocation.** `_fetch_location()` calls `https://ipinfo.io/json` at startup via `urllib.request` and extracts `city`, `region`, `country`. This string is injected into every prompt so the model can contextualise observations with local daylight, weather, and time-of-day patterns.

- **`ffmpeg-python` binding.** v3 switches from `subprocess.run` to the `ffmpeg` Python library's fluent API: `.input(...).output(...).overwrite_output().run(capture_stdout=True, capture_stderr=True)`. This provides structured error reporting via `ffmpeg.Error`.

- **Pillow RGBA overlay compositing.** `_add_overlay` creates a fully transparent `RGBA` overlay image, draws coloured semi-transparent pill boxes for timestamp, risk alerts (red for CRITICAL/HIGH, yellow for MEDIUM), and a fact strip at the bottom. `Image.alpha_composite` merges overlay onto the captured frame and saves back to disk.

- **`winsound.Beep` audio alerts.** On Windows, critical/high risks trigger `winsound.Beep(1000, 500)` (1 kHz, 500ms); lower risks trigger `winsound.Beep(500, 200)`. Gracefully absent on non-Windows via `try/except ImportError`.

- **Adaptive software zoom.** If `zoom_recommendation` differs from `self.current_zoom`, `v4l2-ctl --set-ctrl=zoom_absolute=N` is attempted via subprocess. The zoom level persists until the model recommends a change.

- **Async architecture.** `analyze_scene` is `async` and uses `loop.run_in_executor(None, ...)` to run the blocking Gemini API call in a thread pool, yielding the event loop while waiting.

- **Dependency stack:** `google-generativeai`, `Pillow`, `ffmpeg-python`, `winsound` (Windows only).

---

### `pygenai_mega.py` — Gemini AI Webcam Monitor v4 (Ultimate Desk Assistant)

**Purpose:** The fully-evolved version — a production-grade AI desk assistant with persistent memory, voice synthesis, interactive CLI querying, spatial object tracking, proactive automation hooks, and OCR.

**Technical highlights:**

- **`SceneMemory` — persistent JSON history.** Every analysis result is serialised as a `{timestamp, summary, objects, description, type}` entry and appended to `monitor_history.json` via `json.dump`. On startup, `_load()` reads the file back, so the assistant's context survives restarts. The last 5 entries are formatted as `"Recent Events:\n- [timestamp] summary\n..."` and injected into every prompt.

- **`AudioManager` — thread-safe TTS queue.** `pyttsx3` is not thread-safe, so a dedicated `threading.Thread` owns the engine instance. Text is pushed onto a `queue.Queue`; the worker thread pops items with a 1-second timeout (allowing clean shutdown via `_stop_event`). `engine.setProperty('rate', 150)` slows speech slightly for clarity. If `pyttsx3` is unavailable, `say()` falls back to `print`.

- **Async CLI via executor.** `user_interface_loop()` is an `async def` that calls `await loop.run_in_executor(None, input, "USER > ")`, wrapping the blocking `input()` call so it doesn't freeze the asyncio event loop. User queries are analysed with a fresh frame capture and the full memory context, then spoken via `AudioManager`.

- **Extended response schema.** Adds `objects_seen` (array of `{name, location, state}`) for spatial object tracking, and `suggested_automation` (array of `{trigger, webhook_url, reason}`) for proactive automation. The automation entries are logged; the actual `requests.post` call is commented out, making it easy to enable IFTTT/Home Assistant integration.

- **Multi-platform camera fallback.** `_capture_frame` first tries Linux V4L2. If that fails, it iterates a list of Windows DirectShow device names (`"video=Integrated Camera"`, `"video=USB Video Device"`, `"video=HD Web Camera"`) with `ffmpeg -f dshow`, each with a 5-second `timeout` to avoid hanging on unavailable devices.

- **OCR via prompt instruction.** The prompt includes: `"If the user is holding a document, book, or screen, EXTRACT the text or summarize it."` The model's `description` field carries the extracted content, which is saved to history with `type: "query_response"` for later retrieval via the CLI.

- **Overlay with `draw_pill` helper.** `_add_overlay` uses a local `draw_pill(x, y, text, color)` closure that calculates the text bounding box via `draw.textbbox`, draws a padded rounded rectangle, then draws the text — producing self-sizing label pills without hardcoded widths.

- **Background asyncio task for CLI.** `asyncio.create_task(self.user_interface_loop())` runs the CLI concurrently with the main analysis loop. Both tasks share the same event loop, the same model instance, and the same memory object without any additional locking because asyncio is single-threaded (only `AudioManager` crosses thread boundaries).

- **Dependency stack:** `google-generativeai`, `Pillow`, `ffmpeg-python`, `requests`, `pyttsx3` (optional), `winsound` (optional, Windows only).

---

## Setup

```bash
# 3D cube
pip install pyray numpy numba torch

# BASIC interpreter
pip install matplotlib pillow

# BeagleV dashboard (build on device)
gcc -O2 -march=rv64gc -o beaglev_dashboard beaglev_dashboard.c \
    -lncurses -lsqlite3 -lpthread

# RISC-V cipher shared library
gcc -O3 -fopenmp -shared -fPIC encrypt.c -o libbeagle_crypt.so

# File encryptor (RISC-V / BeagleBoard-V Fire)
# Build with OpenCL:
gcc -DUSE_OPENCL fileEncrypt.c -o fileEncrypt -lOpenCL -pthread
# Build with ASM fallback:
gcc fileEncrypt.c -o fileEncrypt -pthread

# File encryptor (Raspberry Pi 5)
gcc -O3 fileEncryptRPI.c -lpthread -o fileEncryptRPI

# Gemini webcam experiments
pip install google-generativeai pillow ffmpeg-python requests pyttsx3

# ffmpeg system binary (required for all webcam scripts)
sudo apt install ffmpeg       # Debian/Ubuntu
brew install ffmpeg           # macOS
```

**Gemini scripts** require a `GOOGLE_API_KEY` environment variable:
```bash
export GOOGLE_API_KEY="your_key_here"
```

**Pimoroni Presto scripts** (`image_gallery2_vibe.py`, `fileEncryptPresto.py`) run on-device in MicroPython with the Presto firmware — no pip install needed.

---

*Built with AI assistance — experiments in human-AI collaborative development.*
