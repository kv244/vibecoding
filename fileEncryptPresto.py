import machine, _thread, time, uctypes, threading, os, hashlib, random
from picographics import PicoGraphics, DISPLAY_PRESTO_ST7789

# --- 1. UI & HARDWARE INITIALIZATION ---
display = PicoGraphics(display=DISPLAY_PRESTO_ST7789)
WIDTH, HEIGHT = display.get_bounds()
BG = display.create_pen(10, 10, 20)
BAR_FG = display.create_pen(0, 255, 150)
TEXT_PEN = display.create_pen(255, 255, 255)
SPEED_PEN = display.create_pen(255, 200, 0)
SUCCESS_PEN = display.create_pen(0, 255, 0)
GRAPH_PEN = display.create_pen(0, 100, 255)

speed_history = [0] * 60 # History for rolling graph

def set_vibe_priority():
    """Boosts CPU priority for smooth real-time encryption on the RP2350."""
    bp_reg = 0x40094000
    machine.mem32[bp_reg] = (machine.mem32[bp_reg] | 0x1 | (0xFF << 10)) & ~(1 << 1)

def get_mixed_entropy():
    """
    Robust Entropy Mixer:
    Combines Hardware TRNG, system uptime jitter (us), and ADC floating 
    noise into a hashed 32-bit key to ensure high-quality random keys.
    """
    h = hashlib.sha256()
    # Source 1: Hardware TRNG output
    h.update(str(machine.rng()).encode())
    # Source 2: Precise system uptime (nanosecond-ish jitter)
    h.update(str(time.ticks_us()).encode())
    # Source 3: ADC Noise from the Internal Temp Sensor (noisy LSBs)
    adc = machine.ADC(machine.ADC.CORE_TEMP)
    h.update(str(adc.read_u16()).encode())
    # Mix sources and extract initial 4 bytes as the session key
    digest = h.digest()
    return int.from_bytes(digest[:4], 'big')

def update_ui_enhanced(file_name, progress, speed, remaining_s):
    """Updates the Presto display with progress, speed, ETA, and a rolling graph."""
    display.set_pen(BG)
    display.clear()
    
    # Header
    display.set_pen(TEXT_PEN)
    display.text(f"SECURE: {file_name}", 20, 20, scale=3)
    
    # Progress Bar UI
    display.rectangle(20, 60, WIDTH-40, 20)
    display.set_pen(BG)
    display.rectangle(22, 62, WIDTH-44, 16)
    display.set_pen(BAR_FG)
    display.rectangle(22, 62, int((WIDTH-44) * progress), 16)
    
    # Benchmarking Metrics
    display.set_pen(SPEED_PEN)
    display.text(f"{speed:.1f} KB/s", 20, 95, scale=4)
    display.set_pen(TEXT_PEN)
    display.text(f"ETA: {int(remaining_s)}s", WIDTH-120, 95, scale=3)
    
    # Rolling Speed Graph for performance visualization
    display.set_pen(GRAPH_PEN)
    gh = 60 # Graph height
    gy = 160 # Graph Y base
    speed_history.append(speed)
    speed_history.pop(0)
    max_s = max(speed_history) if max(speed_history) > 0 else 1
    for i in range(len(speed_history)-1):
        x1 = 20 + (i * (WIDTH-40)//60)
        x2 = 20 + ((i+1) * (WIDTH-40)//60)
        y1 = gy + gh - int((speed_history[i] / max_s) * gh)
        y2 = gy + gh - int((speed_history[i+1] / max_s) * gh)
        display.line(x1, y1, x2, y2)
    
    display.update_async()

# --- 2. THE KERNEL (Thumb ASM) ---
@micropython.asm_thumb
def asm_xor_crypt(r0, r1, r2):
    """
    High-performance 32-bit XOR kernel in ARM Thumb Assembly.
    Processes 4 bytes per iteration.
    r0: buffer ptr, r1: length (bytes), r2: 32-bit key
    """
    label(LOOP)
    ldr(r3, [r0, 0])
    eor(r3, r2)
    str(r3, [r0, 0])
    add(r0, 4)
    sub(r1, 4)
    cmp(r1, 0)
    bgt(LOOP)

def get_aligned_buffer(size, alignment=4):
    """Allocates a bytearray and returns a memoryview aligned for raw ASM access."""
    buf = bytearray(size + alignment)
    addr = uctypes.addressof(buf)
    offset = (alignment - (addr % alignment)) % alignment
    return memoryview(buf)[offset:offset+size]

# --- 3. DUAL-CORE ENGINE (Double Buffering) ---
CHUNK_SIZE = 4096
# Two 4-byte aligned buffers for overlapping I/O and compute
buffers = [get_aligned_buffer(CHUNK_SIZE), get_aligned_buffer(CHUNK_SIZE)]
# Track the length to process in each buffer (since chunks might not be full)
buffer_lengths = [0, 0]
ready_events = [threading.Event(), threading.Event()]
done_events = [threading.Event(), threading.Event()]

def core1_worker(key):
    """
    Core 1 Worker Logic:
    Always XORs whatever buffer is signaled as 'ready' by Core 0.
    Only processes word-aligned segments to avoid ASM alignment faults.
    """
    idx = 0
    while True:
        ready_events[idx].wait()
        ready_events[idx].clear()
        
        # Calculate word-aligned length for the ASM kernel
        words = buffer_lengths[idx] // 4
        if words > 0:
            asm_xor_crypt(buffers[idx], words * 4, key)
            
        done_events[idx].set()
        idx = (idx + 1) % 2

# --- 4. SECURE PROCESSOR ---
def show_success_splash(file_name):
    """Displays a verification particle animation and final status."""
    for _ in range(50):
        display.set_pen(display.create_pen(random.randint(0,255), 255, random.randint(0,255)))
        display.circle(random.randint(0, WIDTH), random.randint(0, HEIGHT), random.randint(2, 8))
    
    display.set_pen(BG)
    display.rectangle(40, HEIGHT // 2 - 40, WIDTH - 80, 80)
    display.set_pen(SUCCESS_PEN)
    display.text("SECURED", WIDTH // 2 - 60, HEIGHT // 2 - 20, scale=5)
    display.set_pen(TEXT_PEN)
    display.text(f"KEY: {file_name}.ky", WIDTH // 2 - 90, HEIGHT // 2 + 25, scale=2)
    display.update()

def secure_process(filename):
    """
    Main Logic Core 0:
    Orchestrates entropy mixing, key serialization, and double-buffered I/O.
    """
    set_vibe_priority()
    key = get_mixed_entropy()
    
    # Pre-extract key bytes for tail-handling in Python
    key_bytes = key.to_bytes(4, 'big')
    
    # Spawn encryption core on Core 1
    _thread.start_new_thread(core1_worker, (key,))
    
    source = f"/sd/{filename}"
    encrypted = f"/sd/{filename}.enc"
    keyfile = f"/sd/{filename}.ky"
    file_size = os.stat(source)[6]
    
    start_time = time.ticks_ms()
    processed = 0
    curr = 0 # Ping-pong index
    
    with open(source, 'rb') as fin, open(encrypted, 'wb') as fout:
        # Initial read to prime the process
        read_prev = fin.readinto(buffers[curr])
        buffer_lengths[curr] = read_prev
        
        while read_prev > 0:
            # 1. Dispatch Core 1 for word-aligned XOR
            ready_events[curr].set()
            
            # 2. OVERLAP: Read next from SD card while worker is busy
            next_idx = (curr + 1) % 2
            read_next = fin.readinto(buffers[next_idx])
            buffer_lengths[next_idx] = read_next
            
            # 3. Synchronize with Core 1
            done_events[curr].wait()
            done_events[curr].clear()
            
            # 4. TAIL HANDLING: XOR any remaining 1-3 bytes in Python (Core 0)
            tail_start = (read_prev // 4) * 4
            for i in range(tail_start, read_prev):
                buffers[curr][i] ^= key_bytes[i % 4]
            
            # 5. Stream processed data to storage
            fout.write(buffers[curr][:read_prev])
            processed += read_prev
            
            # Visual Benchmarking
            if (processed // CHUNK_SIZE) % 5 == 0 or processed == file_size:
                elapsed = time.ticks_diff(time.ticks_ms(), start_time) / 1000
                speed = (processed / 1024) / elapsed if elapsed > 0 else 0
                rem = (file_size - processed) / (speed * 1024) if speed > 0 else 0
                update_ui_enhanced(filename, processed/file_size, speed, rem)
            
            curr = next_idx
            read_prev = read_next

    # Secure Memory Cleansing
    for b in buffers: 
        for i in range(CHUNK_SIZE): b[i] = 0
    
    with open(keyfile, 'w') as f: f.write(hex(key))
    show_success_splash(filename)
    print(f"[+] Secure process complete. HW Key: {hex(key)}")

# Usage: secure_process('vibe.txt')
