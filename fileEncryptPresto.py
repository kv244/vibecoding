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
    bp_reg = 0x40094000
    machine.mem32[bp_reg] = (machine.mem32[bp_reg] | 0x1 | (0xFF << 10)) & ~(1 << 1)

def get_mixed_entropy():
    """Combines HW RNG, ADC noise, and uptime into a hashed 32-bit key."""
    h = hashlib.sha256()
    # Source 1: Hardware RNG
    h.update(str(machine.rng()).encode())
    # Source 2: Uptime Jitter
    h.update(str(time.ticks_us()).encode())
    # Source 3: ADC Noise (Floating Pin or Temp Sensor)
    adc = machine.ADC(machine.ADC.CORE_TEMP)
    h.update(str(adc.read_u16()).encode())
    # Mix and digest
    digest = h.digest()
    return int.from_bytes(digest[:4], 'big')

def update_ui_enhanced(file_name, progress, speed, remaining_s):
    display.set_pen(BG)
    display.clear()
    
    # Header
    display.set_pen(TEXT_PEN)
    display.text(f"SECURE: {file_name}", 20, 20, scale=3)
    
    # Progress Bar
    display.rectangle(20, 60, WIDTH-40, 20)
    display.set_pen(BG)
    display.rectangle(22, 62, WIDTH-44, 16)
    display.set_pen(BAR_FG)
    display.rectangle(22, 62, int((WIDTH-44) * progress), 16)
    
    # Metrics
    display.set_pen(SPEED_PEN)
    display.text(f"{speed:.1f} KB/s", 20, 95, scale=4)
    display.set_pen(TEXT_PEN)
    display.text(f"ETA: {int(remaining_s)}s", WIDTH-120, 95, scale=3)
    
    # Rolling Speed Graph
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
    label(LOOP)
    ldr(r3, [r0, 0])
    eor(r3, r2)
    str(r3, [r0, 0])
    add(r0, 4)
    sub(r1, 4)
    cmp(r1, 0)
    bgt(LOOP)

# --- 3. DUAL-CORE ENGINE (Double Buffering) ---
CHUNK_SIZE = 4096
# Two buffers for overlapping I/O and compute
buffers = [bytearray(CHUNK_SIZE), bytearray(CHUNK_SIZE)]
ready_events = [threading.Event(), threading.Event()]
done_events = [threading.Event(), threading.Event()]

def core1_worker(key):
    idx = 0
    while True:
        ready_events[idx].wait()
        ready_events[idx].clear()
        asm_xor_crypt(buffers[idx], CHUNK_SIZE, key)
        done_events[idx].set()
        idx = (idx + 1) % 2

# --- 4. CONTROLLER ---
def show_success_splash(file_name):
    """Verfication vibe: Particle flash followed by status."""
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

# --- 4. CONTROLLER ---
def secure_process(filename):
    set_vibe_priority()
    key = get_mixed_entropy()
    _thread.start_new_thread(core1_worker, (key,))
    
    source = f"/sd/{filename}"
    encrypted = f"/sd/{filename}.enc"
    keyfile = f"/sd/{filename}.ky"
    file_size = os.stat(source)[6]
    
    start_time = time.ticks_ms()
    processed = 0
    curr = 0 # Current buffer index for reading
    
    with open(source, 'rb') as fin, open(encrypted, 'wb') as fout:
        # Initial read for the first buffer
        read_prev = fin.readinto(buffers[curr])
        
        while read_prev > 0:
            # 1. Start Core 1 (Worker) on current buffer
            ready_events[curr].set()
            
            # 2. Start Core 0 (Main) reading next chunk while worker is busy XORing
            next_idx = (curr + 1) % 2
            read_next = fin.readinto(buffers[next_idx])
            
            # 3. Wait for Worker to finish XORing the current buffer
            done_events[curr].wait()
            done_events[curr].clear()
            
            # 4. Write processed chunk to disk
            fout.write(buffers[curr][:read_prev])
            processed += read_prev
            
            # Update UI (throttle for performance)
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

# Usage:
# secure_process('topsecret.txt')

# Usage:
# secure_process('secrets.txt')
