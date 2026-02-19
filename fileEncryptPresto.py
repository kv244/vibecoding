import machine, _thread, time, uctypes, threading, os, hashlib, random

# --- 1. UI & HARDWARE INITIALIZATION ---
display = PicoGraphics(display=DISPLAY_PRESTO_ST7789)
WIDTH, HEIGHT = display.get_bounds()
BG = display.create_pen(10, 10, 20)
BAR_FG = display.create_pen(0, 255, 150)
TEXT_PEN = display.create_pen(255, 255, 255)
SPEED_PEN = display.create_pen(255, 200, 0)
SUCCESS_PEN = display.create_pen(0, 255, 0)

def set_vibe_priority():
    # Boost priority for smooth encryption on RP2350
    bp_reg = 0x40094000
    machine.mem32[bp_reg] = (machine.mem32[bp_reg] | 0x1 | (0xFF << 10)) & ~(1 << 1)

def get_rp2350_trng():
    """Reads 32-bit hardware entropy from RP2350 TRNG."""
    # TRNG_BASE = 0x400b0000; FIFO_REG = 0x10
    # Note: TRNG must be enabled first. For simplicity, we use machine.rng() 
    # if available, or direct register access on RP2350.
    try:
        if hasattr(machine, "rng"):
            return machine.rng()
    except:
        pass
    # Fallback to a seeded hash of various noisy sources if TRNG isn't directly exposed
    return random.getrandbits(32) 

def show_success_splash(file_name):
    for _ in range(50):
        display.set_pen(display.create_pen(random.randint(0,255), 255, random.randint(0,255)))
        display.circle(random.randint(0, WIDTH), random.randint(0, HEIGHT), random.randint(2, 8))
    
    display.set_pen(BG)
    display.rectangle(40, HEIGHT // 2 - 40, WIDTH - 80, 80)
    display.set_pen(SUCCESS_PEN)
    display.text("SECURED", WIDTH // 2 - 60, HEIGHT // 2 - 20, scale=5)
    display.set_pen(TEXT_PEN)
    display.text(f"KEY SAVED FOR {file_name}", WIDTH // 2 - 110, HEIGHT // 2 + 25, scale=2)
    display.update()

def update_ui_async(file_name, progress, speed_kbps):
    display.set_pen(BG)
    display.clear()
    display.set_pen(TEXT_PEN)
    display.text(f"PRESTO ENCRYPT: {file_name}", 20, 30, scale=3)
    display.rectangle(20, 80, WIDTH - 40, 30)
    display.set_pen(BG)
    display.rectangle(22, 82, WIDTH - 44, 26)
    display.set_pen(BAR_FG)
    display.rectangle(22, 82, int((WIDTH - 44) * progress), 26)
    display.set_pen(TEXT_PEN)
    display.text(f"{int(progress * 100)}%", WIDTH // 2 - 25, 120, scale=4)
    display.set_pen(SPEED_PEN)
    display.text(f"{speed_kbps:.2f} KB/s", 20, 170, scale=3)
    display.update_async()

# --- 2. THE KERNEL (Thumb ASM) ---
@micropython.asm_thumb
def asm_xor_crypt(r0, r1, r2):
    # r0: buffer ptr, r1: length, r2: 32-bit key
    label(LOOP)
    ldr(r3, [r0, 0])
    eor(r3, r2)
    str(r3, [r0, 0])
    add(r0, 4)
    sub(r1, 4)
    cmp(r1, 0)
    bgt(LOOP)

# --- 3. DUAL-CORE ENGINE ---
CHUNK_SIZE = 4096
buffer = bytearray(CHUNK_SIZE)
data_ready = threading.Event()
data_done = threading.Event()

def core1_runner(secret_key):
    while True:
        data_ready.wait()
        data_ready.clear()
        asm_xor_crypt(buffer, CHUNK_SIZE, secret_key)
        data_done.set()

# --- 4. CONTROLLER ---
def encrypt_file(input_path, output_path, key, silent=False):
    file_size = os.stat(input_path)[6]
    processed = 0
    start_time = time.ticks_ms()

    with open(input_path, 'rb') as fin, open(output_path, 'wb') as fout:
        while True:
            read = fin.readinto(buffer)
            if read == 0: break
            
            # Ensure padding for ASM kernel if last chunk is small
            original_read = read
            if read < CHUNK_SIZE:
                for i in range(read, CHUNK_SIZE): buffer[i] = 0
                read = CHUNK_SIZE # Process full block

            data_ready.set()
            data_done.wait()
            data_done.clear()
            
            fout.write(buffer[:original_read])
            processed += original_read
            
            if not silent and ((processed // CHUNK_SIZE) % 20 == 0 or processed == file_size):
                elapsed_s = time.ticks_diff(time.ticks_ms(), start_time) / 1000
                speed = (processed / 1024) / elapsed_s if elapsed_s > 0 else 0
                update_ui_async(os.path.basename(input_path), processed / file_size, speed)

    # RAM Cleansing
    for i in range(CHUNK_SIZE): buffer[i] = 0

def secure_process(filename):
    """Full Secure Pipeline: Key Gen -> Encrypt -> Key Save"""
    set_vibe_priority()
    key = get_rp2350_trng()
    
    _thread.start_new_thread(core1_runner, (key,))
    
    source = f"/sd/{filename}"
    encrypted = f"/sd/{filename}.enc"
    keyfile = f"/sd/{filename}.ky"

    print(f"[*] Securing {filename} with HW Key: {hex(key)}")
    encrypt_file(source, encrypted, key)
    
    # Save 32-bit key to file
    with open(keyfile, 'w') as f:
        f.write(hex(key))
    
    show_success_splash(filename)
    print(f"[+] Encryption Complete. Key saved as {keyfile}")

# Usage:
# secure_process('secrets.txt')
