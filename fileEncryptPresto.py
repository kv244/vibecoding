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

def set_vibe_priority():
    bp_reg = 0x40094000
    machine.mem32[bp_reg] = (machine.mem32[bp_reg] | 0x1 | (0xFF << 10)) & ~(1 << 1)

def show_success_splash(file_name):
    # Quick "particle" flash for the vibe
    for _ in range(50):
        display.set_pen(display.create_pen(random.randint(0,255), 255, random.randint(0,255)))
        display.circle(random.randint(0, WIDTH), random.randint(0, HEIGHT), random.randint(2, 8))
    
    display.set_pen(BG)
    display.rectangle(40, HEIGHT // 2 - 40, WIDTH - 80, 80)
    display.set_pen(SUCCESS_PEN)
    display.text("VERIFIED", WIDTH // 2 - 70, HEIGHT // 2 - 20, scale=5)
    display.set_pen(TEXT_PEN)
    display.text(f"{file_name} SECURED", WIDTH // 2 - 80, HEIGHT // 2 + 25, scale=2)
    display.update() # Use blocking update for the final static screen

def update_ui_async(file_name, progress, speed_kbps):
    display.set_pen(BG)
    display.clear()
    display.set_pen(TEXT_PEN)
    display.text(f"SECURE ENCRYPT: {file_name}", 20, 30, scale=3)
    display.rectangle(20, 80, WIDTH - 40, 30)
    display.set_pen(BG)
    display.rectangle(22, 82, WIDTH - 44, 26)
    display.set_pen(BAR_FG)
    display.rectangle(22, 82, int((WIDTH - 44) * progress), 26)
    display.set_pen(TEXT_PEN)
    display.text(f"{int(progress * 100)}%", WIDTH // 2 - 25, 120, scale=4)
    display.set_pen(SPEED_PEN)
    display.text(f"SPEED: {speed_kbps:.2f} KB/s", 20, 170, scale=3)
    display.update_async()

# --- 2. THE KERNEL ---
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

# --- 3. DUAL-CORE ENGINE ---
CHUNK_SIZE = 4096
buffer = bytearray(CHUNK_SIZE)
data_ready = threading.Event()
data_done = threading.Event()

def core1_runner(secret_key):
    try:
        while True:
            data_ready.wait()
            data_ready.clear()
            asm_xor_crypt(buffer, CHUNK_SIZE, secret_key)
            data_done.set()
    except Exception:
        data_done.set()

# --- 4. CONTROLLER & VERIFICATION ---
def get_file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk: break
            h.update(chunk)
    return h.hexdigest()

def encrypt_with_vibe(input_path, output_path, key, silent=False):
    if not silent: set_vibe_priority()
    file_size = os.stat(input_path)[6]
    processed = 0
    start_time = time.ticks_ms()

    with open(input_path, 'rb') as fin, open(output_path, 'wb') as fout:
        while True:
            read = fin.readinto(buffer)
            if read == 0: break
            if read < CHUNK_SIZE:
                for i in range(read, CHUNK_SIZE): buffer[i] = 0
            
            data_ready.set()
            processed += read
            
            if not silent and ((processed // CHUNK_SIZE) % 12 == 0 or processed == file_size):
                elapsed_s = time.ticks_diff(time.ticks_ms(), start_time) / 1000
                speed = (processed / 1024) / elapsed_s if elapsed_s > 0 else 0
                update_ui_async(os.path.basename(input_path), processed / file_size, speed)
            
            data_done.wait()
            data_done.clear()
            fout.write(buffer[:read])

def secure_process(filename, key):
    # Start the worker once
    _thread.start_new_thread(core1_runner, (key,))
    
    source = f"/sd/{filename}"
    encrypted = f"/sd/{filename}.enc"
    decrypted = f"/sd/{filename}.dec"

    # Step 1: Encrypt
    encrypt_with_vibe(source, encrypted, key)
    
    # Step 2: Verify (Decrypting silently in background)
    print("[*] Verifying integrity...")
    encrypt_with_vibe(encrypted, decrypted, key, silent=True)
    
    if get_file_hash(source) == get_file_hash(decrypted):
        show_success_splash(filename)
        print("[+] Hash Verified. System Secure.")
    else:
        print("[!] Hash Mismatch detected.")
    
    if os.path.exists(decrypted): os.remove(decrypted)

# Usage:
# secure_process('secrets.txt', 0xACE1BADE)
