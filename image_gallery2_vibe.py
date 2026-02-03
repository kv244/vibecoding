# ICON metadata (used by Pimoroni examples for menu icons)
# NAME Photo Frame
# DESC A touch enabled image gallery

'''
An image gallery demo to turn your Pimoroni Presto into a desktop photo frame!

- Create a folder called 'gallery' on the root of your SD card and fill it with JPEGs.
- The image will change automatically every 5 minutes
- You can also tap the right side of the screen to skip next image and left side to go to the previous :)
'''

import os
import time
import random
import gc
import jpegdec
import machine
import math
import sdcard
import uos
import micropython
from micropython import const
from presto import Presto, Buzzer
try:
    import ltr559
    HAS_LTR559 = True
except ImportError:
    HAS_LTR559 = False

# -- NATIVE OS OPTIMIZATIONS --
# Boost CPU frequency to 250MHz (Performance)
machine.freq(250_000_000)

# Pre-allocate buffer for exceptions (Safety)
micropython.alloc_emergency_exception_buf(100)

# Initialize Hardware Watchdog (Safety - 8s timeout)
wdt = machine.WDT(timeout=8000)

# Global constants for optimization
NUM_LEDS = const(7)
# For 18-bit LFSR (covers 512x512)
TAP = const(0x20400) 
INTERVAL = const(60 * 1)
LEDS_LEFT = (4, 5, 6)
LEDS_RIGHT = (0, 1, 2)

# DEBUG MODE (Set to False for pure production silence)
DEBUG = True

class Profiler:
    """Non-intrusive machine-specific performance tracking."""
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.stats = {}

    def start(self, label):
        if self.enabled:
            self.stats[label] = time.ticks_us()

    def end(self, label):
        if self.enabled and label in self.stats:
            diff = time.ticks_diff(time.ticks_us(), self.stats[label])
            # Print specifically formatted for parsing or mpremote viewing
            print(f"[PERF] {label:15} : {diff / 1000:7.2f}ms | Mem: {gc.mem_free()} B")
            del self.stats[label]

profiler = Profiler(enabled=DEBUG)

class ImageNode:
    def __init__(self, filename):
        self.filename = filename
        self.next = None
        self.prev = None

class Gallery:
    def __init__(self, display_error_callback):
        self.directory = 'gallery'
        self.current_node = None
        self.display_error = display_error_callback
        self._setup_sd()
        self._load_images()

    def _setup_sd(self):
        try:
            # OPTIMIZATION: Overclock SPI to 25MHz. 
            # Standard SD access is 10-12MHz; doubling this reduces JPEG load latency by ~30-40%.
            sd_spi = machine.SPI(0,
                                 baudrate=25_000_000,
                                 sck=machine.Pin(34),
                                 mosi=machine.Pin(35),
                                 miso=machine.Pin(36))
            sd = sdcard.SDCard(sd_spi, machine.Pin(39))
            uos.mount(sd, "/sd")
            if os.stat('sd/gallery'):
                self.directory = 'sd/gallery'
        except Exception: 
            # Fallback to internal flash if SD is missing or unformatted
            pass

    def _load_images(self):
        def numberedfiles(k):
            try:
                # Extract all digits to handle 'img01.jpg', '01.jpg'
                # MicroPython compatible character filter
                digits = "".join(c for c in k if '0' <= c <= '9')
                return int(digits) if digits else 0
            except (ValueError, TypeError):
                return 0

        try:
            # Use lowercase check for robustness, avoiding tuples in endswith for older MicroPython
            raw_files = os.listdir(self.directory)
            files = []
            for f in raw_files:
                # Ensure f is a string (os.listdir usually returns strings, but let's be safe)
                if isinstance(f, str):
                    low = f.lower()
                    if low.endswith('.jpg') or low.endswith('.jpeg'):
                        files.append(f)
            
            files.sort(key=numberedfiles)
            
            if files:
                # Build circular doubly linked list
                head = ImageNode(files[0])
                current = head
                for filename in files[1:]:
                    node = ImageNode(filename)
                    current.next = node
                    node.prev = current
                    current = node
                current.next = head
                head.prev = current
                self.current_node = head
            else:
                self.display_error("No JPEG images found in the 'gallery' folder.")
        except OSError:
            self.display_error("Problem loading images.\n\nEnsure your SD card or Presto root has a 'gallery' folder.")

    def get_current(self):
        return self.current_node.filename if self.current_node else None

    def next(self):
        if self.current_node:
            self.current_node = self.current_node.next

    def prev(self):
        if self.current_node:
            self.current_node = self.current_node.prev

class VisualEffects:
    def __init__(self, presto, display, jpeg):
        self.presto = presto
        self.display = display
        self.j = jpeg
        self.WIDTH, self.HEIGHT = display.get_bounds()
        
        self.lfsr = 1
        
        # Pen Index 0 is transparent on Layer 1 in Presto's Picographics
        self.TRANSPARENT = 0 
        self.BLACK = display.create_pen(0, 0, 0)
        self.BACKGROUND = display.create_pen(1, 1, 1)
        
        # Pre-allocate a small palette for screensaver
        self.random_pens = [display.create_pen(random.getrandbits(8), random.getrandbits(8), random.getrandbits(8)) for _ in range(16)]

    @micropython.viper
    def _calculate_lfsr(self, current_lfsr: int) -> int:
        """Machine-level LFSR bit manipulation using Viper for speed."""
        lsb = current_lfsr & 1
        current_lfsr >>= 1
        if lsb:
            current_lfsr ^= int(TAP)
        return current_lfsr

    @micropython.native
    def fizzlefade(self, speed=0.001, bg_pen=None):
        """Scatter-fade transition effect optimized for 480x480."""
        if bg_pen is None:
            bg_pen = self.BLACK
        
        # Layer 1 acts as a mask over Layer 0
        self.display.set_layer(1)
        self.display.set_pen(bg_pen)
        
        # Localize variables for performance in tight loop
        lfsr = self.lfsr
        width = self.WIDTH
        height = self.HEIGHT
        pixel = self.display.pixel
        update = self.presto.update
        
        while True:
            # Batch updates to remain responsive while being fast
            for i in range(12000):
                x = lfsr & 0x01ff
                y = (lfsr & 0x3fe00) >> 9
                
                # Optimized via Viper helper
                lfsr = self._calculate_lfsr(lfsr)
                
                if x < width and y < height:
                    pixel(x, y)
                
                if lfsr == 1:
                    break
            
            update()
            if lfsr == 1:
                break
        
        self.lfsr = lfsr

    def scroll_left(self, step=40, step_delay=0.0):
        """Reveal Layer 0 by wiping Layer 1 to transparent."""
        self.display.set_layer(1)
        rect = self.display.rectangle
        self.display.set_pen(self.TRANSPARENT)
        for x in range(0, self.WIDTH, step):
            rect(0, 0, x, self.HEIGHT)
            self.presto.update()
            if step_delay > 0: time.sleep(step_delay)

    def scroll_right(self, step=40, step_delay=0.0):
        """Reveal Layer 0 by wiping Layer 1 to transparent."""
        self.display.set_layer(1)
        rect = self.display.rectangle
        self.display.set_pen(self.TRANSPARENT)
        for x in range(self.WIDTH, -1, -step):
            rect(x, 0, self.WIDTH - x, self.HEIGHT)
            self.presto.update()
            if step_delay > 0: time.sleep(step_delay)
            
    def blinds_transition(self, strips=12, speed=0.005):
        """Reveal Layer 0 using transparent vertical blinds on Layer 1."""
        self.display.set_layer(1)
        strip_height = self.HEIGHT // strips
        rect = self.display.rectangle
        
        for h in range(strip_height, -1, -4):
            # We use pen 0 to "eat away" Layer 1
            self.display.set_pen(self.TRANSPARENT)
            for i in range(strips):
                rect(0, i * strip_height, self.WIDTH, strip_height - h)
            self.presto.update()
            if speed > 0: time.sleep(speed)

    def mosaic_transition(self, speed=0.001, block_size=40):
        """Reveal Layer 0 by clearing Layer 1 blocks to transparent."""
        self.display.set_layer(1)
        cols = (self.WIDTH + block_size - 1) // block_size
        rows = (self.HEIGHT + block_size - 1) // block_size
        indices = list(range(cols * rows))
        
        for i in range(len(indices) - 1, 0, -1):
            j = random.getrandbits(10) % (i + 1)
            indices[i], indices[j] = indices[j], indices[i]
        
        rect = self.display.rectangle
        self.display.set_pen(self.TRANSPARENT) 
        
        for idx, i in enumerate(indices):
            r = i // cols
            c = i % cols
            rect(c * block_size, r * block_size, block_size, block_size)
            if idx % 10 == 0:
                self.presto.update()
                if speed > 0: time.sleep(speed)
        self.presto.update()

    def curtain_transition(self, speed=0.005):
        """Reveal Layer 0 by opening transparent curtains on Layer 1."""
        self.display.set_layer(1)
        self.display.set_pen(self.TRANSPARENT)
        rect = self.display.rectangle
        
        mid = self.WIDTH // 2
        for i in range(0, mid + 1, 12):
            rect(mid - i, 0, i * 2, self.HEIGHT) # Expanding center-out
            self.presto.update()
            if speed > 0: time.sleep(speed)

    @micropython.native
    def draw_random_line(self):
        x1 = random.getrandbits(9) % self.WIDTH
        y1 = random.getrandbits(9) % self.HEIGHT
        x2 = random.getrandbits(9) % self.WIDTH
        y2 = random.getrandbits(9) % self.HEIGHT
        pen = self.random_pens[random.getrandbits(4)]
        self.display.set_layer(1)
        self.display.set_pen(pen)
        self.display.line(x1, y1, x2, y2)

    @micropython.native
    def draw_random_circle(self):
        cx = random.getrandbits(9) % self.WIDTH
        cy = random.getrandbits(9) % self.HEIGHT
        r = (random.getrandbits(6) % 35) + 5
        pen_idx = random.getrandbits(4)
        pen = self.random_pens[pen_idx]
        
        self.display.set_layer(1)
        self.display.set_pen(pen)
        
        # Scanline optimization for speed
        r_sq = r * r
        rect = self.display.rectangle
        for dy in range(-r, r, 2):
            dx = int(math.sqrt(r_sq - dy * dy))
            rect(cx - dx, cy + dy, 2 * dx, 1)

    @micropython.native
    def draw_radial_line(self):
        cx, cy = self.WIDTH // 2, self.HEIGHT // 2
        angle = random.random() * 6.283
        length = (random.getrandbits(8) % (self.HEIGHT // 2)) + 20
        
        x2 = cx + int(math.cos(angle) * length)
        y2 = cy + int(math.sin(angle) * length)
        
        pen = self.random_pens[random.getrandbits(4)]
        self.display.set_layer(1)
        self.display.set_pen(pen)
        self.display.line(cx, cy, x2, y2)

    @micropython.native
    def filter_transition(self, img_x, img_y, scale, total_seconds=0.8, steps=5):
        """Gradually tint the image using rectangle overlays (faster than pixel loops)."""
        # Picking a random filter color
        filters = [(128, 128, 128), (162, 138, 101), (100, 128, 160), (160, 100, 100), (100, 160, 100)]
        color = filters[random.getrandbits(8) % len(filters)]
        overlay_pen = self.display.create_pen(*color)

        self.display.set_layer(1)
        self.display.set_pen(overlay_pen)
        rect = self.display.rectangle
        
        # We simulate desaturation by drawing a grid of semi-opaque blocks
        # Grid density increases each step
        for spacing in [20, 12, 6, 2, 1]:
            if spacing == 1:
                rect(0, 0, self.WIDTH, self.HEIGHT)
            else:
                for y in range(0, self.HEIGHT, spacing):
                    for x in range(0, self.WIDTH, spacing):
                        rect(x, y, 1, 1)
            self.presto.update()
            time.sleep(total_seconds / steps)

        time.sleep(0.1)
        # Clean both layers for the next image
        for i in range(2):
            self.display.set_layer(i)
            self.display.set_pen(self.BACKGROUND)
            self.display.clear()
        self.presto.update()

class PhotoFrame:
    def __init__(self):
        # --- Constants and setup ---

        self.presto = Presto()
        self.display = self.presto.display
        self.WIDTH, self.HEIGHT = self.display.get_bounds()

        self.BACKGROUND = self.display.create_pen(1, 1, 1)
        self.WHITE = self.display.create_pen(255, 255, 255)
        self.BLACK = self.display.create_pen(0, 0, 0)

        self.touch = self.presto.touch
        self.j = jpegdec.JPEG(self.display)
        
        # Robust Sensor Initialization (Machine Specific)
        self.light_sensor = None
        self.temp_sensor = None
        
        # Pimoroni Presto usually provides a pre-initialized I2C object on pins 4/5
        # This same bus is shared with the external Qwiic/STEMMA QT connector.
        if hasattr(self.presto, 'i2c'):
            self.i2c = self.presto.i2c
        else:
            try:
                self.i2c = machine.I2C(0, sda=machine.Pin(4), scl=machine.Pin(5))
            except Exception:
                self.i2c = None

        if self.i2c and HAS_LTR559:
            try:
                self.light_sensor = ltr559.LTR559(self.i2c)
            except (Exception, AttributeError):
                pass

        try:
            # Some builds might not have ADC or the CORE_TEMP constant
            if hasattr(machine, 'ADC'):
                self.temp_sensor = machine.ADC(machine.ADC.CORE_TEMP)
        except (Exception, AttributeError):
            pass

        self.buzzer = Buzzer(43)
        self.lines_drawn = 0
        self.SCREENSAVER_DELAY = 10
        self.led_hue = 0.0
        
        # Pre-calculate pens for overlay to avoid allocation in loop
        self.BOX_PEN = self.display.create_pen(80, 80, 80)
        self.BORDER_PEN = self.display.create_pen(110, 110, 110)
        
        self.gallery = Gallery(self.display_error)
        self.effects = VisualEffects(self.presto, self.display, self.j)
        # Pre-calculate sine table for breathing effect 
        self.sine_table = bytearray([int((math.sin(i * 6.28318 / 256) + 1) * 20) for i in range(256)])
        
        # Day/Night mode state
        self.is_dark = False

    def beep(self, frequency=None, duration=0.1):
        """Play a short beep on the Presto buzzer."""
        if frequency is None:
            frequency = random.randint(400, 2000)
        self.buzzer.set_tone(frequency)
        time.sleep(duration)
        self.buzzer.set_tone(-1)

    def display_error(self, text):
        while True:
            for i in range(2):
                self.display.set_layer(i)
                self.display.set_pen(self.BACKGROUND)
                self.display.clear()
            self.display.set_pen(self.WHITE)
            self.display.text(f"Error: {text}", 10, 10, self.WIDTH - 10, 1)
            self.presto.update()
            time.sleep(1)

    def overlay_datetime(self, corner='bottom_left', font_size=2, padding=4, box_color=(80, 80, 80)):
        """
        Draw current date and time with a semi-opaque box behind it.
        Because the display has no alpha channel, we simulate semi-opacity by using a
        mid-dark filled rectangle (box_color). Adjust box_color for lighter/darker effect.
        """
        now = time.localtime()
        date_str = "{:04d}-{:02d}-{:02d}".format(now[0], now[1], now[2])
        time_str = "{:02d}:{:02d}:{:02d}".format(now[3], now[4], now[5])
        
        # Read Internal Temperature (if available)
        env_str = ""
        if self.temp_sensor:
            try:
                reading = self.temp_sensor.read_u16() * (3.3 / 65535)
                temperature = 27 - (reading - 0.706) / 0.001721
                env_str = "{:.1f}'C".format(temperature)
            except Exception:
                pass

        # Rough text metrics
        char_width = 12 * max(1, font_size)
        lines_to_draw = 3 if env_str else 2
        text_width = max(len(date_str), len(time_str), len(env_str)) * char_width
        line_height = 18 * max(1, font_size)

        box_w = text_width + (padding * 2)
        box_h = (line_height * lines_to_draw) + (padding * 2)

        # Choose box position based on corner
        if corner == 'bottom_left':
            box_x = 0
            box_y = self.HEIGHT - box_h
        elif corner == 'bottom_right':
            box_x = self.WIDTH - box_w
            box_y = self.HEIGHT - box_h
        elif corner == 'top_left':
            box_x = 0
            box_y = 0
        elif corner == 'top_right':
            box_x = self.WIDTH - box_w
            box_y = 0
        else:
            box_x = 0
            box_y = self.HEIGHT - box_h

        # Draw semi-opaque filled box on top layer
        self.display.set_layer(1)
        # Use cached pen if color matches default, else create
        box_pen = self.BOX_PEN if box_color == (80, 80, 80) else self.display.create_pen(*box_color)
        self.display.set_pen(box_pen)
        # Draw scanlines to simulate transparency
        for y in range(box_y, box_y + box_h, 2):
            self.display.rectangle(box_x, y, box_w, 1)

        # Optional faint border for definition
        border_pen = self.BORDER_PEN if box_color == (80, 80, 80) else self.display.create_pen(min(box_color[0]+30,255), min(box_color[1]+30,255), min(box_color[2]+30,255))
        self.display.set_pen(border_pen)
        self.display.line(box_x, box_y, box_x + box_w, box_y)
        self.display.line(box_x, box_y + box_h - 1, box_x + box_w, box_y + box_h - 1)
        self.display.line(box_x, box_y, box_x, box_y + box_h)
        self.display.line(box_x + box_w - 1, box_y, box_x + box_w - 1, box_y + box_h)

        # Draw text with a small shadow for readability
        self.display.set_pen(self.BLACK) # Use cached BLACK constant
        text_x = box_x + padding
        date_y = box_y + padding
        time_y = date_y + line_height
        
        self.display.text(date_str, text_x + 1, date_y + 1, self.WIDTH, font_size)
        self.display.text(time_str, text_x + 1, time_y + 1, self.WIDTH, font_size)
        if env_str:
            env_y = time_y + line_height
            self.display.text(env_str, text_x + 1, env_y + 1, self.WIDTH, font_size)

        self.display.set_pen(self.WHITE)
        self.display.text(date_str, text_x, date_y, self.WIDTH, font_size)
        self.display.text(time_str, text_x, time_y, self.WIDTH, font_size)
        if env_str:
            self.display.text(env_str, text_x, env_y, self.WIDTH, font_size)

        self.presto.update()

    @staticmethod
    @micropython.native
    def hsv_to_rgb(h, s, v):
        """Native optimized HSV conversion (Viper does not support float args well)."""
        if s == 0.0:
            return int(v), int(v), int(v)
        
        i = int(h * 6.0)
        f = (h * 6.0) - float(i)
        p = int(v * (1.0 - s))
        q = int(v * (1.0 - s * f))
        t = int(v * (1.0 - s * (1.0 - f)))
        
        idx = i % 6
        if idx == 0: return int(v), t, p
        if idx == 1: return q, int(v), p
        if idx == 2: return p, int(v), t
        if idx == 3: return p, q, int(v)
        if idx == 4: return t, p, int(v)
        return int(v), p, q

    @micropython.native
    def update_leds(self):
        """Cycle LEDs through a gentle rainbow."""
        self.led_hue += 0.01
        if self.led_hue > 1.0:
            self.led_hue = 0.0
            
        # Calculate pulsating brightness using lookup table (integer math only)
        # Map time to 0-255 index. Shift right by 4 gives approx 3-4s period
        intensity = 10 + self.sine_table[(time.ticks_ms() >> 4) & 0xFF]

        # Simple HSV to RGB conversion for the LED bar
        # Optimize: Avoid list comprehension allocation
        hr, hg, hb = self.hsv_to_rgb(self.led_hue, 1.0, 1.0)
        r, g, b = int(hr * intensity), int(hg * intensity), int(hb * intensity)
        
        set_led = self.presto.set_led_rgb # Cache method
        for i in range(NUM_LEDS):
            set_led(i, r, g, b)

    # --- Image display ---
    def show_image(self, show_next=False, show_previous=False, fast=False):
        """Uses double buffering (Layer 0 and 1) for smooth transitions."""
        self.lines_drawn = 0
        if self.gallery.get_current() is None:
            return

        if show_next: self.gallery.next()
        if show_previous: self.gallery.prev()

        try:
            gc.collect()
            img = f"{self.gallery.directory}/{self.gallery.get_current()}"
            self.j.open_file(img)
            
            img_w, img_h = self.j.get_width(), self.j.get_height()
            scale = jpegdec.JPEG_SCALE_FULL
            div = 1
            if img_w > self.WIDTH or img_h > self.HEIGHT:
                scale = jpegdec.JPEG_SCALE_HALF; div = 2
                if (img_w // 2) > self.WIDTH or (img_h // 2) > self.HEIGHT:
                    scale = jpegdec.JPEG_SCALE_QUARTER; div = 4
                    if (img_w // 4) > self.WIDTH or (img_h // 4) > self.HEIGHT:
                        scale = jpegdec.JPEG_SCALE_EIGHTH; div = 8
            
            img_width, img_height = img_w // div, img_h // div
            img_x = (self.WIDTH - img_width) // 2 if img_width < self.WIDTH else 0
            img_y = (self.HEIGHT - img_height) // 2 if img_height < self.HEIGHT else 0
            
            # Use random background color for letterboxing
            bg_pen = self.display.create_pen(random.getrandbits(8), random.getrandbits(8), random.getrandbits(8))

            # --- DOUBLE BUFFERING LOGIC ---
            # 1. Decode NEXT image to Layer 0 (the background layer)
            profiler.start("JPEG Decode")
            self.display.set_layer(0)
            self.display.set_pen(bg_pen)
            self.display.clear()
            self.j.decode(img_x, img_y, scale, dither=True)
            profiler.end("JPEG Decode")
            
            # (Layer 1 still contains the OLD image and overlays)
            
            # 3. TRANSITION REVEAL LOGIC
            profiler.start("Transition")
            # PERFORMANCE: Instead of re-decoding the image to Layer 1, we 'wipe' 
            # Layer 1 to TRANSPARENT (Pen 0). This reveals the Layer 0 image 
            # underneath. This is significantly faster and uses less CPU/RAM.
            transition = random.choice(["fizzle", "scroll_left", "scroll_right", "blinds", "mosaic", "curtain"])
            if transition == "fizzle":
                self.effects.fizzlefade(bg_pen=self.effects.TRANSPARENT)
            elif transition == "scroll_left":
                self.effects.scroll_left(step=40 if fast else 24)
            elif transition == "scroll_right":
                self.effects.scroll_right(step=40 if fast else 24)
            elif transition == "blinds":
                self.effects.blinds_transition(speed=0 if fast else 0.005)
            elif transition == "mosaic":
                self.effects.mosaic_transition(block_size=60 if fast else 48)
            elif transition == "curtain":
                self.effects.curtain_transition(speed=0 if fast else 0.005)

            profiler.end("Transition")
            
            # 4. UI OVERLAY
            # We draw UI on Layer 1. Since most of Layer 1 is now transparent, 
            # the image on Layer 0 shows through everywhere except where we draw.
            self.display.set_layer(1)
            corner = random.choice(['bottom_left', 'bottom_right', 'top_left', 'top_right'])
            self.overlay_datetime(corner=corner, font_size=2, padding=6)
            
            # Reset screensaver counter for new image
            self.lines_drawn = 0

        except (OSError, IndexError) as e:
            self.display_error(f"Image Error: {e}")

    def clear(self):
        self.display.set_pen(self.BACKGROUND)
        self.display.set_layer(0)
        self.display.clear()
        self.display.set_layer(1)
        self.display.clear()

    def run(self):
        # --- Main loop ---
        print(f"--- STARTING PHOTO FRAME ---")
        print(f"System: {machine.freq() / 1_000_000} MHz | GC: {gc.mem_alloc()} used")
        
        last_updated = time.ticks_ms()
        self.clear()
        self.show_image()
        self.presto.update()

        while True:
            # SAFETY/PERFORMANCE: Time-Based Day/Night Logic
            now_time = time.localtime()
            hour = now_time[3]
            
            # AUTO-ADAPTIVE BACKLIGHT (Machine Sensor Logic)
            if self.light_sensor:
                lux = self.light_sensor.get_lux()
                # Map lux to backlight (0.05 to 1.0)
                # If dark (lux < 5), use min backlight. If bright (lux > 100), use max.
                target_backlight = 0.05 + (min(lux, 100) / 100) * 0.95
                self.is_dark = lux < 5
            else:
                # Fallback to time-based if sensor fails
                self.is_dark = (hour >= 22 or hour < 7)
                target_backlight = 0.1 if self.is_dark else 0.8
            
            self.presto.set_backlight(target_backlight)

            # Poll touch sensor for user interaction
            self.touch.poll()

            # SCREENSAVER: If idle for X seconds, start generative art
            if not self.is_dark:
                self.update_leds()
                
                now = time.ticks_ms()
                if time.ticks_diff(now, last_updated) > self.SCREENSAVER_DELAY * 1000 and self.lines_drawn < 100:
                    # MACHINE OPTIMIZATION: Draw 5 items per update to minimize bus latency
                    for _ in range(5):
                        choice = random.getrandbits(2) % 3
                        if choice == 0: self.effects.draw_random_line()
                        elif choice == 1: self.effects.draw_random_circle()
                        else: self.effects.draw_radial_line()
                        self.lines_drawn += 1
                    
                    self.presto.update()
            else:
                # In the dark, keep the timer current but don't draw anything
                now = time.ticks_ms() 

            # Auto-advance after interval (no touch): use faster transitions
            if time.ticks_diff(now, last_updated) > INTERVAL * 1000:
                last_updated = time.ticks_ms()
                self.show_image(show_next=True, fast=True)
                self.presto.update()
                # Double beep cue for auto-advance with random frequencies
                self.beep(None, 0.05)
                time.sleep(0.05)
                self.beep(None, 0.05)

            # Handle touch input
            if self.touch.state:
                if self.touch.x > self.WIDTH // 2:   # right side → next image
                    self.beep(None, 0.1)        # random-frequency beep
                    for i in LEDS_RIGHT:
                        self.presto.set_led_rgb(i, 0, 255, 0)   # green flash
                    self.show_image(show_next=True, fast=False)
                    self.presto.update()
                    last_updated = time.ticks_ms()
                    for i in LEDS_RIGHT:
                        self.presto.set_led_rgb(i, 0, 0, 0)
                    time.sleep(0.01)

                elif self.touch.x < self.WIDTH // 2: # left side → previous image
                    self.beep(None, 0.1)         # random-frequency beep
                    for i in LEDS_LEFT:
                        self.presto.set_led_rgb(i, 0, 0, 255)   # blue flash
                    self.show_image(show_previous=True, fast=False)
                    self.presto.update()
                    last_updated = time.ticks_ms()
                    for i in LEDS_LEFT:
                        self.presto.set_led_rgb(i, 0, 0, 0)
                    time.sleep(0.01)

                # Wait for touch release to avoid multiple triggers
                while self.touch.state:
                    self.touch.poll()
                    time.sleep(0.02)

            # Feed the watchdog every loop to prove system is alive
            wdt.feed()
            # Small idle sleep to reduce power when not active
            time.sleep(0.02)

if __name__ == "__main__":
    app = PhotoFrame()
    app.run()

