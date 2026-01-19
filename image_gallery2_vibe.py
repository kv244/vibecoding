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
from micropython import const
import micropython

import jpegdec
import machine
import math
import sdcard
import uos
from presto import Presto, Buzzer

# Global constants for optimization
NUM_LEDS = const(7)
TAP = const(0xdc29)
INTERVAL = const(60 * 1)
LEDS_LEFT = (4, 5, 6)
LEDS_RIGHT = (0, 1, 2)

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
            sd_spi = machine.SPI(0,
                                 sck=machine.Pin(34, machine.Pin.OUT),
                                 mosi=machine.Pin(35, machine.Pin.OUT),
                                 miso=machine.Pin(36, machine.Pin.OUT))
            sd = sdcard.SDCard(sd_spi, machine.Pin(39))
            uos.mount(sd, "/sd")
            if os.stat('sd/gallery'):
                self.directory = 'sd/gallery'
        except OSError:
            pass

    def _load_images(self):
        def numberedfiles(k):
            try:
                return int(k[:-4])
            except ValueError:
                return 0

        try:
            files = list(file for file in sorted(os.listdir(self.directory), key=numberedfiles)
                         if file.endswith('.jpg'))
            
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
        except OSError:
            self.display_error("Problem loading images.\n\nEnsure that your Presto or SD card contains a 'gallery' folder in the root")

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
        
        self.BLACK = display.create_pen(0, 0, 0)
        self.BACKGROUND = display.create_pen(1, 1, 1)
        
        # Pre-allocate a small palette for screensaver to avoid memory fragmentation/leak
        self.random_pens = [display.create_pen(random.getrandbits(8), random.getrandbits(8), random.getrandbits(8)) for _ in range(16)]

    @micropython.native
    def fizzlefade(self, speed=0.01, bg_pen=None):
        """Scatter-fade transition effect. 'speed' controls the pause after each update."""
        if bg_pen is None:
            bg_pen = self.BLACK
        self.display.set_pen(bg_pen)
        self.display.set_layer(1)
        
        # Localize variables for performance in tight loop
        lfsr = self.lfsr
        width = self.WIDTH
        height = self.HEIGHT
        pixel = self.display.pixel  # Cache method lookup
        
        while True:
            for i in range(5000):
                x = lfsr & 0x00ff
                y = (lfsr & 0xff00) >> 8
                lsb = lfsr & 1
                lfsr >>= 1
                if lsb:
                    lfsr ^= TAP
                
                # Adjust x by -1 as per original logic (likely 1-based LFSR mapping)
                px = x - 1
                if 0 <= px < width and y < height:
                    pixel(px, y)
                
                if lfsr == 1:
                    break
            self.presto.update()
            time.sleep(speed)
            if lfsr == 1:
                break
        
        self.lfsr = lfsr

    def scroll_left(self, img_x, img_y, scale, step=20, step_delay=0.01, bg_pen=None):
        """Scroll new image in from right to left. step_delay controls speed (smaller = faster)."""
        if bg_pen is None:
            bg_pen = self.BACKGROUND
        self.display.set_layer(1)
        for offset in range(self.WIDTH, -1, -step):
            self.display.set_pen(bg_pen)
            self.display.clear()
            self.j.decode(img_x - offset, img_y, scale, dither=True)
            self.presto.update()
            time.sleep(step_delay)

    def scroll_right(self, img_x, img_y, scale, step=20, step_delay=0.01, bg_pen=None):
        """Scroll new image in from left to right. step_delay controls speed (smaller = faster)."""
        if bg_pen is None:
            bg_pen = self.BACKGROUND
        self.display.set_layer(1)
        for offset in range(-self.WIDTH, 1, step):
            self.display.set_pen(bg_pen)
            self.display.clear()
            self.j.decode(img_x - offset, img_y, scale, dither=True)
            self.presto.update()
            time.sleep(step_delay)
            
    def blinds_transition(self, img_x, img_y, scale, strips=10, speed=0.02, bg_pen=None):
        """Reveal the image using horizontal blinds effect."""
        if bg_pen is None:
            bg_pen = self.BACKGROUND
        self.display.set_layer(1)
        strip_height = self.HEIGHT // strips
        rect = self.display.rectangle # Cache method
        width = self.WIDTH
        
        # Animate the blinds opening (bar_height goes from strip_height down to 0)
        for bar_height in range(strip_height, -1, -2):
            self.display.set_pen(bg_pen)
            self.display.clear()
            # 1. Draw the full image
            self.j.decode(img_x, img_y, scale, dither=True)
            
            # 2. Draw black bars over it to simulate the blinds
            self.display.set_pen(self.BLACK)
            for i in range(strips):
                y_pos = (i * strip_height) + (strip_height - bar_height)
                rect(0, y_pos, width, bar_height)
            
            self.presto.update()
            time.sleep(speed)

    def mosaic_transition(self, speed=0.001, block_size=30, bg_pen=None):
        """Cover the screen with random blocks."""
        if bg_pen is None: bg_pen = self.BLACK
        self.display.set_layer(1)
        self.display.set_pen(bg_pen)
        
        cols = (self.WIDTH + block_size - 1) // block_size
        rows = (self.HEIGHT + block_size - 1) // block_size
        indices = list(range(cols * rows))
        
        # Shuffle indices
        for i in range(len(indices) - 1, 0, -1):
            j = random.getrandbits(10) % (i + 1)
            indices[i], indices[j] = indices[j], indices[i]
        
        rect = self.display.rectangle
        update = self.presto.update
        
        for idx, i in enumerate(indices):
            r = i // cols
            c = i % cols
            rect(c * block_size, r * block_size, block_size, block_size)
            if idx % 5 == 0:
                update()
                time.sleep(speed)
        update()

    def curtain_transition(self, speed=0.01, bg_pen=None):
        """Close curtains from the sides."""
        if bg_pen is None: bg_pen = self.BLACK
        self.display.set_layer(1)
        self.display.set_pen(bg_pen)
        rect = self.display.rectangle
        update = self.presto.update
        
        mid = self.WIDTH // 2
        for i in range(0, mid + 1, 8):
            rect(0, 0, i, self.HEIGHT) # Left curtain
            rect(self.WIDTH - i, 0, i, self.HEIGHT) # Right curtain
            update()
            time.sleep(speed)

    def draw_random_line(self):
        x1 = random.randint(0, self.WIDTH)
        y1 = random.randint(0, self.HEIGHT)
        x2 = random.randint(0, self.WIDTH)
        y2 = random.randint(0, self.HEIGHT)
        # Use pre-allocated pens randomly
        pen = random.choice(self.random_pens)
        self.display.set_layer(1)
        self.display.set_pen(pen)
        self.display.line(x1, y1, x2, y2)
        self.presto.update()

    def draw_random_circle(self):
        cx = random.randint(0, self.WIDTH)
        cy = random.randint(0, self.HEIGHT)
        r = random.randint(5, 40)
        pen = random.choice(self.random_pens)
        self.display.set_layer(1)
        self.display.set_pen(pen)
        
        # Draw scanlines to simulate transparency
        r_sq = r * r
        for dy in range(-r, r, 2):
            dx = int(math.sqrt(r_sq - dy * dy))
            self.display.rectangle(cx - dx, cy + dy, 2 * dx, 1)
            
        self.presto.update()

    def draw_radial_line(self):
        cx, cy = self.WIDTH // 2, self.HEIGHT // 2
        angle = random.random() * 6.283
        length = random.randint(20, min(self.WIDTH, self.HEIGHT) // 2)
        
        x2 = cx + int(math.cos(angle) * length)
        y2 = cy + int(math.sin(angle) * length)
        
        pen = random.choice(self.random_pens)
        self.display.set_layer(1)
        self.display.set_pen(pen)
        self.display.line(cx, cy, x2, y2)
        self.presto.update()

    @micropython.native
    def filter_transition(self, img_x, img_y, scale, total_seconds=3.0, steps=8, bg_pen=None):
        """
        Gradually shift the currently displayed image to a tinted look over total_seconds.
        """
        if bg_pen is None:
            bg_pen = self.BACKGROUND
        # Draw the image on the top layer so overlays sit above it
        self.display.set_layer(1)
        self.j.decode(img_x, img_y, scale, dither=True)
        self.presto.update()

        # Pick a random filter color
        filters = [
            (128, 128, 128), # Grayscale
            (162, 138, 101), # Sepia
            (100, 128, 160), # Cool Blue
            (160, 100, 100), # Warm Red
            (100, 160, 100), # Matrix Green
        ]
        color = random.choice(filters)
        filter_pen = self.display.create_pen(*color)

        # Parameters for the simulated desaturation
        step_delay = total_seconds / max(1, steps)
        
        # Use the chosen filter pen
        overlay_pen = filter_pen

        # Start with a coarse spacing and increase density each step
        start_spacing = max(10, steps + 1)
        # compute decrement so we get roughly 'steps' iterations
        decrement = max(1, start_spacing // steps)
        spacing = start_spacing
        
        pixel = self.display.pixel # Cache method
        width = self.WIDTH
        height = self.HEIGHT
        
        while spacing > 0:
            # Optimization: If spacing is 1, it's a solid fill. Use rectangle instead of pixel loop.
            if spacing == 1:
                self.display.set_pen(overlay_pen)
                self.display.rectangle(0, 0, self.WIDTH, self.HEIGHT)
                self.presto.update()
                break

            self.display.set_pen(overlay_pen)
            self.display.set_layer(1)
            for y in range(0, height, spacing):
                for x in range(0, width, spacing):
                    pixel(x, y)
            self.presto.update()
            time.sleep(step_delay)
            spacing -= decrement

        # Final full overlay
        self.display.set_pen(overlay_pen)
        self.display.rectangle(0, 0, self.WIDTH, self.HEIGHT)
        self.presto.update()

        # Hold the grayscale image for a short moment so the shift is visible
        time.sleep(0.2)

        # Clear both layers to prepare for next image
        self.display.set_pen(self.BACKGROUND)
        self.display.set_layer(0)
        self.display.clear()
        self.display.set_layer(1)
        self.display.set_pen(bg_pen)
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

        self.buzzer = Buzzer(43)
        self.lines_drawn = 0
        self.SCREENSAVER_DELAY = 10
        self.led_hue = 0.0
        
        # Pre-calculate pens for overlay to avoid allocation in loop
        self.BOX_PEN = self.display.create_pen(80, 80, 80)
        self.BORDER_PEN = self.display.create_pen(110, 110, 110)
        
        self.gallery = Gallery(self.display_error)
        self.effects = VisualEffects(self.presto, self.display, self.j)
        # Pre-calculate sine table for breathing effect (0-40 range) to avoid math.sin in loop
        self.sine_table = bytearray([int((math.sin(i * 6.28318 / 256) + 1) * 20) for i in range(256)])

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

        # Rough text metrics (adjust if your font metrics differ)
        char_width = 12 * max(1, font_size)
        text_width = max(len(date_str), len(time_str)) * char_width
        line_height = 18 * max(1, font_size)

        box_w = text_width + (padding * 2)
        box_h = (line_height * 2) + (padding * 2)

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

        self.display.set_pen(self.WHITE)
        self.display.text(date_str, text_x, date_y, self.WIDTH, font_size)
        self.display.text(time_str, text_x, time_y, self.WIDTH, font_size)

        self.presto.update()

    @staticmethod
    @micropython.native
    def hsv_to_rgb(h, s, v):
        if s == 0.0:
            return v, v, v
        i = int(h * 6.0)
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        # Optimization: Use if/return for native code efficiency (avoids list allocation)
        if i == 0: return v, t, p
        if i == 1: return q, v, p
        if i == 2: return p, v, t
        if i == 3: return p, q, v
        if i == 4: return t, p, v
        if i == 5: return v, p, q

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
        """
        show_next / show_previous: navigation flags
        fast: if True, use faster transitions (used when auto-advancing with no touch)
        """
        self.lines_drawn = 0
        if self.gallery.get_current() is None:
            return

        if show_next:
            self.gallery.next()
        if show_previous:
            self.gallery.prev()
        try:
            gc.collect()  # Clean up memory before loading new image
            img = f"{self.gallery.directory}/{self.gallery.get_current()}"
            self.j.open_file(img)
            
            # Calculate scale to fit image to screen
            img_w, img_h = self.j.get_width(), self.j.get_height()
            scale = jpegdec.JPEG_SCALE_FULL
            div = 1
            if img_w > self.WIDTH or img_h > self.HEIGHT:
                scale = jpegdec.JPEG_SCALE_HALF
                div = 2
                if (img_w // 2) > self.WIDTH or (img_h // 2) > self.HEIGHT:
                    scale = jpegdec.JPEG_SCALE_QUARTER
                    div = 4
                    if (img_w // 4) > self.WIDTH or (img_h // 4) > self.HEIGHT:
                        scale = jpegdec.JPEG_SCALE_EIGHTH
                        div = 8
            
            img_width = img_w // div
            img_height = img_h // div
            
            img_x = (self.WIDTH // 2) - (img_width // 2) if img_width < self.WIDTH else 0
            img_y = (self.HEIGHT // 2) - (img_height // 2) if img_height < self.HEIGHT else 0
            
            # Determine background color
            bg_pen = self.BACKGROUND
            if img_width < self.WIDTH or img_height < self.HEIGHT:
                bg_pen = self.display.create_pen(random.getrandbits(8), random.getrandbits(8), random.getrandbits(8))

            # Clear background layer and draw the image on layer 0 first
            self.display.set_layer(0)
            self.display.set_pen(bg_pen)
            self.display.clear()
            self.j.decode(img_x, img_y, scale, dither=True)
            self.presto.update()

            # --- Random transition choice ---
            transition = random.choice(["fizzle", "scroll_left", "scroll_right", "blinds", "mosaic", "curtain"])
            if transition == "fizzle":
                # faster speed when auto-advancing
                self.effects.fizzlefade(speed=0.001 if fast else 0.005, bg_pen=bg_pen)
            elif transition == "scroll_left":
                self.effects.scroll_left(img_x, img_y, scale, step=40 if fast else 30, step_delay=0.001 if fast else 0.005, bg_pen=bg_pen)
            elif transition == "scroll_right":
                self.effects.scroll_right(img_x, img_y, scale, step=40 if fast else 30, step_delay=0.001 if fast else 0.005, bg_pen=bg_pen)
            elif transition == "blinds":
                self.effects.blinds_transition(img_x, img_y, scale, strips=12, speed=0.001 if fast else 0.01, bg_pen=bg_pen)
            elif transition == "mosaic":
                self.effects.mosaic_transition(speed=0.001 if fast else 0.002, block_size=30, bg_pen=bg_pen)
            elif transition == "curtain":
                self.effects.curtain_transition(speed=0.001 if fast else 0.01, bg_pen=bg_pen)

            # Always run filter transition (gradual tint) before clearing
            # Use shorter total_seconds when fast=True
            self.effects.filter_transition(img_x, img_y, scale, total_seconds=(0.5 if fast else 1.5), steps=(4 if fast else 8), bg_pen=bg_pen)

            # After grayscale_transition clears the screen, draw the next final image on top layer
            self.display.set_layer(1)
            self.j.decode(img_x, img_y, scale, dither=True)

            # Overlay date and time with a semi-opaque box
            corner = random.choice(['bottom_left', 'bottom_right', 'top_left', 'top_right'])
            self.overlay_datetime(corner=corner, font_size=2, padding=6, box_color=(80, 80, 80))

        except OSError:
            self.display_error("Unable to find/read file.\n\nCheck that the 'gallery' folder in the root of your SD card contains JPEG images!")
        except IndexError:
            self.display_error("Unable to read images in the 'gallery' folder.\n\nCheck the files are present and are in JPEG format.")

    def clear(self):
        self.display.set_pen(self.BACKGROUND)
        self.display.set_layer(0)
        self.display.clear()
        self.display.set_layer(1)
        self.display.clear()

    def run(self):
        # --- Main loop ---
        last_updated = time.ticks_ms()
        self.clear()
        self.show_image()
        self.presto.update()

        while True:
            self.touch.poll()

            # Update ambient LEDs
            self.update_leds()

            # Screensaver: draw random lines if idle
            now = time.ticks_ms()
            if time.ticks_diff(now, last_updated) > self.SCREENSAVER_DELAY * 1000 and self.lines_drawn < 100:
                choice = random.randint(0, 2)
                if choice == 0:
                    self.effects.draw_random_line()
                elif choice == 1:
                    self.effects.draw_random_circle()
                else:
                    self.effects.draw_radial_line()
                self.lines_drawn += 1

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

            # Small idle sleep to reduce CPU usage
            time.sleep(0.05)

if __name__ == "__main__":
    app = PhotoFrame()
    app.run()
