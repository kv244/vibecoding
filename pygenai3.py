
import ffmpeg
import subprocess
import time
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import google.generativeai as genai

# Configure Gemini with your API key
genai.configure(api_key="AIzaSyD9f6Dlo_t1j88xGwvwLI9eKqoJFjo83ko")

# Default to Gemini 2.5 (flash variant if available)
DEFAULT_MODEL = "models/gemini-2.5-flash"
model = genai.GenerativeModel(DEFAULT_MODEL)

def set_zoom(value):
    """Set webcam zoom via v4l2-ctl."""
    subprocess.run([
        "v4l2-ctl", "-d", "/dev/video0",
        f"--set-ctrl=zoom_absolute={value}"
    ], check=True)

def capture_frame(filename):
    """Capture one frame from webcam and save to filename."""
    (
        ffmpeg
        .input('/dev/video0', f='v4l2')
        .output(filename, vframes=1, pix_fmt='yuv420p')
        .run(capture_stdout=True, capture_stderr=True)
    )

def add_timestamp(filename):
    """Overlay timestamp text onto the image."""
    img = Image.open(filename)
    draw = ImageDraw.Draw(img)

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 24)
    except:
        font = ImageFont.load_default()

    # Use textbbox for accurate text size
    bbox = draw.textbbox((0, 0), ts, font=font)
    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]

    x, y = img.width - text_w - 10, img.height - text_h - 10
    draw.text((x, y), ts, font=font, fill="white")

    img.save(filename)
    return img, ts

while True:
    for zoom in [50, 100, 150, 200]:
        set_zoom(zoom)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_zoom{zoom}_{ts}.jpg"

        capture_frame(filename)
        img, ts_overlay = add_timestamp(filename)

        # Richer prompt with explicit zoom level included
        prompt = (
            f"Analyze this webcam frame captured at zoom level {zoom}. "
            f"Provide a detailed description of visible objects, people, colors, "
            f"lighting conditions, and any notable patterns or textures. "
            f"Also mention the timestamp overlay ({ts_overlay}) and explain how "
            f"the zoom level {zoom} affects clarity, composition, and context."
        )

        response = model.generate_content([prompt, img])
        print(f"Gemini 2.5 says (zoom {zoom}):", response.text)

        time.sleep(5)
