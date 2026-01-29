import subprocess, time
import google.generativeai as genai
from PIL import Image

# 1. Configure with your API key
genai.configure(api_key="")

# 2. List available models
print("Available models:")
models = list(genai.list_models())
for m in models:
    print(" -", m.name)

# 3. Pick a model automatically (prefer flash if available)
model_name = None
for m in models:
    if "gemini-2.5-flash" in m.name:
        model_name = m.name
        break
if not model_name:
    model_name = models[0].name  # fallback to first available

print(f"\nUsing model: {model_name}")
model = genai.GenerativeModel(model_name)

# 4. Loop: capture webcam frame and send to Gemini
while True:
    # Capture one frame from webcam
    subprocess.run([
        "ffmpeg", "-loglevel", "error", "-f", "v4l2", "-i", "/dev/video0",
        "-frames:v", "1", "snapshot.jpg", "-y"
    ], check=True)

    # Open image with Pillow
    img = Image.open("snapshot.jpg")

    # Send to Gemini
    response = model.generate_content(
        ["Describe this webcam frame:", img]
    )
    print("Gemini says:", response.text)

    time.sleep(5)  # wait before next frame
