import subprocess, time, PIL.Image
import google.generativeai as genai

genai.configure(api_key="AIzaSyDWEoHyGo1wuyWnDk16VrUj731ZN9Yq4dk")


for m in genai.list_models():
    print(m.name)

model = genai.GenerativeModel("models/gemini-pro-latest")

while True:
    # Capture one frame
    subprocess.run([
        "ffmpeg", "-f", "v4l2", "-i", "/dev/video0",
        "-frames:v", "1", "snapshot.jpg", "-y"
    ], check=True)

    # Send to Gemini
    img = PIL.Image.open("snapshot.jpg")
    response = model.generate_content(["What's happening?", img])
    print(response.text)

    time.sleep(5)  # wait before next frame


