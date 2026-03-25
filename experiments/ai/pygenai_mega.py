import os
import asyncio
import json
import ffmpeg
import subprocess
import logging
import collections
import time
import requests
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import google.generativeai as genai
import threading
import queue
from typing import Optional, Tuple, List, Dict, Any

# Optional Dependencies
try:
    import pyttsx3
except ImportError:
    pyttsx3 = None

try:
    import winsound
except ImportError:
    winsound = None

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- MEGA SCHEMA ---
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "zoom_recommendation": {"type": "integer", "description": "100 to 500"},
        "risks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "hazard": {"type": "string"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high", "critical"]},
                    "action": {"type": "string"}
                }
            }
        },
        "objects_seen": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "location": {"type": "string", "description": "relative position (top-left, etc)"},
                    "state": {"type": "string"}
                }
            }
        },
        "interesting_facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "fact": {"type": "string"}
                }
            }
        },
        "suggested_automation": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "trigger": {"type": "string"},
                    "webhook_url": {"type": "string"},
                    "reason": {"type": "string"}
                }
            }
        },
        "event_summary": {"type": "string"}
    },
    "required": ["description", "zoom_recommendation", "risks", "objects_seen", "interesting_facts", "event_summary"]
}

# --- MODULES ---

class SceneMemory:
    """Manages persistent history and spatial tracking of objects."""
    def __init__(self, db_path="monitor_history.json", max_entries=100):
        self.db_path = db_path
        self.max_entries = max_entries
        self.history = self._load()

    def _load(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r") as f:
                    return json.load(f)
            except:
                return []
        return []

    def save_snapshot(self, data: Dict, is_query: bool = False):
        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary": data.get("event_summary", ""),
            "objects": data.get("objects_seen", []),
            "description": data.get("description", ""),
            "type": "query_response" if is_query else "scene_log"
        }
        self.history.append(entry)
        if len(self.history) > self.max_entries:
            self.history.pop(0)
        
        try:
            with open(self.db_path, "w") as f:
                json.dump(self.history, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")

    def get_context(self) -> str:
        """Returns a string representation of recent history for the AI."""
        if not self.history:
            return "No prior history recorded."
        last_5 = self.history[-5:]
        context = "Recent Events:\n"
        for h in last_5:
            context += f"- [{h['timestamp']}] {h['summary']}\n"
        return context

class AudioManager:
    """Handles Text-to-Speech (TTS) via a thread-safe queue."""
    def __init__(self):
        self.queue = queue.Queue()
        self.engine = None
        self._stop_event = threading.Event()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        
        if pyttsx3:
            try:
                self.thread.start()
            except:
                logger.warning("Audio worker failed to start.")

    def _worker(self):
        """Dedicated thread worker for pyttsx3 (not thread-safe)."""
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 150)
        except Exception as e:
            logger.error(f"TTS Init Error: {e}")
            return

        while not self._stop_event.is_set():
            try:
                text = self.queue.get(timeout=1.0)
                if self.engine:
                    logger.info(f"AI VOICE: {text}")
                    self.engine.say(text)
                    self.engine.runAndWait()
                else:
                    print(f">>> AI (TTS Init Failed): {text}")
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"TTS Runtime Error: {e}")

    def say(self, text: str):
        if pyttsx3:
            self.queue.put(text)
        else:
            print(f">>> AI: {text}")

    def stop(self):
        self._stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2)

# --- MAIN ENGINE ---

class GeminiMegaMonitor:
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={"response_mime_type": "application/json"}
        )
        self.memory = SceneMemory()
        self.audio = AudioManager()
        self.current_zoom = 100
        self.is_running = False
        self.device = "/dev/video0" # Default
        
    def _capture_frame(self, filename: str):
        """Captures a frame using ffmpeg with Windows/Linux fallbacks."""
        # Linux / Dev interface
        try:
            (
                ffmpeg
                .input(self.device, f='v4l2', framerate='15', video_size='1280x720')
                .output(filename, vframes=1, pix_fmt='yuv420p', loglevel='error')
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            return
        except:
            pass

        # Windows Fallbacks
        win_devices = ["video=Integrated Camera", "video=USB Video Device", "video=HD Web Camera"]
        for dev in win_devices:
            try:
               subprocess.run([
                   "ffmpeg", "-y", "-f", "dshow", "-i", dev, 
                   "-vframes", "1", "-loglevel", "error", filename
               ], check=True, capture_output=True, timeout=5)
               return
            except:
               continue
        
        logger.error("All camera capture methods failed.")

    def _add_overlay(self, filename: str, insights: Dict):
        img = Image.open(filename).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        try: font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
        except: font = ImageFont.load_default()

        def draw_pill(x, y, text, color):
            bbox = draw.textbbox((x, y), text, font=font)
            p = 6
            draw.rectangle([bbox[0]-p, bbox[1]-p, bbox[2]+p, bbox[3]+p], fill=color, outline=(255,255,255,50))
            draw.text((x, y), text, font=font, fill=(255,255,255,255))
            return bbox[3] + p + 10

        curr_y = 20
        curr_y = draw_pill(20, curr_y, f"AI SIDEKICK ACTIVE | ZOOM: {self.current_zoom}", (0, 0, 0, 180))
        
        for risk in insights.get("risks", []):
            color = (200, 0, 0, 200) if risk["severity"] in ["high", "critical"] else (180, 150, 0, 180)
            curr_y = draw_pill(20, curr_y, f"ALERT: {risk['hazard']}", color)

        if insights.get("interesting_facts"):
            fact = insights["interesting_facts"][0]
            draw_pill(20, img.height - 50, f"FACT: {fact['subject']} - {fact['fact'][:60]}", (0, 100, 50, 200))

        combined = Image.alpha_composite(img, overlay)
        combined.convert("RGB").save(filename)

    async def analyze_frame(self, img_path: str, user_query: Optional[str] = None) -> Dict:
        img = Image.open(img_path)
        history_context = self.memory.get_context()
        current_time = datetime.now().strftime("%H:%M:%S")
        
        prompt = f"""
        You are an advanced AI Sidekick monitoring this webcam.
        TIME: {current_time}
        
        CONTEXT (Memory):
        {history_context}
        
        USER COMMAND (Optional): {user_query if user_query else "None"}
        
        GOAL: Be a proactive, helpful desk assistant.
        
        TASKS:
        1. SPATIAL TRACKING: List key objects (phone, mug, keys, etc) and their relative positions.
        2. RISK/AUTOMATION: If a hazardous event or a known trigger condition is met, specify it.
        3. OCR/SIDEKICK: If the user is holding a document, book, or screen, EXTRACT the text or summarize it.
        4. QUERY RESPONSE: If a 'USER COMMAND' is present, prioritize answering it accurately using history + current view.
        
        SCHEMA:
        {json.dumps(RESPONSE_SCHEMA, indent=2)}
        """

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: self.model.generate_content([prompt, img]))
            data = json.loads(response.text)
            
            # Non-blocking save
            loop.run_in_executor(None, self.memory.save_snapshot, data, user_query is not None)
            
            return data
        except Exception as e:
            logger.error(f"AI Error: {e}")
            return {"description": "Error", "risks": [], "objects_seen": [], "interesting_facts": [], "event_summary": "Error"}

    async def user_interface_loop(self):
        """Asynchronous CLI to allow real-time questioning of the AI."""
        print("\n[AI SIDEKICK READY] You can ask me questions! (e.g., 'Where is my coffee?', 'What happened at 8 PM?')")
        print("Type 'exit' to stop.\n")
        
        loop = asyncio.get_event_loop()
        while self.is_running:
            try:
                query = await loop.run_in_executor(None, input, "USER > ")
                if query.lower() in ["exit", "quit"]:
                    self.is_running = False
                    break
                
                print(f"[*] Analyzing query: '{query}'...")
                # Capture a fresh frame for context
                filename = "query_context.jpg"
                self._capture_frame(filename)
                
                insights = await self.analyze_frame(filename, user_query=query)
                response = insights.get("description", "I'm not sure.")
                
                # Speak answer
                self.audio.say(response)
                print(f"AI > {response}\n")
                
            except Exception as e:
                logger.error(f"UI Error: {e}")

    async def run(self):
        self.is_running = True
        self.audio.say("Mega Engine Online. I am watching.")
        
        # Start the UI loop as a background task
        asyncio.create_task(self.user_interface_loop())
        
        while self.is_running:
            try:
                filename = "latest_mega.jpg"
                self._capture_frame(filename)
                
                insights = await self.analyze_frame(filename)
                self._add_overlay(filename, insights)
                
                # Proactive Audio Alerts
                if insights.get("risks"):
                    critical = [r for r in insights["risks"] if r["severity"] == "critical"]
                    if critical:
                        self.audio.say(f"Warning! I detect a {critical[0]['hazard']}. {critical[0]['action']}")
                    else:
                        if winsound: winsound.Beep(600, 150)

                # Automation logic
                for action in insights.get("suggested_automation", []):
                    logger.info(f"TRIGGER: {action['trigger']} ({action['reason']})")
                    # requests.post(action['webhook_url'], json={"event": action['trigger']})

                await asyncio.sleep(10)
            except KeyboardInterrupt:
                self.is_running = False
            except Exception as e:
                logger.error(f"Loop error: {e}")
                await asyncio.sleep(10)

if __name__ == "__main__":
    monitor = GeminiMegaMonitor()
    asyncio.run(monitor.run())
