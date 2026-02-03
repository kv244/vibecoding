import os
import asyncio
import json
import ffmpeg
import subprocess
import logging
import collections
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import google.generativeai as genai
from typing import Optional, Tuple, List, Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# JSON Schema for Gemini Response
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
        "interesting_facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string"},
                    "fact": {"type": "string"},
                    "confidence": {"type": "number"}
                }
            }
        },
        "event_summary": {"type": "string", "description": "One sentence summary of what changed since last frame"}
    },
    "required": ["description", "zoom_recommendation", "risks", "interesting_facts", "event_summary"]
}

class GeminiAdvancedMonitor:
    """
    webcam monitor with Advanced AI Reasoning: Risk detection, Fact extraction, and Scene Memory.
    """
    def __init__(self, model_name: str = "gemini-2.0-flash", device: str = "/dev/video0"):
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            logger.warning("GOOGLE_API_KEY not found. Ensure it is set for real use.")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=model_name,
            generation_config={"response_mime_type": "application/json"}
        )
        self.device = device
        self.current_zoom = 100
        self.is_running = False
        self.history = collections.deque(maxlen=10) # Store last 10 event summaries
        
        # Audio alert toggle (can be expanded to play sound)
        self.alert_enabled = True

    def _set_hardware_zoom(self, value: int):
        """Attempts to set hardware zoom via v4l2-ctl."""
        try:
            subprocess.run([
                "v4l2-ctl", "-d", self.device,
                f"--set-ctrl=zoom_absolute={value}"
            ], check=True, capture_output=True)
            logger.debug(f"Hardware zoom sync: {value}")
        except:
            pass # Silent fail if camera doesn't support zoom

    def _capture_frame(self, filename: str):
        """Captures a single frame using ffmpeg."""
        try:
            (
                ffmpeg
                .input(self.device, f='v4l2', framerate='15', video_size='1280x720')
                .output(filename, vframes=1, pix_fmt='yuv420p', loglevel='error')
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            logger.error(f"Capture error: {e.stderr.decode() if e.stderr else 'Unknown'}")
            raise

    def _add_overlay(self, filename: str, zoom: int, insights: Optional[Dict] = None):
        """Overlays timestamp, zoom, and critical alerts onto the image."""
        img = Image.open(filename).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Font setup
        try: font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
        except: font = ImageFont.load_default()

        ts = datetime.now().strftime("%H:%M:%S")
        
        # 1. Base Info
        info_text = f"[{ts}] ZOOM: {zoom}"
        draw.text((20, 20), info_text, font=font, fill=(255, 255, 255, 255), stroke_width=2, stroke_fill=(0,0,0,255))

        # 2. Risk Overlays
        if insights and insights.get("risks"):
            y_offset = 60
            for risk in insights["risks"]:
                severity = risk["severity"].upper()
                msg = f"!!! {severity}: {risk['hazard']} !!!"
                # Red for critical/high, yellow for medium
                color = (255, 0, 0, 255) if severity in ["CRITICAL", "HIGH"] else (255, 255, 0, 255)
                draw.text((20, y_offset), msg, font=font, fill=color, stroke_width=2, stroke_fill=(0,0,0,255))
                y_offset += 35

        # 3. Fact Highlights
        if insights and insights.get("interesting_facts"):
            fact = insights["interesting_facts"][0] # Just show first one
            draw.text((20, img.height - 50), f"FACT: {fact['subject']} - {fact['fact'][:50]}...", 
                      font=font, fill=(0, 255, 100, 255), stroke_width=1, stroke_fill=(0,0,0,255))

        combined = Image.alpha_composite(img, overlay)
        combined.convert("RGB").save(filename)

    async def analyze_scene(self, img: Image.Image) -> Dict[str, Any]:
        """
        Multimodal reasoning engine: Risk, Facts, and Temporal Context.
        """
        history_context = "\n".join(list(self.history))
        prompt = f"""
        Analyze this webcam frame with high-level reasoning.
        CONTEXT (Past Events):
        {history_context if history_context else "Monitoring started."}

        TASKS:
        1. General Description: What is happening now?
        2. RISK IDENTIFICATION: Check for fire, smoke, intruders, safety violations, or suspicious activity.
        3. FACT EXTRACTION: Identify specific objects (plants, gadgets, books) and provide a concise 'fact of interest'.
        4. ADAPTIVE ZOOM: Recommend a zoom level (100-500). Zoom IN for small details/risks. Zoom OUT if view is blocked.
        5. TEMPORAL ANALYSIS: What changed since the 'Past Events' listed above?

        OUTPUT SCHEMA:
        {json.dumps(RESPONSE_SCHEMA, indent=2)}
        """

        try:
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, lambda: self.model.generate_content([prompt, img]))
            data = json.loads(response.text)
            
            # Update history
            self.history.append(data.get("event_summary", "Stable scene."))
            return data
        except Exception as e:
            logger.error(f"AI Reasoning Error: {e}")
            return {"description": "Analysis failed", "zoom_recommendation": 100, "risks": [], "interesting_facts": [], "event_summary": "Error"}

    async def run(self):
        self.is_running = True
        logger.info("AI Monitor Active [Risk + Fact Engine]")
        
        insights = None
        while self.is_running:
            try:
                self._set_hardware_zoom(self.current_zoom)
                
                filename = f"monitor_latest.jpg"
                self._capture_frame(filename)
                
                # Analyze image
                img = Image.open(filename)
                insights = await self.analyze_scene(img)
                
                # Apply overlay with insights
                self._add_overlay(filename, self.current_zoom, insights)
                
                # Update Zoom logic
                next_zoom = insights.get("zoom_recommendation", 100)
                if next_zoom != self.current_zoom:
                    logger.info(f"AI requested zoom pivot: {self.current_zoom} -> {next_zoom}")
                    self.current_zoom = next_zoom

                # Print Log
                print("\n" + "="*50)
                print(f"SCENE: {insights['description']}")
                for risk in insights['risks']:
                    print(f"--- [!] RISK [{risk['severity']}]: {risk['hazard']} -> {risk['action']}")
                for fact in insights['interesting_facts']:
                    print(f"--- [?] FACT: {fact['subject']}: {fact['fact']}")
                print("="*50 + "\n")

                await asyncio.sleep(8) # Polling rate
                
            except KeyboardInterrupt:
                self.is_running = False
            except Exception as e:
                logger.error(f"Monitor loop failure: {e}")
                await asyncio.sleep(10)

if __name__ == "__main__":
    monitor = GeminiAdvancedMonitor()
    try:
        asyncio.run(monitor.run())
    except KeyboardInterrupt:
        pass
