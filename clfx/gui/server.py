import os
import subprocess
import threading
import uuid
import time
import platform as pf
from flask import Flask, request, send_from_directory, jsonify

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB Limit

# -- Paths ---------------------------------------------------------------
GUI_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(GUI_DIR)

# K_SERVICE is set automatically by GCP Cloud Run; absent = local dev
_is_cloud = bool(os.environ.get('K_SERVICE'))
_is_win   = pf.system() == 'Windows'

# Linux binary inside Docker; .exe on Windows local dev
ENGINE_PATH = os.path.join(ROOT_DIR, 'clfx.exe' if _is_win else 'clfx')

# Cloud Run: /tmp (ephemeral but writable); local: gui subdirs
OUTPUT_DIR  = '/tmp/clfx_output'  if _is_cloud else os.path.join(GUI_DIR, 'output')
UPLOADS_DIR = '/tmp/clfx_uploads' if _is_cloud else os.path.join(GUI_DIR, 'uploads')

GUI_VERSION = "1.1.0"

VALID_EFFECTS = {
    'gain', 'pan', 'eq', 'lowpass', 'distortion', 'bitcrush', 'compress', 'gate',
    'autowah', 'chorus', 'flange', 'phase', 'tremolo', 'widening', 'ringmod',
    'pitch', 'echo', 'pingpong', 'reverb', 'convolve', 'freeze'
}

for d in [OUTPUT_DIR, UPLOADS_DIR]:
    os.makedirs(d, exist_ok=True)

def sanitize_path(path, base_dir):
    """Ensure path is within base_dir and normalized."""
    if not path:
        return None
    full_path = os.path.realpath(os.path.join(base_dir, path))
    if not full_path.startswith(os.path.realpath(base_dir)):
        return None
    return full_path

# ── Persistent daemon worker ─────────────────────────────────────────────────

class ClfxWorker:
    """Wraps the clfx --daemon subprocess.

    Keeps OpenCL context alive across requests so platform/device/program
    initialisation only happens once per container instance.  A threading.Lock
    serialises jobs: Cloud Run scales by adding containers, not threads.
    """

    def __init__(self):
        self._proc = None
        self._lock = threading.Lock()
        self._start()

    def _start(self):
        try:
            self._proc = subprocess.Popen(
                [ENGINE_PATH, '--daemon'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,     # inherit server stderr so logs appear in Cloud Run
                cwd=ROOT_DIR,
                text=True,
                bufsize=1        # line-buffered
            )
            print(f"[worker] Daemon PID {self._proc.pid} started", flush=True)
        except Exception as e:
            print(f"[worker] Failed to start daemon: {e}", flush=True)
            self._proc = None

    def _ensure_alive(self):
        if self._proc is None or self._proc.poll() is not None:
            print("[worker] Daemon exited — restarting", flush=True)
            self._start()

    def _read_response(self, timeout=120):
        """Read one response line with a timeout."""
        import select, sys
        if _is_win:
            # select() doesn't work on Windows pipes; use a thread with timeout
            result = [None]
            def reader():
                try:
                    result[0] = self._proc.stdout.readline()
                except Exception:
                    pass
            t = threading.Thread(target=reader, daemon=True)
            t.start()
            t.join(timeout)
            return result[0]
        else:
            rlist, _, _ = select.select([self._proc.stdout], [], [], timeout)
            if rlist:
                return self._proc.stdout.readline()
            return None

    def process(self, args: list, timeout=120):
        """Send a job to the daemon and return (success, message).

        args = [input_path, output_path, effect, param, ...]
        """
        with self._lock:
            self._ensure_alive()
            if self._proc is None:
                return False, "Engine daemon not available"

            try:
                msg = f"{len(args)}\n" + "\n".join(args) + "\n"
                self._proc.stdin.write(msg)
                self._proc.stdin.flush()
            except Exception as e:
                return False, f"Failed to write to daemon: {e}"

            line = self._read_response(timeout)
            if not line:
                # Daemon timed out or died
                self._proc = None
                return False, "Daemon timed out or died"

            line = line.rstrip('\n')
            if line == "OK":
                return True, ""
            elif line.startswith("ERR: "):
                return False, line[5:]
            else:
                return False, f"Unexpected daemon response: {line!r}"


# One worker per gunicorn worker process (Cloud Run: 1 worker = 1 container)
_worker = ClfxWorker()

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"}), 400

    if file and file.filename.lower().endswith('.wav'):
        safe_filename = f"{uuid.uuid4().hex[:8]}_{os.path.basename(file.filename)}"
        save_path = os.path.join(UPLOADS_DIR, safe_filename)
        file.save(save_path)

        bits, ch, wav_err = None, None, None
        frames, fs, duration = None, None, None
        try:
            import wave
            with wave.open(save_path, 'rb') as wf:
                bits = wf.getsampwidth() * 8
                ch   = wf.getnchannels()
                frames = wf.getnframes()
                fs = wf.getframerate()
                if fs > 0:
                    duration = round(frames / fs, 2)
        except wave.Error as e:
            wav_err = str(e)
        except Exception as e:
            wav_err = str(e)

        if wav_err is not None:
            try: os.remove(save_path)
            except OSError: pass
            err_lower = wav_err.lower()
            if 'unknown format' in err_lower or 'format' in err_lower:
                return jsonify({"success": False,
                    "error": "Engine requires 16-bit PCM WAV, but this file uses a compressed/non-PCM format "
                             f"({wav_err}). Convert with: ffmpeg -i input.wav -acodec pcm_s16le -ar 44100 output.wav"}), 400
            return jsonify({"success": False, "error": f"Invalid WAV file: {wav_err}"}), 400

        if bits != 16:
            try: os.remove(save_path)
            except OSError: pass
            return jsonify({"success": False,
                "error": f"Engine requires 16-bit PCM WAV, but this file is {bits}-bit. "
                         f"Convert with: ffmpeg -i input.wav -acodec pcm_s16le -ar 44100 output.wav"}), 400

        info_str = f"Loaded: {ch}ch 16-bit PCM"
        if duration is not None:
            info_str = f"{ch}ch · {fs}Hz · {duration}s"

        return jsonify({
            "success": True,
            "filename": safe_filename,
            "info": info_str,
            "sample_rate": fs,
            "duration": duration,
            "channels": ch
        })

    return jsonify({"success": False, "error": "Only WAV files allowed"}), 400

@app.route('/')
def index():
    return send_from_directory(GUI_DIR, 'index.html')

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    filename = os.path.basename(filename)
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)

@app.route('/uploads/<filename>', methods=['GET'])
def get_upload_file(filename):
    filename = os.path.basename(filename)
    return send_from_directory(UPLOADS_DIR, filename)

@app.route('/system-info', methods=['GET'])
def get_system_info():
    info = {
        "os": f"{pf.system()} {pf.release()}",
        "architecture": pf.machine(),
        "gui_version": GUI_VERSION,
        "engine_version": "Unknown",
        "engine": "Unknown"
    }
    try:
        result = subprocess.run([ENGINE_PATH, '--info'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            engine_details = []
            for line in lines:
                if "CLFX Engine Version:" in line:
                    info["engine_version"] = line.split(":")[1].strip()
                if line.startswith('Platform') or line.strip().startswith('Device'):
                    engine_details.append(line.strip())
            if engine_details:
                info["engine"] = engine_details
    except Exception as e:
        info["engine"] = f"Probe failed: {str(e)}"

    return jsonify(info)

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    input_file_raw = data.get('inputFile')
    effects = data.get('effects', [])
    output_name_raw = data.get('outputName', 'processed.wav')

    try:
        global_mix = float(data.get('mix', 1.0))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid mix value"}), 400

    # 1. Sanitize input path
    full_input_path = sanitize_path(input_file_raw, UPLOADS_DIR)
    if not full_input_path or not os.path.exists(full_input_path):
        full_input_path = sanitize_path(input_file_raw, ROOT_DIR)
    if not full_input_path or not os.path.exists(full_input_path):
        return jsonify({"error": f"Input file not found: {input_file_raw}"}), 400

    # 2. UUID-prefixed output filename (prevents race conditions under concurrency)
    output_basename = os.path.basename(output_name_raw)
    if not output_basename.endswith('.wav'):
        output_basename += '.wav'
    output_filename = f"{uuid.uuid4().hex[:8]}_{output_basename}"
    full_output_path = os.path.join(OUTPUT_DIR, output_filename)

    # 3. Build args list for the daemon
    args = [full_input_path, full_output_path]

    for fx in effects:
        fx_type = fx.get('type')
        if not fx_type:
            continue
        if fx_type not in VALID_EFFECTS:
            return jsonify({"error": f"Unknown or restricted effect: {fx_type}"}), 400

        args.append(fx_type)

        if fx_type == 'convolve':
            ir_raw = fx.get('ir')
            full_ir_path = sanitize_path(ir_raw, ROOT_DIR)
            if not full_ir_path:
                return jsonify({"error": f"Invalid IR path: {ir_raw}"}), 400
            args.append(full_ir_path)
        else:
            try:
                if 'p1' in fx: args.append(str(float(fx['p1'])))
                if 'p2' in fx: args.append(str(float(fx['p2'])))
            except (ValueError, TypeError):
                return jsonify({"error": f"Invalid numeric parameter in {fx_type}"}), 400

    if abs(global_mix - 1.0) > 0.01:
        args.extend(['gain', str(global_mix)])

    if len(args) < 3:
        return jsonify({"error": "No effects specified"}), 400

    print(f"[process] args: {args}", flush=True)

    # 4. Dispatch to persistent daemon
    success, errmsg = _worker.process(args, timeout=120)

    if success:
        return jsonify({
            "success": True,
            "output": output_filename,
            "audio": f"output/{output_filename}",
            "stdout": ""
        })
    else:
        return jsonify({
            "success": False,
            "error": errmsg or "Processing failed",
            "cmd": ' '.join(args)
        }), 500

@app.route('/visualize', methods=['POST'])
def visualize():
    """Read a WAV file from output dir and return downsampled peak data for browser canvas rendering."""
    data = request.json
    filename = data.get('file', '')
    filename = os.path.basename(filename)
    full_path = os.path.join(OUTPUT_DIR, filename)

    if not os.path.exists(full_path):
        full_path = os.path.join(UPLOADS_DIR, filename)
        if not os.path.exists(full_path):
            return jsonify({"error": f"File not found: {filename}"}), 404

    try:
        import wave, struct
        with wave.open(full_path, 'rb') as wf:
            channels = wf.getnchannels()
            frames = wf.getnframes()
            fs = wf.getframerate()
            raw = wf.readframes(frames)

        fmt = f"<{frames * channels}h"
        samples = struct.unpack(fmt, raw)

        NUM_BLOCKS = 1000
        samples_per_block = max(1, len(samples) // (NUM_BLOCKS * channels))
        peaks_l, peaks_r = [], []
        for b in range(NUM_BLOCKS):
            start = b * samples_per_block * channels
            chunk_l = [abs(samples[i]) / 32768.0 for i in range(start, min(start + samples_per_block * channels, len(samples)), channels)]
            peaks_l.append(max(chunk_l) if chunk_l else 0)
            if channels > 1:
                chunk_r = [abs(samples[i]) / 32768.0 for i in range(start + 1, min(start + samples_per_block * channels + 1, len(samples)), channels)]
                peaks_r.append(max(chunk_r) if chunk_r else 0)

        return jsonify({
            "channels": channels,
            "sample_rate": fs,
            "duration": round(frames / fs, 2),
            "peaks_l": peaks_l,
            "peaks_r": peaks_r if channels > 1 else peaks_l
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(GUI_DIR, path)

if __name__ == '__main__':
    is_debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    print(f"CLFX Dashboard v{GUI_VERSION} (debug={is_debug}) at http://localhost:5000")
    app.run(port=5000, debug=is_debug)
