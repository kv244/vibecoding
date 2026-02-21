import os
import subprocess
import uuid
import time
import platform as pf
from flask import Flask, request, send_from_directory, jsonify

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB Limit

# Base path for the clfx engine - use absolute script directory
GUI_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(GUI_DIR)
ENGINE_PATH = os.path.join(ROOT_DIR, 'clfx.exe')
OUTPUT_DIR = os.path.join(GUI_DIR, 'output')
UPLOADS_DIR = os.path.join(GUI_DIR, 'uploads')

GUI_VERSION = "1.0.1"

VALID_EFFECTS = {'gain', 'echo', 'lowpass', 'bitcrush', 'tremolo', 'widening',
                 'pingpong', 'chorus', 'autowah', 'distortion', 'ringmod',
                 'pitch', 'gate', 'pan', 'eq', 'freeze', 'convolve', 'compress',
                 'reverb', 'flange', 'phase', 'lowpass'}

for d in [OUTPUT_DIR, UPLOADS_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

def sanitize_path(path, base_dir):
    """Ensure path is within base_dir and normalized."""
    if not path:
        return None
    # Join and resolve
    full_path = os.path.realpath(os.path.join(base_dir, path))
    # Check if full_path starts with base_dir (jail check)
    if not full_path.startswith(os.path.realpath(base_dir)):
        return None
    return full_path

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"}), 400
    
    if file and file.filename.lower().endswith('.wav'):
        # Add random prefix to prevent collisions
        safe_filename = f"{uuid.uuid4().hex[:8]}_{os.path.basename(file.filename)}"
        save_path = os.path.join(UPLOADS_DIR, safe_filename)
        file.save(save_path)

        # Validate WAV is 16-bit PCM (engine requirement)
        # NOTE: all os.remove() calls must happen AFTER the `with` block so
        #       the file handle is closed first (Windows does not allow deleting open files).
        bits, ch, wav_err = None, None, None
        try:
            import wave
            with wave.open(save_path, 'rb') as wf:
                bits = wf.getsampwidth() * 8
                ch   = wf.getnchannels()
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


        return jsonify({"success": True, "filename": safe_filename,
                        "info": f"Loaded: {ch}ch 16-bit PCM"})

    return jsonify({"success": False, "error": "Only WAV files allowed"}), 400

@app.route('/')
def index():
    return send_from_directory(GUI_DIR, 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(GUI_DIR, path)

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
        # Run engine with --info
        result = subprocess.run([ENGINE_PATH, '--info'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            # Parse engine output for relevant lines
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

    # 1. Sanitize Paths - Prioritize UPLOADS_DIR
    full_input_path = sanitize_path(input_file_raw, UPLOADS_DIR)
    if not full_input_path or not os.path.exists(full_input_path):
        full_input_path = sanitize_path(input_file_raw, ROOT_DIR)
    
    if not full_input_path or not os.path.exists(full_input_path):
        return jsonify({"error": f"Input file not found: {input_file_raw}"}), 400

    # Ensure output name is just a filename, or sanitize it within OUTPUT_DIR
    output_filename = os.path.basename(output_name_raw)
    if not output_filename.endswith('.wav'):
        output_filename += '.wav'
    full_output_path = os.path.join(OUTPUT_DIR, output_filename)

    # 2. Build Command Line
    cmd = [ENGINE_PATH, full_input_path, full_output_path]

    for fx in effects:
        fx_type = fx.get('type')
        if not fx_type: continue
        
        # Whitelist validation
        if fx_type not in VALID_EFFECTS:
            return jsonify({"error": f"Unknown or restricted effect: {fx_type}"}), 400
        
        cmd.append(fx_type)
        
        if fx_type == 'convolve':
            ir_raw = fx.get('ir')
            # IR files can be in ROOT_DIR
            full_ir_path = sanitize_path(ir_raw, ROOT_DIR)
            if not full_ir_path:
                return jsonify({"error": f"Invalid IR path: {ir_raw}"}), 400
            cmd.append(full_ir_path)
        else:
            try:
                if 'p1' in fx: cmd.append(str(float(fx['p1'])))
                if 'p2' in fx: cmd.append(str(float(fx['p2'])))
            except (ValueError, TypeError):
                return jsonify({"error": f"Invalid numeric parameter in {fx_type}"}), 400

    # Apply master mix as a final gain stage if not at unity (engine has no --mix flag)
    if abs(global_mix - 1.0) > 0.01:
        cmd.extend(['gain', str(global_mix)])

    try:
        print(f"Executing: {' '.join(cmd)}")
        # Run engine from project root with timeout to prevent hanging
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT_DIR, timeout=120)
        
        if result.returncode == 0:
            # Generate visualization
            # visualize.py is in root, needs path to output relative to root
            rel_output_path = os.path.relpath(full_output_path, ROOT_DIR)
            viz_cmd = ['python', 'visualize.py', rel_output_path]
            try:
                viz_result = subprocess.run(viz_cmd, capture_output=True, text=True, cwd=ROOT_DIR, timeout=30)
                if viz_result.returncode != 0:
                    print(f"Visualization failed: {viz_result.stderr}")
            except subprocess.TimeoutExpired:
                print("Visualization timed out")

            # Move waveform.png to output if it's in the root
            root_waveform = os.path.join(ROOT_DIR, 'waveform.png')
            if os.path.exists(root_waveform):
                target_waveform = os.path.join(OUTPUT_DIR, 'waveform.png')
                if os.path.exists(target_waveform):
                    os.remove(target_waveform)
                os.rename(root_waveform, target_waveform)

            return jsonify({
                "success": True, 
                "output": output_filename,
                "waveform": "output/waveform.png",
                "audio": f"output/{output_filename}",
                "stdout": result.stdout
            })
        else:
            diag = f"Exit code {result.returncode}"
            if result.stderr.strip(): diag += f" | stderr: {result.stderr.strip()}"
            if result.stdout.strip(): diag += f" | stdout: {result.stdout.strip()}"
            if not result.stderr.strip() and not result.stdout.strip():
                diag += " | No output — engine binary may be missing or not compiled for this platform."
            return jsonify({
                "success": False,
                "error": diag,
                "cmd": ' '.join(cmd)
            }), 500
    except subprocess.TimeoutExpired:
        return jsonify({"success": False, "error": "Processing timed out after 120 seconds"}), 504
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/visualize', methods=['POST'])
def visualize():
    """Read a WAV file from output dir and return downsampled peak data for browser canvas rendering."""
    data = request.json
    filename = data.get('file', '')
    filename = os.path.basename(filename)  # Safety: strip paths
    full_path = os.path.join(OUTPUT_DIR, filename)

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

        # Downsample to ~1000 peak blocks for canvas
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


if __name__ == '__main__':
    is_debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    print(f"CLFX Dashboard starting (debug={is_debug}) at http://localhost:5000")
    app.run(port=5000, debug=is_debug)
