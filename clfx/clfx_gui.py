import os
import sys
import subprocess
import threading
import wave
import struct
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

if os.name == 'nt':
    import winsound

class CLFXGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("CLFX Audio Effects Engine")
        self.root.geometry("1000x700")

        # Effect definitions
        self.effects = {
            "gain": {"params": [("multiplier", 0.1, 2.0, 1.0)]},
            "pan": {"params": [("value", -1.0, 1.0, 0.0)]},
            "eq": {"params": [("center_freq", 0.0, 1.0, 0.5), ("gain", 0.0, 10.0, 1.0)]},
            "lowpass": {"params": [("strength", 0.0, 1.0, 0.5)]},
            "distortion": {"params": [("drive", 2.0, 10.0, 5.0)]},
            "bitcrush": {"params": [("bits", 1, 16, 8)]},
            "compress": {"params": [("threshold", 0.0, 1.0, 0.5), ("ratio", 1.0, 20.0, 2.0)]},
            "gate": {"params": [("threshold", 0.0, 1.0, 0.5), ("reduction", 0.0, 1.0, 0.5)]},
            "autowah": {"params": [("depth", 0.0, 1.0, 0.5), ("rate", 0.0, 1.0, 0.5)]},
            "chorus": {"params": []},
            "flange": {"params": [("depth", 0.0, 1.0, 0.5), ("feedback", 0.0, 1.0, 0.5)]},
            "phase": {"params": [("depth", 0.0, 1.0, 0.5), ("rate", 0.0, 1.0, 0.5)]},
            "tremolo": {"params": [("freq", 0.1, 20.0, 5.0), ("depth", 0.0, 1.0, 0.5)]},
            "widening": {"params": [("width", 0.0, 2.0, 1.0)]},
            "ringmod": {"params": [("freq", 10.0, 1000.0, 440.0)]},
            "pitch": {"params": [("ratio", 0.5, 2.0, 1.0)]},
            "echo": {"params": [("delay", 100, 5000, 500), ("decay", 0.0, 1.0, 0.5)]},
            "pingpong": {"params": [("delay", 100, 5000, 500), ("decay", 0.0, 1.0, 0.5)]},
            "reverb": {"params": [("size", 0.0, 1.0, 0.5), ("mix", 0.0, 1.0, 0.3)]},
            "convolve": {"params": [("ir_file",)]},
            "freeze": {"params": [("amount", 0.0, 1.0, 0.5), ("randomness", 0.0, 1.0, 0.5)]}
        }

        # Daemon process
        self.daemon = None
        self.lock = threading.Lock()

        # Store original input waveform path for redrawing
        self._input_waveform_path = None
        # Track the last successfully processed output file for Play
        self._last_processed_path = None
        self._playback_thread = None

        # GUI setup
        self.setup_ui()

        # Start daemon
        self.start_daemon()

    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Input file section
        input_frame = ttk.LabelFrame(main_frame, text="Input File", padding="10")
        input_frame.pack(fill=tk.X, pady=5)

        ttk.Label(input_frame, text="Input WAV:").grid(row=0, column=0, sticky=tk.W)
        self.input_path = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.input_path, width=50).grid(row=0, column=1, sticky=tk.W)
        ttk.Button(input_frame, text="Browse...", command=self.browse_input).grid(row=0, column=2, padx=5)

        self.file_info = tk.StringVar()
        ttk.Label(input_frame, textvariable=self.file_info).grid(row=1, column=0, columnspan=3, sticky=tk.W)

        # Output file section
        output_frame = ttk.LabelFrame(main_frame, text="Output File", padding="10")
        output_frame.pack(fill=tk.X, pady=5)

        ttk.Label(output_frame, text="Output WAV:").grid(row=0, column=0, sticky=tk.W)
        self.output_path = tk.StringVar()
        ttk.Entry(output_frame, textvariable=self.output_path, width=50).grid(row=0, column=1, sticky=tk.W)
        ttk.Button(output_frame, text="Browse...", command=self.browse_output).grid(row=0, column=2, padx=5)

        # Effect chain section
        effect_frame = ttk.LabelFrame(main_frame, text="Effect Chain", padding="10")
        effect_frame.pack(fill=tk.X, pady=5)

        # Effect list
        list_frame = ttk.Frame(effect_frame)
        list_frame.pack(fill=tk.X)

        self.effect_listbox = tk.Listbox(list_frame, height=6)
        self.effect_listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.effect_listbox.bind('<<ListboxSelect>>', lambda e: self.update_param_frame())

        scrollbar = ttk.Scrollbar(list_frame, command=self.effect_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.effect_listbox.config(yscrollcommand=scrollbar.set)

        # Effect controls
        control_frame = ttk.Frame(effect_frame)
        control_frame.pack(fill=tk.X, pady=5)

        self.effect_var = tk.StringVar()
        effect_dropdown = ttk.Combobox(control_frame, textvariable=self.effect_var,
                                      values=list(self.effects.keys()), state="readonly")
        effect_dropdown.pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, text="Add Effect", command=self.add_effect).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Remove Effect", command=self.remove_effect).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Move Up", command=self.move_up).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Move Down", command=self.move_down).pack(side=tk.LEFT, padx=5)

        # Effect parameters frame
        self.param_frame = ttk.Frame(effect_frame)
        self.param_frame.pack(fill=tk.X, pady=5)

        # Process / Play buttons
        process_frame = ttk.Frame(main_frame)
        process_frame.pack(fill=tk.X, pady=5)
        ttk.Button(process_frame, text="Process", command=self.process).pack(side=tk.LEFT, padx=5)
        ttk.Button(process_frame, text="Play", command=self.play_output).pack(side=tk.LEFT, padx=5)
        ttk.Button(process_frame, text="Stop", command=self.stop_playback).pack(side=tk.LEFT, padx=5)

        # Status/log area
        log_frame = ttk.LabelFrame(main_frame, text="Status/Log", padding="10")
        log_frame.pack(fill=tk.X, pady=5)

        self.log_text = ScrolledText(log_frame, height=5)
        self.log_text.pack(fill=tk.X)

        # Waveform preview — fixed height so it's always visible
        preview_frame = ttk.LabelFrame(main_frame, text="Waveform Preview", padding="10")
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.preview_canvas = tk.Canvas(preview_frame, bg="white", height=120)
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)

        # Tooltips
        self.tooltips = {
            "gain": "Adjust the overall volume (multiplier).",
            "pan": "Pan the audio left (-1.0) to right (1.0).",
            "eq": "Equalizer: center frequency (0.0-1.0) and gain (0.0-10.0).",
            "lowpass": "Low-pass filter strength (0.0-1.0).",
            "distortion": "Distortion drive amount (2.0-10.0).",
            "bitcrush": "Bit depth reduction (1-16).",
            "compress": "Compression: threshold (0.0-1.0) and ratio (1.0-20.0).",
            "gate": "Noise gate: threshold (0.0-1.0) and reduction (0.0-1.0).",
            "autowah": "Auto-wah: depth (0.0-1.0) and rate (0.0-1.0).",
            "chorus": "Chorus effect (no parameters).",
            "flange": "Flanger: depth (0.0-1.0) and feedback (0.0-1.0).",
            "phase": "Phaser: depth (0.0-1.0) and rate (0.0-1.0).",
            "tremolo": "Tremolo: frequency (Hz) and depth (0.0-1.0).",
            "widening": "Stereo widening (0.0-2.0).",
            "ringmod": "Ring modulator: frequency (Hz).",
            "pitch": "Pitch shift: ratio (0.5-2.0).",
            "echo": "Echo: delay (samples) and decay (0.0-1.0).",
            "pingpong": "Ping-pong delay: delay (samples) and decay (0.0-1.0).",
            "reverb": "Reverb: size (0.0-1.0) and mix (0.0-1.0).",
            "convolve": "Convolution reverb: impulse response file.",
            "freeze": "Freeze effect: amount (0.0-1.0) and randomness (0.0-1.0)."
        }

        for effect, tooltip in self.tooltips.items():
            effect_dropdown.bind('<Enter>', lambda e, t=tooltip: self.show_tooltip(e, t))
            effect_dropdown.bind('<Leave>', lambda e: self.hide_tooltip())

        # Menu bar
        self.setup_menu()

    def setup_menu(self):
        menubar = tk.Menu(self.root)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="About / Engine Info", command=self.show_engine_info)
        menubar.add_cascade(label="Help", menu=help_menu)

        self.root.config(menu=menubar)

    def show_engine_info(self):
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            exe_name = "clfx.exe" if os.name == "nt" else "clfx"
            exe_path = os.path.join(script_dir, exe_name)
            if not os.path.exists(exe_path):
                exe_path = exe_name  # fall back to PATH

            result = subprocess.run(
                [exe_path, "--info"],
                capture_output=True, text=True, timeout=10
            )

            dialog = tk.Toplevel(self.root)
            dialog.title("About / Engine Info")
            dialog.geometry("600x400")
            dialog.transient(self.root)
            dialog.grab_set()

            text_frame = ttk.Frame(dialog)
            text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            text = ScrolledText(text_frame, wrap=tk.WORD)
            text.pack(fill=tk.BOTH, expand=True)

            text.insert(tk.END, "GUI Version: 1.1.0\n\n")
            text.insert(tk.END, "Engine Version / OpenCL / GPU Info:\n")
            text.insert(tk.END, result.stdout or "(no output)")
            if result.stderr:
                text.insert(tk.END, f"\n{result.stderr}")
            text.insert(tk.END, f"\n\nEffects Supported ({len(self.effects)}):\n")
            for name in self.effects:
                text.insert(tk.END, f"  {name}\n")

            text.config(state=tk.DISABLED)

            ttk.Button(dialog, text="OK", command=dialog.destroy).pack(pady=5)

        except subprocess.TimeoutExpired:
            messagebox.showerror("Error", "Engine info command timed out")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to get engine info: {str(e)}")

    def play_output(self):
        output_path = self._last_processed_path or self.output_path.get()
        if not output_path:
            messagebox.showerror("Error", "No processed file available. Run Process first.")
            return
        if not os.path.exists(output_path):
            messagebox.showerror("Error", "Processed file not found. Run Process first.")
            return
        self.stop_playback()  # stop any current playback first
        if os.name == 'nt':
            def _play():
                try:
                    winsound.PlaySound(output_path, winsound.SND_FILENAME)
                except Exception:
                    pass
            self._playback_thread = threading.Thread(target=_play, daemon=True)
            self._playback_thread.start()
        elif sys.platform == 'darwin':
            subprocess.Popen(['afplay', output_path])
        else:
            subprocess.Popen(['aplay', output_path])

    def stop_playback(self):
        if os.name == 'nt':
            try:
                winsound.PlaySound(None, winsound.SND_PURGE)
            except Exception:
                pass

    def show_tooltip(self, event, text):
        self.tooltip = tk.Toplevel(self.root)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{event.x_root + 15}+{event.y_root + 15}")
        label = ttk.Label(self.tooltip, text=text, padding=5, relief="solid")
        label.pack()
        self.tooltip_label = label

    def hide_tooltip(self):
        if hasattr(self, 'tooltip'):
            self.tooltip.destroy()

    def browse_input(self):
        filepath = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if filepath:
            self.input_path.set(filepath)
            self._input_waveform_path = filepath
            self.update_file_info(filepath)
            self.update_output_path(filepath)
            self.update_waveform(filepath)

    def browse_output(self):
        filepath = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV files", "*.wav")])
        if filepath:
            self.output_path.set(filepath)

    def update_file_info(self, filepath):
        try:
            with wave.open(filepath, 'rb') as wav:
                frames = wav.getnframes()
                rate = wav.getframerate()
                channels = wav.getnchannels()
                duration = frames / float(rate)
                info = f"Duration: {duration:.2f}s, Channels: {channels}, Sample Rate: {rate}Hz"
                self.file_info.set(info)
        except Exception as e:
            self.file_info.set(f"Error: {str(e)}")

    def update_output_path(self, input_path):
        base = os.path.splitext(input_path)[0]
        self.output_path.set(f"{base}_processed.wav")

    def add_effect(self):
        effect_name = self.effect_var.get()
        if not effect_name:
            return

        params = self.effects[effect_name]["params"]
        param_values = []

        if params:
            param_frame = ttk.Frame(self.param_frame)
            param_frame.pack(fill=tk.X, pady=2)

            entries = []
            for i, param in enumerate(params):
                if len(param) == 1:
                    param_name = param[0]
                    min_val = max_val = default = None
                else:
                    param_name, min_val, max_val, default = param

                frame = ttk.Frame(param_frame)
                frame.pack(fill=tk.X, pady=2)

                label = ttk.Label(frame, text=f"{param_name}:", width=15)
                label.pack(side=tk.LEFT)

                if param_name == "ir_file":
                    entry = ttk.Entry(frame, width=40)
                    entry.pack(side=tk.LEFT, expand=True, fill=tk.X)
                    button = ttk.Button(frame, text="Browse...", command=lambda e=entry: self.browse_ir(e))
                    button.pack(side=tk.LEFT, padx=5)
                else:
                    entry = ttk.Spinbox(frame, from_=min_val, to=max_val, increment=(max_val-min_val)/100, width=10)
                    entry.set(default)
                    entry.pack(side=tk.LEFT)

                entries.append((param_name, entry))

            param_values = [(name, entry.get()) for name, entry in entries]
            param_frame.entries = entries
        else:
            param_values = []

        effect_str = effect_name
        if param_values:
            effect_str += " " + " ".join([str(v) for _, v in param_values])

        self.effect_listbox.insert(tk.END, effect_str)
        self.effect_listbox.selection_clear(0, tk.END)
        self.effect_listbox.selection_set(tk.END)
        self.effect_listbox.see(tk.END)

    def browse_ir(self, entry):
        filepath = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if filepath:
            entry.delete(0, tk.END)
            entry.insert(0, filepath)

    def remove_effect(self):
        selection = self.effect_listbox.curselection()
        if not selection:
            return
        self.effect_listbox.delete(selection[0])
        self.update_param_frame()

        # Redraw waveform based on remaining chain state
        effects_remaining = self.effect_listbox.size() > 0
        output_path = self.output_path.get()
        output_exists = output_path and os.path.exists(output_path)

        if effects_remaining and output_exists:
            self.update_waveform(output_path)
        elif self._input_waveform_path and os.path.exists(self._input_waveform_path):
            self.update_waveform(self._input_waveform_path)
        else:
            self.preview_canvas.delete("all")

    def move_up(self):
        selection = self.effect_listbox.curselection()
        if selection and selection[0] > 0:
            index = selection[0]
            effect = self.effect_listbox.get(index)
            self.effect_listbox.delete(index)
            self.effect_listbox.insert(index - 1, effect)
            self.effect_listbox.selection_clear(0, tk.END)
            self.effect_listbox.selection_set(index - 1)
            self.update_param_frame()

    def move_down(self):
        selection = self.effect_listbox.curselection()
        if selection and selection[0] < self.effect_listbox.size() - 1:
            index = selection[0]
            effect = self.effect_listbox.get(index)
            self.effect_listbox.delete(index)
            self.effect_listbox.insert(index + 1, effect)
            self.effect_listbox.selection_clear(0, tk.END)
            self.effect_listbox.selection_set(index + 1)
            self.update_param_frame()

    def update_param_frame(self):
        for widget in self.param_frame.winfo_children():
            widget.destroy()

        selection = self.effect_listbox.curselection()
        if not selection:
            return

        effect_str = self.effect_listbox.get(selection[0])
        parts = effect_str.split()
        effect_name = parts[0]
        params = self.effects[effect_name]["params"]

        # Header: show which effect is selected and its index
        idx = selection[0]
        header = ttk.Label(
            self.param_frame,
            text=f"Parameters — [{idx + 1}] {effect_name}",
            font=("", 9, "bold")
        )
        header.pack(anchor=tk.W, pady=(2, 4))

        if not params:
            ttk.Label(self.param_frame, text="(no parameters)", foreground="grey").pack(anchor=tk.W)
            return

        param_frame = ttk.Frame(self.param_frame)
        param_frame.pack(fill=tk.X, pady=2)

        entries = []
        for i, param in enumerate(params):
            if len(param) == 1:
                param_name = param[0]
                min_val = max_val = default = None
            else:
                param_name, min_val, max_val, default = param

            frame = ttk.Frame(param_frame)
            frame.pack(fill=tk.X, pady=2)

            label = ttk.Label(frame, text=f"{param_name}:", width=15)
            label.pack(side=tk.LEFT)

            if param_name == "ir_file":
                entry = ttk.Entry(frame, width=40)
                entry.pack(side=tk.LEFT, expand=True, fill=tk.X)
                button = ttk.Button(frame, text="Browse...", command=lambda e=entry: self.browse_ir(e))
                button.pack(side=tk.LEFT, padx=5)
            else:
                entry = ttk.Spinbox(frame, from_=min_val, to=max_val, increment=(max_val-min_val)/100, width=10)
                entry.set(default)
                entry.pack(side=tk.LEFT)

            entries.append((param_name, entry))

        if len(parts) > 1:
            param_values = parts[1:]
            for (param_name, entry), value in zip(entries, param_values):
                try:
                    entry.delete(0, tk.END)
                    entry.insert(0, value)
                except Exception:
                    pass

        param_frame.entries = entries

        # Apply button writes current spinbox values back to the listbox entry
        def apply_params(eff_name=effect_name, ent_list=entries, sel_idx=idx):
            new_str = eff_name
            vals = [e.get() for _, e in ent_list]
            if vals:
                new_str += " " + " ".join(vals)
            self.effect_listbox.delete(sel_idx)
            self.effect_listbox.insert(sel_idx, new_str)
            self.effect_listbox.selection_set(sel_idx)

        ttk.Button(self.param_frame, text="Apply Params", command=apply_params).pack(
            anchor=tk.W, pady=(4, 0)
        )

    def get_effect_chain(self):
        return [self.effect_listbox.get(i) for i in range(self.effect_listbox.size())]

    def process(self):
        input_path = self.input_path.get()
        output_path = self.output_path.get()

        if not input_path or not output_path:
            messagebox.showerror("Error", "Please select input and output files")
            return

        if not os.path.exists(input_path):
            messagebox.showerror("Error", "Input file does not exist")
            return

        effects = self.get_effect_chain()
        if not effects:
            messagebox.showerror("Error", "No effects in chain")
            return

        self.log_text.insert(tk.END, "Processing...\n")
        self.log_text.see(tk.END)
        self.root.update()

        threading.Thread(
            target=self.run_daemon_job,
            args=(input_path, output_path, effects),
            daemon=True
        ).start()

    def run_daemon_job(self, input_path, output_path, effects):
        try:
            with self.lock:
                if not self.daemon or self.daemon.poll() is not None:
                    self.start_daemon()

                if self.daemon is None:
                    raise RuntimeError("clfx daemon could not be started")

                args = [input_path, output_path]
                for effect in effects:
                    args.extend(effect.split())

                lines = [str(len(args))] + args
                payload = "\n".join(lines) + "\n"
                self.daemon.stdin.write(payload)
                self.daemon.stdin.flush()

                response = self.daemon.stdout.readline().strip()

            if response == "OK":
                self._last_processed_path = output_path  # track what was actually written
                self.root.after(0, lambda: self.log_text.insert(tk.END, "Processing completed successfully!\n"))
                self.root.after(0, lambda: self.log_text.see(tk.END))
                self.root.after(0, self.update_waveform, output_path)
            else:
                self.root.after(0, lambda r=response: self.log_text.insert(tk.END, f"Engine reported: {r}\n"))
                self.root.after(0, lambda: self.log_text.see(tk.END))

        except Exception as e:
            self.root.after(0, lambda err=str(e): self.log_text.insert(tk.END, f"Error: {err}\n"))
            self.root.after(0, lambda: self.log_text.see(tk.END))

    def start_daemon(self):
        try:
            if self.daemon and self.daemon.poll() is None:
                self.daemon.terminate()
                try:
                    self.daemon.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.daemon.kill()

            script_dir = os.path.dirname(os.path.abspath(__file__))
            exe_name = "clfx.exe" if os.name == "nt" else "clfx"
            exe_path = os.path.join(script_dir, exe_name)
            if not os.path.exists(exe_path):
                exe_path = exe_name

            self.daemon = subprocess.Popen(
                [exe_path, "--daemon"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            self.root.after(0, lambda: self.log_text.insert(tk.END, "CLFX daemon started.\n"))
            self.root.after(0, lambda: self.log_text.see(tk.END))
        except FileNotFoundError:
            self.root.after(0, lambda: self.log_text.insert(
                tk.END, "Error: clfx executable not found. Place clfx.exe next to this script.\n"))
            self.root.after(0, lambda: self.log_text.see(tk.END))
        except Exception as e:
            self.root.after(0, lambda err=str(e): self.log_text.insert(
                tk.END, f"Error starting daemon: {err}\n"))
            self.root.after(0, lambda: self.log_text.see(tk.END))

    def update_waveform(self, filepath):
        try:
            with wave.open(filepath, 'rb') as wav:
                frames = wav.getnframes()
                channels = wav.getnchannels()
                sampwidth = wav.getsampwidth()
                frames_data = wav.readframes(frames)

            if sampwidth == 2:
                fmt = f"<{frames * channels}h"
                max_val = 32768
            elif sampwidth == 4:
                fmt = f"<{frames * channels}i"
                max_val = 2147483648
            else:
                self.log_text.insert(tk.END, "Unsupported sample width for waveform preview.\n")
                return

            samples = struct.unpack(fmt, frames_data)

            canvas_width = self.preview_canvas.winfo_width()
            canvas_height = self.preview_canvas.winfo_height()
            if canvas_width < 10:
                canvas_width = 800
            if canvas_height < 10:
                canvas_height = 120
            center = canvas_height // 2

            self.preview_canvas.delete("all")
            self.preview_canvas.create_line(0, center, canvas_width, center, fill="#cccccc")

            total_samples = frames * channels
            samples_per_pixel = max(1, total_samples // (canvas_width * channels))

            for px in range(canvas_width):
                start = px * samples_per_pixel * channels
                end = start + samples_per_pixel * channels
                segment = samples[start:end:channels]
                if not segment:
                    break
                peak = max(abs(s) for s in segment)
                bar_half = int((peak / max_val) * (center - 4))
                self.preview_canvas.create_line(
                    px, center - bar_half, px, center + bar_half,
                    fill="#1a7abf"
                )

        except Exception as e:
            self.log_text.insert(tk.END, f"Waveform preview error: {str(e)}\n")
            self.log_text.see(tk.END)


if __name__ == "__main__":
    root = tk.Tk()
    app = CLFXGUI(root)
    root.mainloop()
