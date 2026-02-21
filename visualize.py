import wave
import struct
import math
import sys
import os

def visualize_ascii(filename, width=60):
    try:
        f = wave.open(filename, 'rb')
    except Exception as e:
        print(f"Error opening file: {e}")
        return

    channels = f.getnchannels()
    frames = f.getnframes()
    
    # Read all frames
    raw_data = f.readframes(frames)
    f.close()
    
    # Unpack 16-bit PCM
    fmt = f"<{frames * channels}h"
    samples = struct.unpack(fmt, raw_data)
    
    # Normalize to float and get mono peak if stereo
    float_samples = []
    for i in range(0, len(samples), channels):
        s = abs(samples[i] / 32768.0)
        if channels > 1:
            s_r = abs(samples[i+1] / 32768.0)
            s = max(s, s_r)
        float_samples.append(s)

    print(f"\nWaveform: {filename} ({channels} channels, {frames} frames)")
    
    samples_per_col = len(float_samples) // width
    if samples_per_col < 1: samples_per_col = 1
    
    for i in range(width):
        start = i * samples_per_col
        end = start + samples_per_col
        chunk = float_samples[start:end]
        if not chunk: break
        peak = max(chunk)
        bar = "#" * int(peak * 50)
        print(f"{int(peak*100):3}% | {bar}")

def plot_waveform(filename):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Matplotlib/Numpy not found. Falling back to ASCII visualization.")
        visualize_ascii(filename)
        return

    try:
        f = wave.open(filename, 'rb')
        channels = f.getnchannels()
        fs = f.getframerate()
        frames = f.getnframes()
        data = f.readframes(frames)
        f.close()

        samples = np.frombuffer(data, dtype=np.int16)
        if channels == 2:
            left = samples[0::2]
            right = samples[1::2]
        else:
            left = samples
            right = samples

        time = np.linspace(0, frames / fs, num=frames)

        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.plot(time, left, color='blue')
        plt.title(f"Waveform: {filename}")
        plt.ylabel("Amplitude (Left)")
        plt.grid(True)

        if channels == 2:
            plt.subplot(2, 1, 2)
            plt.plot(time, right, color='red')
            plt.ylabel("Amplitude (Right)")
            plt.grid(True)
        
        plt.xlabel("Time (s)")
        plt.tight_layout()
        
        out_img = "waveform.png"
        plt.savefig(out_img)
        print(f"High-resolution waveform saved to {out_img}")
        
    except Exception as e:
        print(f"Plotting failed: {e}")
        visualize_ascii(filename)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize.py <audio.wav>")
    else:
        plot_waveform(sys.argv[1])
