import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

# Parameters
sr = 44100
duration = 2.0
base_freq = 440.0  # A4 for clearer visualization

def get_dyad(ratio, dur=duration, rate=sr):
    t = np.linspace(0, dur, int(rate * dur), False)
    sig1 = np.sin(2 * np.pi * base_freq * t)
    sig2 = np.sin(2 * np.pi * (base_freq * ratio) * t)
    return t, sig1, sig2

def analyze_ratio(name, ratio):
    print(f"Testing {name} (Ratio: {ratio:.4f})")
    t, s1, s2 = get_dyad(ratio)
    mixed = (s1 + s2) / 2
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. Lissajous Curve (s1 vs s2)
    # We only plot a small slice (first 1000 samples) to see the shape
    ax1.plot(s1[:1000], s2[:1000], lw=1, color='magma' if "Chaos" in name else 'teal')
    ax1.set_title(f"Lissajous: {name}")
    ax1.set_axis_off()
    
    # 2. FFT (Frequency Spectrum)
    fft_data = np.abs(np.fft.rfft(mixed))
    freqs = np.fft.rfftfreq(len(mixed), 1/sr)
    ax2.plot(freqs, fft_data)
    ax2.set_xlim(base_freq - 100, (base_freq * ratio) + 100)
    ax2.set_title("Frequency Spectrum")
    
    plt.tight_layout()
    plt.show()

    # Play
    sd.play(mixed, sr)
    sd.wait()

# --- The "Challenge" Menu ---

# 1. The Perfect Fifth (The Gold Standard of Pleasant)
analyze_ratio("Perfect Fifth", 3/2)

# 2. The Tritone (Augmented 4th - Historically "The Devil in Music")
# Ratio is sqrt(2), which is irrational.
analyze_ratio("Tritone", np.sqrt(2))

# 3. The Golden Ratio (The ultimate inharmonic sound)
# This ratio never closes its loop.
phi = (1 + 5**0.5) / 2
analyze_ratio("Golden Ratio (Phi)", phi)

# 4. Critical Band Roughness (Very close frequencies)
# This creates massive "beating"
analyze_ratio("Microtonal Friction", 1.05) 
