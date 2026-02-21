"""
Complex Ratio Audio Visualizer & Lissajous Renderer
Renders sonic hyperspace: magnitude= pitch ratio, phase= spatial swirl
Tests: 3:2, √5, φ+i, 19:31 chaos
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cmath

# Audio params (RISC-V DAC ready)
SAMPLE_RATE = 48000
DURATION = 5.0
T = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION))

def complex_ratio_wave(ratio_complex):
    """Generate stereo wave from complex ratio r = |r|e^(iθ)"""
    f1 = 220.0  # A3
    f_base = f1
    
    # Magnitude → frequency ratio
    freq_ratio = abs(ratio_complex)
    f2 = f_base * freq_ratio
    
    # Phase → spatial rotation
    phase_shift = cmath.phase(ratio_complex)
    
    # Real wave (left channel)
    left = np.sin(2 * np.pi * f_base * T)
    
    # Complex partner: magnitude-scaled + phase-rotated (right channel)
    right = (np.sin(2 * np.pi * f2 * T + phase_shift) + 
             np.cos(2 * np.pi * f2 * T + phase_shift)) / np.sqrt(2)
    
    return left, right

def lissajous_complex(ratio_complex, t_samples=10000):
    """3D Lissajous for complex ratios: x,y,z projection"""
    t = np.linspace(0, 10, t_samples)
    f1 = 1.0
    freq_ratio = abs(ratio_complex)
    phase = cmath.phase(ratio_complex)
    
    x = np.sin(2 * np.pi * f1 * t)
    y = np.sin(2 * np.pi * freq_ratio * t + phase)
    z = np.cos(2 * np.pi * freq_ratio * t + phase)  # Imaginary quadrature
    
    return x, y, z

# Test cases from conversation
test_ratios = {
    "3:2 Fifth": 3/2 + 0j,
    "√5 (~2.236)": np.sqrt(5) + 0j,
    "Golden φ+i": (1 + np.sqrt(5))/2 + 1j,
    "19:31 Chaos": 19/31 + 0j,
    "Pure Imag i": 0 + 1j,
    "Xenakis Swirl": 2.1 + 1.7j
}

# Generate & plot everything
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (name, r) in enumerate(test_ratios.items()):
    ax = axes[idx]
    
    # Audio waveform (stereo Lissajous)
    left, right = complex_ratio_wave(r)
    ax.plot(T[:1000], left[:1000], 'b-', label='Left (Real)', alpha=0.7)
    ax.plot(T[:1000], right[:1000], 'r--', label='Right (Complex)', alpha=0.7)
    ax.set_title(f'{name}\n|r|={abs(r):.3f}, θ={np.degrees(cmath.phase(r)):.0f}°')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('complex_ratios_audio.png', dpi=300)
plt.show()

# 3D Lissajous animations
fig2 = plt.figure(figsize=(12, 4))

for idx, (name, r) in enumerate([("√5 Chaos", np.sqrt(5)+0j), ("φ+i Helix", (1+np.sqrt(5))/2 + 1j)]):
    ax = fig2.add_subplot(1, 2, idx+1, projection='3d')
    
    def update_lissajous(frame):
        ax.clear()
        t_frame = np.linspace(0, frame/10, 5000)
        x, y, z = lissajous_complex(r, len(t_frame))
        ax.plot(x, y, z, 'c-', alpha=0.7, lw=0.5)
        ax.set_title(f'{name}: 3D Helix Projection')
        ax.set_xlim([-1.5, 1.5])
        ax.set_ylim([-1.5, 1.5])
        ax.set_zlim([-1.5, 1.5])
    
    ani = FuncAnimation(fig2, update_lissajous, frames=100, interval=50)
    ani.save(f'lissajous_{name.lower().replace(" ", "_")}.gif', writer='pillow')

print("Render complete. Play stereo waves—feel the phase swirl!")
print("FFT these: rational=peaks, irrational=smeared noise, complex=spatial motion.")
