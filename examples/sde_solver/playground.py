import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# Sampling settings
fs = 500  # Sampling frequency in Hz (lower for focus on lower frequencies)
t = np.linspace(0, 2, 2 * fs, endpoint=False)  # 2 seconds of data

# Create two simple signals with higher power at lower frequencies
# Signal 1: A sine wave at 5 Hz and 10 Hz combined
signal1 = 3 * np.sin(2 * np.pi * 5 * t) + 2 * np.sin(2 * np.pi * 10 * t)

# Signal 2: A sine wave at 3 Hz and 7 Hz combined
signal2 = 4 * np.sin(2 * np.pi * 3 * t) + 1.5 * np.sin(2 * np.pi * 7 * t)

# Compute Power Spectral Density (PSD) using Welch's method
f1, Pxx1 = welch(signal1, fs, nperseg=512)
f2, Pxx2 = welch(signal2, fs, nperseg=512)

# Convert PSD to dB/Hz
Pxx1_dB = 10 * np.log10(Pxx1)
Pxx2_dB = 10 * np.log10(Pxx2)

# Plotting the signals and their PSDs
plt.figure(figsize=(12, 8))

# Plot time-domain signals
plt.subplot(2, 1, 1)
plt.plot(t, signal1, label=r'Signal 1: $3 \sin(2 \pi \cdot 5 \cdot t) + 2 \sin(2 \pi \cdot 10 \cdot t)$')
plt.plot(t, signal2, label=r'Signal 2: $4 \sin(2 \pi \cdot 3 \cdot t) + 1.5 \sin(2 \pi \cdot 7 \cdot t)$')
plt.title("Time-Domain Signals")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

# Plot PSDs in dB/Hz
plt.subplot(2, 1, 2)
plt.plot(f1, Pxx1_dB, label=r'Signal 1: $3 \sin(2 \pi \cdot 5 \cdot t) + 2 \sin(2 \pi \cdot 10 \cdot t)$')
plt.plot(f2, Pxx2_dB, label=r'Signal 2: $4 \sin(2 \pi \cdot 3 \cdot t) + 1.5 \sin(2 \pi \cdot 7 \cdot t)$')
plt.title("Power Spectral Density (PSD) in dB/Hz")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD (dB/Hz)")
plt.xlim(0,50)
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()