import numpy as np
import matplotlib.pyplot as plt
from commun_utils.utils import apply_plt_personal_settings

apply_plt_personal_settings()

# --- parameters ---
fs = 50e9  # sampling frequency (100 GHz)
tau = 10e-12  # true skew = 5 ps
N = 4096
t = np.arange(N) / fs

# test frequencies
freqs = np.linspace(1e9, 32e9, 30)  # 1–20 GHz

phases = []

for f in freqs:
    x = np.sin(2 * np.pi * f * t)
    y = np.sin(2 * np.pi * f * (t - tau))  # delayed version

    # compute phase difference using FFT bin
    X = np.fft.fft(x)
    Y = np.fft.fft(y)

    # frequency bin index
    k = int(np.round(f * N / fs))

    phase_diff = np.angle(Y[k] / X[k])
    phases.append(phase_diff)

phases = np.unwrap(phases)

# linear fit: phase(f) = 2π·tau·f
p = np.polyfit(freqs, phases, 1)
tau_est = p[0] / (2 * np.pi)

print(f"Estimated skew: {tau_est * 1e12:.3f} ps")

plt.figure(figsize=(8, 5))
plt.plot(freqs / 1e9, phases, 'o', label='Measured phase')
plt.plot(freqs / 1e9, np.polyval(p, freqs), '-', label='Linear fit')

plt.xlabel("Frequency (GHz)")
plt.ylabel("Phase shift (rad)")
plt.title("Phase vs Frequency — Slope = Time Skew")
plt.grid(True, which='both')
plt.legend(loc="best")

# --- Annotate estimated skew ---
plt.text(
    0.05, 0.95,
    f"Estimated skew = {tau_est * 1e12:.2f} ps",
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment='top',
    bbox=dict(facecolor='white', alpha=0.8, boxstyle='round')
)

plt.show()
