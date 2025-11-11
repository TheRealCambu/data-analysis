import numpy as np
import matplotlib.pyplot as plt
from commun_utils.utils import apply_plt_personal_settings

apply_plt_personal_settings()

# -------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------
lambda_center = 1550e-9  # center wavelength [m]
bandwidth_nm = 4  # ASE bandwidth [nm]
lambda_min = 1540e-9
lambda_max = 1560e-9
n_points = 5001

# RRC filter parameters
Rs = 32e9  # symbol rate [baud]
alpha = 0.2  # roll-off factor
c = 3e8  # speed of light [m/s]

# -------------------------------------------------------------------
# Wavelength grid and frequency grid
# -------------------------------------------------------------------
lambdas = np.linspace(lambda_min, lambda_max, n_points)
freqs = c / lambdas
f_center = c / lambda_center
f_span = np.max(freqs) - np.min(freqs)
df = np.abs(freqs[1] - freqs[0])

# frequency axis centered at f_center
freqs_shifted = freqs - f_center

# -------------------------------------------------------------------
# 1. Root Raised Cosine spectral shape - work directly in wavelength
# -------------------------------------------------------------------
# Convert symbol rate to wavelength bandwidth
delta_f = Rs * (1 + alpha)  # total bandwidth in Hz
delta_lambda = (lambda_center ** 2 / c) * delta_f  # corresponding wavelength span

# Use wavelength detuning
lambda_detuning = lambdas - lambda_center


def rrc_spectrum_wavelength(delta_lambda, Rs, alpha, lambda_0):
    """RRC shape directly in wavelength domain."""
    # Convert to equivalent frequency detuning
    delta_f_equiv = (c / lambda_0 ** 2) * np.abs(delta_lambda)

    H = np.zeros_like(delta_f_equiv)
    f1 = (1 - alpha) * Rs / 2
    f2 = (1 + alpha) * Rs / 2

    H[delta_f_equiv <= f1] = 1.0
    idx = (delta_f_equiv > f1) & (delta_f_equiv < f2)
    H[idx] = 0.5 * (1 + np.cos(np.pi / (alpha * Rs) * (delta_f_equiv[idx] - f1)))
    return H


signal = rrc_spectrum_wavelength(lambda_detuning, Rs, alpha, lambda_center)

# Normalize to 1 mW
signal /= np.trapezoid(signal, lambdas)
signal *= 1e-3

# -------------------------------------------------------------------
# 2. ASE noise (flat within 4 nm)
# -------------------------------------------------------------------
ase_total_power = 1e-6  # -30 dBm total ASE over 4 nm
ase = np.ones_like(lambdas)
ase_slice = (lambdas > (lambda_center - bandwidth_nm / 2 * 1e-9)) & \
            (lambdas < (lambda_center + bandwidth_nm / 2 * 1e-9))
ase *= ase_slice.astype(float)
ase /= np.trapezoid(ase, lambdas)
ase *= ase_total_power

# -------------------------------------------------------------------
# 3. Combined spectrum
# -------------------------------------------------------------------
spectrum_total = signal + ase

# Normalize so that max = 0 dBm (visual)
scaling_factor = 1e-3 / np.max(spectrum_total)
signal *= scaling_factor
ase *= scaling_factor
spectrum_total = signal + ase

# -------------------------------------------------------------------
# 4. Compute OSNR (0.1 nm reference)
# -------------------------------------------------------------------
ase_band = (lambdas > (lambda_center - delta_lambda / 2)) & \
           (lambdas < (lambda_center + delta_lambda / 2))
p_ase_0_1nm = np.trapezoid(ase[ase_band], lambdas[ase_band])
p_signal = np.trapezoid(signal, lambdas)
OSNR_dB = 10 * np.log10(p_signal / p_ase_0_1nm)
print(f"Simulated OSNR ≈ {OSNR_dB:.2f} dB")

# -------------------------------------------------------------------
# 5. Plot
# -------------------------------------------------------------------
plt.figure()
plt.plot(lambdas * 1e9, 10 * np.log10(signal / 1e-3), label='RRC Signal')
plt.plot(lambdas * 1e9, 10 * np.log10(ase / 1e-3), label='ASE noise')
plt.plot(lambdas * 1e9, 10 * np.log10(spectrum_total / 1e-3), label='Signal + ASE')
plt.axvspan((lambda_center - 2e-9) * 1e9, (lambda_center + 2e-9) * 1e9,
            color='gray', alpha=0.2, label='4 nm ASE slice')
plt.ylim(-50, 5)
plt.xlim(1546.5, 1553.5)
plt.xlabel("Wavelength [nm]")
plt.ylabel("Power Spectral Density [dBm]")
plt.title(f"RRC-shaped Signal + EDFA ASE Noise — OSNR ≈ {OSNR_dB:.1f} dB")
plt.legend()
plt.grid(True, which="both")
plt.tight_layout()
plt.show()
