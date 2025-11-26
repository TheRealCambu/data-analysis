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

# Receiver electrical noise
electrical_noise_dBm_per_Hz = -170  # typical thermal noise floor
receiver_bandwidth_Hz = Rs * (1 + alpha)  # bandwidth of the filter

# -------------------------------------------------------------------
# Wavelength and frequency grid
# -------------------------------------------------------------------
lambdas = np.linspace(lambda_min, lambda_max, n_points)
freqs = c / lambdas
f_center = c / lambda_center
df = np.abs(freqs[1] - freqs[0])
freqs_shifted = freqs - f_center

# -------------------------------------------------------------------
# RRC spectrum in wavelength domain
# -------------------------------------------------------------------
delta_lambda = (lambda_center ** 2 / c) * Rs * (1 + alpha)
lambda_detuning = lambdas - lambda_center


def rrc_spectrum_wavelength(delta_lambda, Rs, alpha, lambda_0):
    delta_f_equiv = (c / lambda_0 ** 2) * np.abs(delta_lambda)
    H = np.zeros_like(delta_f_equiv)
    f1 = (1 - alpha) * Rs / 2
    f2 = (1 + alpha) * Rs / 2
    H[delta_f_equiv <= f1] = 1.0
    idx = (delta_f_equiv > f1) & (delta_f_equiv < f2)
    H[idx] = 0.5 * (1 + np.cos(np.pi / (alpha * Rs) * (delta_f_equiv[idx] - f1)))
    return H


signal = rrc_spectrum_wavelength(lambda_detuning, Rs, alpha, lambda_center)
signal /= np.trapezoid(signal, lambdas)
signal *= 1e-3  # 1 mW peak

# -------------------------------------------------------------------
# ASE noise + realistic floor
# -------------------------------------------------------------------
ase_total_power = 1e-6  # -30 dBm total ASE over 4 nm
ase_floor_power = 1e-14  # -140 dBm outside ASE band

ase = np.ones_like(lambdas) * ase_floor_power
ase_slice = ((lambdas > (lambda_center - bandwidth_nm / 2 * 1e-9)) &
             (lambdas < (lambda_center + bandwidth_nm / 2 * 1e-9)))

# normalize ASE inside the band
ase_inside = np.ones_like(lambdas[ase_slice])
ase_inside /= np.trapezoid(ase_inside, lambdas[ase_slice])
ase_inside *= ase_total_power
ase[ase_slice] = ase_inside

# -------------------------------------------------------------------
# Add receiver electrical noise
# -------------------------------------------------------------------
# convert dBm/Hz to linear Watts
noise_power_W = 10 ** ((electrical_noise_dBm_per_Hz - 30) / 10) * receiver_bandwidth_Hz
electrical_noise = np.ones_like(lambdas) * noise_power_W

# -------------------------------------------------------------------
# Total spectrum
# -------------------------------------------------------------------
spectrum_total = signal + ase + electrical_noise

# Normalize so that max = 0 dBm (for visualization)
scaling_factor = 1e-3 / np.max(spectrum_total)
signal *= scaling_factor
ase *= scaling_factor
electrical_noise *= scaling_factor
spectrum_total = signal + ase + electrical_noise

# -------------------------------------------------------------------
# OSNR calculation (0.1 nm reference)
# -------------------------------------------------------------------
ase_band = ((lambdas > (lambda_center - delta_lambda / 2)) &
            (lambdas < (lambda_center + delta_lambda / 2)))
p_ase_0_1nm = np.trapezoid(ase[ase_band], lambdas[ase_band])
p_signal = np.trapezoid(signal, lambdas)
OSNR_dB = 10 * np.log10(p_signal / p_ase_0_1nm)
print(f"Simulated OSNR ≈ {OSNR_dB:.2f} dB")

# -------------------------------------------------------------------
# Plot
# -------------------------------------------------------------------
plt.figure()
plt.plot(lambdas * 1e9, 10 * np.log10(signal / 1e-3),
         label='RRC Signal', color='blue', lw=2)
plt.plot(lambdas * 1e9, 10 * np.log10(ase / 1e-3),
         label='ASE noise', color='red', lw=2, ls='--')
plt.plot(lambdas * 1e9, 10 * np.log10(electrical_noise / 1e-3),
         label='Electrical noise', color='green', lw=2, ls=':')
plt.plot(lambdas * 1e9, 10 * np.log10(spectrum_total / 1e-3),
         label='Total Spectrum', color='black', lw=2, ls='-.')
plt.axvspan((lambda_center - bandwidth_nm / 2 * 1e-9) * 1e9,
            (lambda_center + bandwidth_nm / 2 * 1e-9) * 1e9,
            color='gray', alpha=0.2, label='WaveShaper 4nm band')
plt.ylim(-170, 3)
plt.xlim(1547.5, 1552.5)
plt.xlabel("Wavelength [nm]")
plt.ylabel("Power Spectral Density [dBm]")
plt.title(f"Transmitted Spectrum — OSNR ≈ {OSNR_dB:.1f} dB")
plt.grid(True, which='both')
plt.legend(loc="best")
plt.tight_layout()
plt.show()
