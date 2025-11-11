import numpy as np
from scipy.special import erfc
import matplotlib.pyplot as plt


def theoretical_ber_vs_snr(snr: np.ndarray, M: int, diff_encoding: bool = False) -> np.ndarray:
    """ snr in linear units """
    ber = 2 / np.log2(M) * (1 - 1 / np.sqrt(M)) * erfc(np.sqrt(3 * snr / 2 / (M - 1)))
    return (2 if diff_encoding else 1) * ber



# def theoretical_ber_vs_snr(snr_linear: np.ndarray, M: int, diff_encoding: bool = False) -> np.ndarray:
#     """Compute theoretical BER vs SNR (linear units).
#        - For QPSK (M=4), uses exact formulas for coherent and differential detection.
#     """
#     if M != 4:
#         # Keep your original approximate formula for M-QAM in general
#         ber = 2 / np.log2(M) * (1 - 1 / np.sqrt(M)) * erfc(np.sqrt(3 * snr_linear / 2 / (M - 1)))
#         return ber
#
#     if diff_encoding:
#         # Differentially detected QPSK (DQPSK)
#         ber = 0.5 * np.exp(-snr_linear)
#     else:
#         # Coherently detected QPSK (same as BPSK)
#         ber = 0.5 * erfc(np.sqrt(snr_linear))
#     return ber


# Example plotting
snr_db = np.linspace(8, 15, 100)
snr_linear = 10 ** (snr_db / 10)
M = 16

ber_coh = theoretical_ber_vs_snr(snr_linear, M, diff_encoding=False)
ber_diff = theoretical_ber_vs_snr(snr_linear, M, diff_encoding=True)

plt.figure(figsize=(8, 5))
plt.plot(snr_db, 10 * np.log10(ber_coh), label="Coherent QPSK")
plt.plot(snr_db, 10 * np.log10(ber_diff), label="Differential QPSK (DQPSK)", linestyle='--')
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.grid(True, which='both', linestyle='--', linewidth=0.7)
plt.legend()
plt.title("Theoretical BER for Coherent vs Differential QPSK")
plt.show()
