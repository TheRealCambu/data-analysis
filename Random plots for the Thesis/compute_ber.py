import numpy as np
from commun_utils.theoretical_formulas import theoretical_ber_vs_snr, osnr_to_snr

osnr = 18
M = 16
symbol_rate = 32e9

snr_lin_vect = osnr_to_snr(
    OSNR_vect=osnr,
    symbol_rate=symbol_rate
)

ber = theoretical_ber_vs_snr(snr_lin_vect, M)

print(ber)
