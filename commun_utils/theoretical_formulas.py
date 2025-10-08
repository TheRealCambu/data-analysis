from typing import Union

import numpy as np
from scipy.special import erfc


def theoretical_ber_vs_snr(snr: np.ndarray, M: int) -> np.ndarray:
    """ snr in linear units """
    return 2 / np.log2(M) * (1 - 1 / np.sqrt(M)) * erfc(np.sqrt(3 * snr / 2 / (M - 1)))


def osnr_to_snr(osnr_dB_vect: Union[np.ndarray, float], symbol_rate: float):
    rx_bw_Hz = symbol_rate / 2
    osnr_lin_vect = 10 ** (0.1 * osnr_dB_vect)
    snr_lin_vect = osnr_lin_vect * 12.5e9 / rx_bw_Hz / 2
    return snr_lin_vect


def get_k(bits_per_symbol: int) -> float:
    # Compute modulation-dependent normalization factor
    if bits_per_symbol == 2:  # QPSK
        return np.sqrt(1.0)
    elif bits_per_symbol == 4:  # 16-QAM
        return np.sqrt(9 / 5)
    else:
        raise ValueError("Format not supported. Choose between 2 (QPSK) and 4 (16QAM).")


def gamma_i_function(i: int, M: int) -> float:
    return 1 - i / np.sqrt(M)


def beta_i_function(i: int) -> float:
    return 2 * i - 1


def evm_one(osnr_value: float, k: float, M: int, bits_per_symbol: int) -> float:
    a1 = 1.0 / osnr_value
    a2 = np.sqrt(96.0 / (np.pi * (M - 1) * osnr_value))
    b = 12.0 / (M - 1)

    first_sum = 0.0
    second_sum = 0.0
    for i in np.arange(1, bits_per_symbol):
        beta_i = beta_i_function(i)
        gamma_i = gamma_i_function(i, M)
        first_sum += gamma_i * np.exp(-3 * beta_i ** 2 * osnr_value / (2 * (M - 1)))
        second_sum += gamma_i * beta_i * erfc(np.sqrt((3 * beta_i ** 2 * osnr_value) / (2 * (M - 1))))

    first_term = a1 - a2 * first_sum
    second_term = b * second_sum

    return (1.0 / k) * np.sqrt(first_term + second_term)


def theoretical_evm_vs_osnr(
        osnr: Union[int, float, np.ndarray],
        bits_per_symbol: int,
        M: int,
        input_type: str = "lin"
):
    # Get modulation-dependent normalization factor
    k = get_k(
        bits_per_symbol=bits_per_symbol
    )

    # Convert dB to linear if needed
    osnr = np.asarray(osnr, dtype=np.float32)
    if input_type == "dB":
        osnr = 10 ** (0.1 * osnr)
    elif input_type != "lin":
        raise ValueError("input_type must be 'lin' or 'dB'.")

    # Vectorize evaluation if input is array-like
    if osnr.ndim > 0:
        return np.array([evm_one(val, k, M, bits_per_symbol) for val in osnr], dtype=np.float32)
    else:
        return evm_one(osnr.item(), k, M, bits_per_symbol)


def theoretical_ber_from_evm(
        EVM_m: Union[int, float, np.ndarray],
        M: int
):
    L = np.sqrt(M)
    # Get modulation-dependent normalization factor
    k = get_k(
        bits_per_symbol=int(np.sqrt(M))
    )
    # Convert to numpy array
    EVM_m = np.asarray(EVM_m, dtype=np.float32)
    log2L = np.log2(L)
    # inner factor that multiplies numerator / denom_L
    inner_factor = 1.0 / ((k * EVM_m) ** 2 * np.log2(M))
    sqrt_arg = 3.0 * log2L / (L ** 2 - 1.0) * inner_factor
    z = np.sqrt(sqrt_arg)
    prefactor = (1.0 - 1.0 / L) / log2L
    ber = prefactor * erfc(z)
    return ber
