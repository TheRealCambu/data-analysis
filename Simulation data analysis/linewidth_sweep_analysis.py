import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

from commun_utils.theoretical_formulas import (
    theoretical_ber_vs_snr,
    theoretical_evm_vs_osnr,
    theoretical_evm_from_ber,
    osnr_to_snr
)
from commun_utils.utils import apply_plt_personal_settings

mod_format_vect = ["QPSK", "16QAM"]
pll_active_vect = ["True", "False"]
bps_bypassed_vect = ["True", "False"]
linewidth_vect = [
    "100", "200", "300", "400", "500",
    "600", "700", "800", "900", "1000"
]

folder_path = (r"C:\Users\simon\Politecnico Di Torino Studenti Dropbox\Simone Cambursano\Politecnico"
               r"\Tesi\Data-analysis\Simulation Sweeps\Linewidth Sweeps\First Batch")

# folder_path = (r"C:\Users\39338\Politecnico Di Torino Studenti Dropbox\Simone Cambursano\Politecnico"
#                r"\Tesi\Data-analysis\Simulation Sweeps\Linewidth Sweeps\First Batch"
#                r"\results_16QAM_32.0GBaud_CRAlgo_gardner_PLLActive_False_BPSBypassed_False_Linewidth_200.00kHz.npz")

# Apply personal matplotlib settings
apply_plt_personal_settings()

ber_fec_threshold = 2e-2
# ber_filter_threshold = 0.26
ber_filter_threshold = 0.1732
# ber_filter_threshold = 0.254

y_values_columns = {
    "berTot": {"title": "BER", "ylabel": "BER"},
    "EVMTot": {"title": "EVM", "ylabel": "EVM [%]"},
    # "berEVMTot": {"title": "BER$_{\mathrm{EVM}}$", "ylabel": r"BER$_{\mathrm{EVM}}$"}
}

mod_format_dict = {
    2: {
        "mod_format_string": "DP-QPSK",
        "cardinality": 4
    },
    4: {
        "mod_format_string": "DP-16QAM",
        "cardinality": 16
    }
}

for plot_type, plot_type_dict in y_values_columns.items():
    for mod_format in mod_format_vect:
        for pll_active, bps_bypassed in zip(pll_active_vect, bps_bypassed_vect):
            if pll_active == "True" and bps_bypassed == "True" and mod_format == "QPSK":
                # Skip because there is no carrier recovery in this case so BER = 0.5
                continue
            else:
                load_npz_vect = []
                for linewidth in linewidth_vect:
                    current_npz_file = (f"results_{mod_format}_32.0GBaud_CRAlgo_gardner_PLLActive_{pll_active}_"
                                        f"BPSBypassed_{bps_bypassed}_Linewidth_{linewidth}.00kHz.npz")
                    current_path = os.path.join(folder_path, current_npz_file)
                    load_npz_vect.append(dict(np.load(current_path, allow_pickle=True)))
                linewidth_x = np.unique(
                    [x["cfg_channel"].item()["laser_phase_noise"]["delta_nu"] for x in load_npz_vect]
                )
                y_values = np.array([x[plot_type].flatten() for x in load_npz_vect])
                temp_linewidth_x = linewidth_x.copy()[:, np.newaxis]
                symbol_rate = np.unique([x["cfg_common"].item()["symbol_rate"] for x in load_npz_vect])
                bits_per_symbol = np.unique([x["cfg_common"].item()["bits_per_symbol"] for x in load_npz_vect])
                OSNR_dB_vect = np.unique([x["OSNR_dB_vect"] for x in load_npz_vect])
                temp_OSNR_dB_vect = OSNR_dB_vect.copy()[:, np.newaxis]
                assert len(symbol_rate) == 1
                assert len(bits_per_symbol) == 1
                assert temp_OSNR_dB_vect.shape[1] == 1
                assert temp_linewidth_x.shape[1] == 1
                symbol_rate = symbol_rate.item()
                bits_per_symbol = int(bits_per_symbol.item())
                # Apply policy of filtering and min/max/mean
                
# In the title put:
# 1) BER/EVM vs Linewidth
# 2) OSNR
# 3) Symbol rate
