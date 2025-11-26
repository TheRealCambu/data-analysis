import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from commun_utils.theoretical_formulas import (
    theoretical_ber_vs_snr,
    theoretical_evm_vs_osnr,
    theoretical_evm_from_ber,
    osnr_to_snr
)
from commun_utils.utils import apply_plt_personal_settings

mod_format_vect = ["QPSK", "16QAM"]
baud_rate_vect = ["30.0", "32.0", "34.0", "36.0", "38.0", "40.0"]
# baud_rate_vect = ["32.0", "34.0", "36.0", "38.0", "40.0"]
# mod_format_vect = ["QPSK"]
# baud_rate_vect = ["38.0", "40.0"]

folder_path_first = (r"C:\Users\39338\Politecnico Di Torino Studenti Dropbox\Simone Cambursano"
                     r"\Politecnico\Tesi\Data-analysis\Simulation Sweeps\Rate Sweeps\First Batch")

folder_path_second = (r"C:\Users\39338\Politecnico Di Torino Studenti Dropbox\Simone Cambursano"
                      r"\Politecnico\Tesi\Data-analysis\Simulation Sweeps\Rate Sweeps\Second Batch")

# mod_format_vect = ["QPSK"]
# baud_rate_vect = ["30.0"]
cr_algo_vect = ["gardner", "fd", "godard"]
# cr_algo_vect = ["gardner", "fd"]

# Apply personal matplotlib settings
apply_plt_personal_settings()

markers = ['o', 's', 'v', '^', 'd', 'x']
# linestyles = ['-', '-', '-']
# linestyles = ['-', '--', '-.']
linestyles = [(5, (10, 3)), (0, (5, 5)), (0, (1, 1))]
colors = plt.cm.tab10(np.linspace(0, 1.0, len(cr_algo_vect)))

ber_fec_threshold = 2e-2
ber_filter_threshold = 0.427
# ber_filter_threshold = 0.48

# ber_filter_threshold = 0.25
# ber_filter_threshold = 0.269

y_values_columns = {
    "berTot": {"title": "BER", "ylabel": "BER"},
    # "EVMTot": {"title": "EVM", "ylabel": "EVM [%]"},
    # "berEVMTot": {"title": "BER$_{\mathrm{EVM}}$", "ylabel": r"BER$_{\mathrm{EVM}}$"}
}
cr_algo_dict = {
    "gardner": "Gardner",
    "fd": "Fast square-timing",
    "godard": "Godard"
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
        for baud_rate in baud_rate_vect:
            load_npz_vect_first = []
            load_npz_vect_second = []
            for cr_algo in cr_algo_vect:
                ######## First Batch ########
                current_npz_file_first = f"results_{mod_format}_{baud_rate}GBaud_2Polarizations_ClockRecAlgo_{cr_algo}.npz"
                current_path_first = os.path.join(folder_path_first, current_npz_file_first)
                load_npz_vect_first.append(dict(np.load(current_path_first, allow_pickle=True)))
                ######## Second Batch ########
                current_npz_file_second = f"results_{mod_format}_{baud_rate}GBaud_CRAlgo_{cr_algo}.npz"
                current_path_second = os.path.join(folder_path_second, current_npz_file_second)
                load_npz_vect_second.append(dict(np.load(current_path_second, allow_pickle=True)))
            symbol_rate = np.unique([x["cfg_common"].item()["symbol_rate"] for x in load_npz_vect_second])
            bits_per_symbol = np.unique([x["cfg_common"].item()["bits_per_symbol"] for x in load_npz_vect_second])
            OSNR_dB_vect = np.unique([x["OSNR_dB_vect"] for x in load_npz_vect_second])
            temp_OSNR_dB_vect = OSNR_dB_vect.copy()[:, np.newaxis]
            assert len(symbol_rate) == 1
            assert len(bits_per_symbol) == 1
            assert temp_OSNR_dB_vect.shape[1] == 1
            symbol_rate = symbol_rate.item()
            bits_per_symbol = int(bits_per_symbol.item())
            cardinality = mod_format_dict[bits_per_symbol]["cardinality"]
            modulation_format = mod_format_dict[bits_per_symbol]["mod_format_string"]
            OSNR_dB_vect_dense = np.linspace(OSNR_dB_vect[0] - 1, OSNR_dB_vect[-1] + 1, 10000)
            snr_lin_vect_dense = osnr_to_snr(OSNR_vect=OSNR_dB_vect_dense, symbol_rate=symbol_rate)
            is_ber = 'ber' in plot_type
            if is_ber:
                filter_threshold = ber_filter_threshold
                theory_curve = theoretical_ber_vs_snr(
                    snr=snr_lin_vect_dense,
                    M=cardinality
                )
                scale = 1
            else:
                filter_threshold = theoretical_evm_from_ber(ber=ber_filter_threshold, M=cardinality)
                print(filter_threshold)
                if filter_threshold < 0:
                    filter_threshold = 150 / 100
                theory_curve = theoretical_evm_vs_osnr(
                    osnr=snr_lin_vect_dense,
                    bits_per_symbol=bits_per_symbol,
                    M=cardinality
                )
                scale = 100
            plt.figure()
            plt.xlim(left=np.min(OSNR_dB_vect) - 0.3, right=np.max(OSNR_dB_vect) - 0.9)
            if is_ber:
                fec_threshold = ber_fec_threshold
                fec_threshold_label = f"FEC threshold={fec_threshold:.0e}"
            else:
                fec_threshold = theoretical_evm_from_ber(ber=ber_fec_threshold, M=cardinality) * 100
                fec_threshold_label = f"FEC threshold={fec_threshold:.2f}%"
            plt.axhline(fec_threshold,
                        color='black',
                        # color='darkred', linestyle=':',
                        linewidth=2, label=fec_threshold_label)

            for idx, (data_dict_first, data_dict_second) in enumerate(zip(load_npz_vect_first, load_npz_vect_second)):
                marker = markers[idx % len(markers)]
                color = colors[idx]
                linestyle = linestyles[idx]
                data_dict_clock_recovery = data_dict_second['cfg_rx'].item()['clock_recovery']
                current_cr_algo = data_dict_clock_recovery['algorithm']
                cr_algo_label = cr_algo_dict[current_cr_algo]
                sps_cr = data_dict_clock_recovery[current_cr_algo]['sps_cr']
                all_osnr_valid = True
                filtered_curves = []
                for i, (y_first, y_second) in enumerate(zip(data_dict_first[plot_type], data_dict_second[plot_type])):
                    y_arr = np.array(np.concatenate((y_first, y_second)), dtype=np.float32)

                    if OSNR_dB_vect[i] == 12.00 and mod_format == "QPSK" and baud_rate == "34.0":
                        y_filtered = np.where(y_arr > 0.2697655, np.nan, y_arr)
                    else:
                        y_filtered = np.where(y_arr > filter_threshold, np.nan, y_arr)

                    if np.isnan(y_filtered).all():
                        print(f"⚠ OSNR={OSNR_dB_vect[i]:.2f} dB → all BER invalid")
                        filtered_curves.append(np.nan)  # << keep curve, but mark missing point
                    else:
                        filtered_curves.append(np.nanmean(y_filtered))
                        # filtered_curves.append(np.nanmin(y_filtered))

                # Convert to array
                current_curve = np.array(filtered_curves, dtype=float)
                # If entire curve is NaN -> skip plotting for this algorithm
                if np.isnan(current_curve).all():
                    print(f"❌ All BER/EVM values NaN for {cr_algo_label} → skipping plot for this CR algorithm.")
                    continue

                # if all_osnr_valid:
                #     # --- Repair small gaps (only 1 or 2 NaN values) -----------------------
                #     nan_mask = np.isnan(current_curve)
                #     valid_mask = ~nan_mask
                #     osnr_valid = OSNR_dB_vect[valid_mask]
                #     ber_valid = current_curve[valid_mask]
                #
                #     # Only interpolate if we have >= 2 valid points
                #     if len(ber_valid) >= 2:
                #         # Linear interpolation over OSNR
                #         f_interp = interp1d(osnr_valid, ber_valid, bounds_error=False, fill_value="extrapolate")
                #         # Replace only NaN values
                #         current_curve[nan_mask] = f_interp(OSNR_dB_vect[nan_mask])

                if all_osnr_valid:
                    # --- Repair small gaps (only 1 or 2 NaN values) -----------------------
                    nan_mask = np.isnan(current_curve)
                    valid_mask = ~nan_mask
                    osnr_valid = OSNR_dB_vect[valid_mask]
                    ber_valid = current_curve[valid_mask]
                    # Interpolate in log10(ber) domain
                    log_ber_valid = np.log10(ber_valid)
                    f_interp = interp1d(
                        osnr_valid,
                        log_ber_valid,
                        bounds_error=False,
                        fill_value="extrapolate"
                    )
                    # Interpolate missing points (still in log domain)
                    interp_log_ber = f_interp(OSNR_dB_vect[nan_mask])
                    # Convert back to linear BER
                    current_curve[nan_mask] = 10 ** interp_log_ber

                    # If all OSNR points valid → plot
                    if is_ber:
                        # Interpolate measured BER curve (in log scale for better numerical stability)
                        ber_interp = interp1d(current_curve, OSNR_dB_vect, bounds_error=False)
                        theory_interp = interp1d(theory_curve, OSNR_dB_vect_dense, bounds_error=False)
                        # Compute OSNR values at the FEC threshold
                        osnr_measured_at_fec = ber_interp(ber_fec_threshold)
                        osnr_theoretical_at_fec = theory_interp(ber_fec_threshold)
                        if not np.isnan(osnr_measured_at_fec) and not np.isnan(osnr_theoretical_at_fec):
                            osnr_penalty = osnr_measured_at_fec - osnr_theoretical_at_fec
                            label_with_penalty = f"{cr_algo_label} | Penalty@FEC={osnr_penalty:.2f} dB"
                        else:
                            label_with_penalty = f"{cr_algo_label}"
                    else:
                        label_with_penalty = f"{cr_algo_label}"
                    plt.plot(
                        OSNR_dB_vect, current_curve * scale, marker,
                        linewidth=1.5,
                        # markersize=5.5,
                        # linestyle=linestyle,
                        linestyle='-',
                        color=color, label=label_with_penalty
                    )
                else:
                    print(f"❌ Skipped plot for {cr_algo_label} - SpS={sps_cr} - "
                          f"DSP chain failed to converge for at least one OSNR value.")
            plt.plot(
                OSNR_dB_vect_dense, theory_curve * scale, color="green",
                linewidth=2, label=f"Theoretical {plot_type_dict['title']}"
            )

            # Labels and title
            plt.yscale("log" if is_ber else "linear")
            plt.xlabel("OSNR [dB] per 0.1nm")
            plt.ylabel(plot_type_dict["ylabel"])
            plt.title(f"{plot_type_dict['title']} vs OSNR ({modulation_format}@{symbol_rate / 1e9:.0f}GBaud)")
            plt.title(f"{plot_type_dict['title']} vs OSNR ({modulation_format}@{symbol_rate / 1e9:.0f}GBaud)")
            plt.legend(loc="best")
            plt.grid(True, which="both")
            plt.tight_layout()
            if is_ber:
                if mod_format == "QPSK":
                    bottom_ylim = 8e-5
                    top_ylim = 6e-2
                else:
                    bottom_ylim = 8e-4
                    top_ylim = 2e-1
                plt.ylim(bottom=bottom_ylim, top=top_ylim)
            else:
                if mod_format == "QPSK":
                    bottom_ylim = 24
                else:
                    bottom_ylim = 9
                plt.ylim(bottom=bottom_ylim)
            # plt.show()
            # Construct a filename with loop indices and/or values
            filename = (
                f"{plot_type}_{mod_format}_{baud_rate.replace('.0', '')}GBaud.png"
            )
            save_path = os.path.join(
                r"C:\Users\39338\Politecnico Di Torino Studenti Dropbox\Simone Cambursano\Politecnico"
                r"\Tesi\Data-analysis\Simulation Sweeps\Rate Sweeps\Final Plots",
                filename
            )
            plt.savefig(save_path, dpi=400, bbox_inches='tight')
            # plt.show()
            plt.close()
# for i, (y_first, y_second) in enumerate(zip(data_dict_first[plot_type], data_dict_second[plot_type])):
#     # Convert to NumPy array
#     y_arr = np.array(np.concatenate((y_first, y_second)), dtype=np.float32)
#     # Replace invalid BER values (> threshold) with NaN
#     y_filtered = np.where(y_arr > filter_threshold, np.nan, y_arr)
#     # Check if all values are invalid for this OSNR
#     # if np.isnan(y_filtered).all():
#     #     print(f"⚠️  All BER values invalid (>{filter_threshold * scale:.2f}{'' if is_ber else '%'})"
#     #           f" for OSNR={OSNR_dB_vect[i]:.2f} dB — skipping {cr_algo_label} CR Algorithm.")
#     #     all_osnr_valid = False
#     #     break
#     if np.isnan(y_filtered).all():
#         filtered_curves.append(np.nan)
#     else:
#         filtered_curves.append(np.nanmin(y_filtered))
#
#     # Compute the mean ignoring NaNs
#     filtered_curves.append(np.nanmin(y_filtered))
# if is_ber:
#     if mod_format == "QPSK":
#         bottom_ylim = 6e-5
#         top_ylim = 6e-2
#     else:
#         bottom_ylim = 3e-4
#         top_ylim = 1.2e-1
#     plt.ylim(bottom=bottom_ylim, top=top_ylim)
# else:
#     if mod_format == "QPSK":
#         bottom_ylim = 24
#     else:
#         bottom_ylim = 9
#     plt.ylim(bottom=bottom_ylim)
# # Construct a filename with loop indices and/or values
# filename = (
#     f"{plot_type}_{mod_format}_{baud_rate.replace('.0', '')}GBaud.png"
# )
# save_path = os.path.join(
#     r"C:\Users\39338\Politecnico Di Torino Studenti Dropbox\Simone Cambursano\Politecnico"
#     r"\Tesi\Data-analysis\Simulation Sweeps\Rate Sweeps\Final Plots",
#     filename
# )
# plt.savefig(save_path, dpi=400, bbox_inches='tight')
# plt.close()
# y_values_columns = {
#     "berTot": {"title": "BER", "ylabel": "BER"},
#     # "EVMTot": {"title": "EVM", "ylabel": "EVM [%]"},
#     # "berEVMTot": {"title": "BER$_{\mathrm{EVM}}$", "ylabel": r"BER$_{\mathrm{EVM}}$"}
# }
# # y_values_columns = {
# #     "berX": {"title": "BER", "ylabel": "BER"},
# #     "EVMX": {"title": "EVM", "ylabel": "EVM [%]"},
# #     "berEVMX": {"title": "BER$_{\mathrm{EVM}}$", "ylabel": r"BER$_{\mathrm{EVM}}$"}
# # }
