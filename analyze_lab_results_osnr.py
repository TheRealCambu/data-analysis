import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from commun_utils.theoretical_formulas import (
    theoretical_ber_vs_snr,
    theoretical_evm_vs_osnr,
    theoretical_ber_from_evm,
    osnr_to_snr
)
from commun_utils.utils import apply_plt_personal_settings, filter_outliers


def compute_osnr_penalty(x_data, y_data, theory_y_data, theory_x_data, fec_threshold):
    if y_data.shape[0] <= 2:
        return np.nan, ""
    if np.nanmin(y_data) <= fec_threshold <= np.nanmax(y_data):
        try:
            f_interp = interp1d(y_data, x_data, bounds_error=False)
            osnr_data_at_fec = float(f_interp(fec_threshold))
        except (ValueError, TypeError) as e:
            print(f"[Warning] Could not interpolate OSNR at FEC: {e}")
            osnr_data_at_fec = np.nan
    else:
        osnr_data_at_fec = np.nan

    penalty_label = ""
    if not np.isnan(osnr_data_at_fec):
        try:
            f_theory_interp = interp1d(theory_y_data, theory_x_data, bounds_error=False)
            osnr_theory_at_fec = float(f_theory_interp(fec_threshold))
            osnr_penalty = osnr_data_at_fec - osnr_theory_at_fec
            penalty_label = f"  (Penalty @ FEC: {osnr_penalty:.2f} dB)"
        except (ValueError, TypeError) as e:
            print(f"[Warning] Could not compute theoretical OSNR at FEC: {e}")
    return penalty_label



def plot_multiple_ber(
        kind_of_plot: str,
        algo_type: str,
        data_vectors: List[np.ndarray],
        filter_threshold: float,
        fec_threshold: float,
        filename: str,
        x_values_sorted_indices_list: List[np.ndarray],
        x_values_data_list: List[np.ndarray],
        x_values_dense: np.ndarray,
        theory_value: np.ndarray,
        extra_title_label: str,
        legend_labels: List[str],
        save_plot: bool,
        directory_to_save_images: str,
        base_string_for_saving_image: str,
        alternative_plot: str
) -> None:
    label_dict = {
        "ber": {
            "title": "BER",
            "ylabel": "BER"
        },
        "evm": {
            "title": "EVM",
            "ylabel": r"EVM$_{\mathrm{m}}$ [%]"
        },
        "ber_evm": {
            "title": "BER$_{\mathrm{EVM}}$",
            "ylabel": r"BER$_{\mathrm{EVM}}$"
        },
    }
    label_info = label_dict.get(kind_of_plot, label_dict["ber"])

    plt.figure()
    markers = ['o', 's', 'v', '^', 'd', 'x']
    colors = plt.cm.tab10(np.linspace(0, 1, len(data_vectors)))

    vector_lengths = []
    if alternative_plot == 'max':
        plot_type = 'Maximum'
    elif alternative_plot == 'min':
        plot_type = 'Minimum'
    else:
        plot_type = 'Average'

    for idx, (data_vector, x_idx, x_data, legend_label) in enumerate(
            zip(data_vectors, x_values_sorted_indices_list, x_values_data_list, legend_labels)
    ):
        # Filter outliers
        # if ((filter_threshold < 5e-1 and "ber" in kind_of_plot) or
        #     (filter_threshold < 1.5 and "ber" not in kind_of_plot)) and not alternative_plot:
        filtered_data_tot = np.array(
            filter_outliers(
                upper_threshold=filter_threshold,
                lower_threshold=1e-31,
                input_values=data_vector[x_idx]
            )
        )
        # else:
        #     filtered_data_tot = np.array(data_vector[x_idx])

        # Each row corresponds to a single x value with multiple samples
        # We keep a row only if it has at least one non-NaN and nonzero value
        valid_mask = np.any(~np.isnan(filtered_data_tot) & (filtered_data_tot != 0), axis=1)

        # Apply mask to both x and data
        x_data_valid = x_data[valid_mask]
        filtered_data_valid = filtered_data_tot[valid_mask, :]

        # Mean, min, max
        data_mean = np.nanmean(filtered_data_valid, axis=1)
        data_min = np.nanmin(filtered_data_valid, axis=1)
        data_max = np.nanmax(filtered_data_valid, axis=1)

        # Plot each dataset
        marker = markers[idx % len(markers)]
        color = colors[idx]

        if kind_of_plot == "ber":
            if alternative_plot == 'max':
                mask = (data_max <= filter_threshold) & (data_max > 1e-32)
                x_data_filtered = x_data_valid[mask]
                filtered_data_max = data_max[mask]
                if filtered_data_max.shape[0] > 2:
                    penalty_label = compute_osnr_penalty(
                        x_data=x_data_filtered,
                        y_data=filtered_data_max,
                        theory_y_data=theory_value,
                        theory_x_data=x_values_dense,
                        fec_threshold=fec_threshold,
                    )
                    # Final plot
                    plt.semilogy(
                        x_data_filtered, filtered_data_max, marker + '-',
                        color=color,
                        label=f"{legend_label}{penalty_label}"
                    )
                    vector_lengths.append(x_data_filtered)
            elif alternative_plot == 'min':
                mask = (data_min <= filter_threshold) & (data_min > 1e-32)
                x_data_filtered = x_data_valid[mask]
                filtered_data_min = data_min[mask]
                if filtered_data_min.shape[0] > 2:
                    penalty_label = compute_osnr_penalty(
                        x_data=x_data_filtered,
                        y_data=filtered_data_min,
                        theory_y_data=theory_value,
                        theory_x_data=x_values_dense,
                        fec_threshold=fec_threshold,
                    )
                    plt.semilogy(
                        x_data_filtered, filtered_data_min, marker + '-',
                        color=color, label=f"{legend_label}{penalty_label}"
                    )
                    vector_lengths.append(x_data_filtered)
            else:
                if data_mean.shape[0] > 2:
                    penalty_label = compute_osnr_penalty(
                        x_data=x_data_valid,
                        y_data=data_mean,
                        theory_y_data=theory_value,
                        theory_x_data=x_values_dense,
                        fec_threshold=fec_threshold,
                    )
                    plt.semilogy(
                        x_data_valid, data_mean, marker + '-',
                        color=color, label=f"{legend_label}{penalty_label}"
                    )
                    plt.fill_between(x_data_valid, data_min, data_max, alpha=0.2, color=color)
                    vector_lengths.append(x_data_valid)
        elif kind_of_plot == "ber_evm":
            if alternative_plot == 'max':
                mask = (data_max <= filter_threshold) & (data_max > 1e-32)
                x_data_filtered = x_data_valid[mask]
                filtered_data_max = data_max[mask]
                if filtered_data_max.shape[0] > 2:
                    # Final plot
                    plt.semilogy(x_data_filtered, filtered_data_max, marker + '-', color=color, label=f"{legend_label}")
                    vector_lengths.append(x_data_filtered)
            elif alternative_plot == 'min':
                mask = (data_min <= filter_threshold) & (data_min > 1e-32)
                x_data_filtered = x_data_valid[mask]
                filtered_data_min = data_min[mask]
                if filtered_data_min.shape[0] > 2:
                    plt.semilogy(x_data_filtered, filtered_data_min, marker + '-', color=color, label=f"{legend_label}")
                    vector_lengths.append(x_data_filtered)
            else:
                if data_mean.shape[0] > 2:
                    plt.semilogy(x_data_valid, data_mean, marker + '-', color=color, label=f"{legend_label}")
                    plt.fill_between(x_data_valid, data_min, data_max, alpha=0.2, color=color)
                    vector_lengths.append(x_data_valid)
        else:
            if alternative_plot == 'max':
                mask = (data_max <= filter_threshold) & (data_max > 1e-32)
                x_data_filtered = x_data_valid[mask]
                filtered_data_max = data_max[mask]
                if filtered_data_max.shape[0] > 2:
                    plt.plot(
                        x_data_filtered, filtered_data_max * 100, marker + '-',
                        color=color, label=f"{legend_label}"
                    )
                    vector_lengths.append(x_data_filtered)
            elif alternative_plot == 'min':
                mask = (data_min <= filter_threshold) & (data_min > 1e-32)
                x_data_filtered = x_data_valid[mask]
                filtered_data_min = data_min[mask]
                if filtered_data_min.shape[0] > 2:
                    plt.plot(
                        x_data_filtered, filtered_data_min * 100, marker + '-',
                        color=color, label=f"{legend_label}"
                    )
                    vector_lengths.append(x_data_filtered)
            else:
                if data_mean.shape[0] > 2:
                    plt.plot(
                        x_data_valid, data_mean * 100, marker + '-', color=color, label=f"{legend_label}")
                    plt.fill_between(x_data_valid, data_min * 100, data_max * 100, alpha=0.2, color=color)
                    vector_lengths.append(x_data_valid)

    temp_min = np.min([np.min(x) for x in vector_lengths if len(x) > 0])
    temp_max = np.max([np.max(x) for x in vector_lengths if len(x) > 0])
    plt.xticks(np.arange(temp_min - 1, temp_max + 1, 1))
    plt.xlim(left=16.0, right=25.1)
    # plt.xlim(left=15.7, right=23.7)

    if 'ber' in kind_of_plot:
        plt.semilogy(
            x_values_dense, theory_value, '-.', color="darkred",
            linewidth=2.0, label=f"Theoretical {label_info['title']}"
        )
        if 'evm' in kind_of_plot:
            plt.xticks(np.arange(temp_min - 3, temp_max + 3, 3))
            plt.xlim(left=16.0, right=38.3)
            plt.ylim(top=8e-2, bottom=1e-5)
            # plt.ylim(top=8e-2, bottom=2e-6)
        else:
            plt.ylim(top=8e-2, bottom=2e-3)
            # plt.ylim(top=8e-2, bottom=2e-3)
    else:
        plt.plot(
            x_values_dense, theory_value * 100, '-.',
            linewidth=2.0, color="darkred", label=f"Theoretical {label_info['title']}"
        )
        plt.ylim(top=26, bottom=4)

    # Reference lines
    if kind_of_plot not in ["evm", "ber_evm"]:
        plt.axhline(
            fec_threshold, color='darkred', linestyle=':', linewidth=2.5,
            label=f"FEC threshold = {fec_threshold:.0e}"
        )

    # Labels and title
    plt.xlabel("OSNR [dB] per 0.1nm")
    plt.ylabel(label_info["ylabel"])
    plt.title(f"{plot_type} {label_info['title']} vs OSNR {extra_title_label}")
    plt.legend(loc="best")
    plt.grid(True, which="both")
    plt.tight_layout()

    if save_plot:
        image_name = filename.replace("PROCESSED_osnr_sweep", "").replace(".npz", "")
        if alternative_plot in ["min", "max"]:
            base_string_for_saving_image = f"{alternative_plot}_" + base_string_for_saving_image
        full_path = os.path.join(
            directory_to_save_images,
            algo_type.lower().replace(" ", "_") + '_' + base_string_for_saving_image + image_name + ".png"
        )
        plt.savefig(full_path, dpi=400)
        print(f"✅ Saved: {full_path}")

    plt.show()


# root_folder = (r"C:\Users\39338\Politecnico Di Torino Studenti Dropbox\Simone Cambursano\Politecnico"
#                r"\Tesi\Data-analysis\Lab results\v4 - Processed Datasets -- Final OPT")

root_folder = (r"C:\Users\39338\Politecnico Di Torino Studenti Dropbox\Simone Cambursano\Politecnico"
               r"\Tesi\Data-analysis\Lab results\v3 - Processed Datasets -- First OPT")
# tr_algo_list = ["Gardner", "Frequency Domain"]
# baud_rate_and_mod_format_list = ["30GBd QPSK", "34.28GBd QPSK", "40GBd QPSK", "30GBd 16QAM", "34.28GBd 16QAM"]
tr_algo_list = ["Gardner"]
# tr_algo_list = ["Frequency Domain"]
baud_rate_and_mod_format_list = ["30GBd QPSK"]
# baud_rate_and_mod_format_list = ["34.28GBd QPSK"]
# baud_rate_and_mod_format_list = ["40GBd QPSK"]
# baud_rate_and_mod_format_list = ["30GBd 16QAM"]
# baud_rate_and_mod_format_list = ["34.28GBd 16QAM"]

sweep_type = 'osnr'
folder_to_store_images = os.path.join(root_folder, "Final Plots", sweep_type.upper())

files_dict = {}

# Collect files
for baud_rate_and_mod_format in baud_rate_and_mod_format_list:
    for tr_algo in tr_algo_list:
        folder_path = os.path.join(root_folder, tr_algo, baud_rate_and_mod_format)
        files_in_current_folder = [
            f for f in os.listdir(folder_path)
            if f.endswith(".npz") and sweep_type in f
        ]
        print(files_in_current_folder)

        for npz_file in files_in_current_folder:
            dpe_type = "w_dpe" if "w_dpe" in npz_file else "wo_dpe"

            # Ensure nested keys exist correctly
            files_dict.setdefault(baud_rate_and_mod_format, {})
            files_dict[baud_rate_and_mod_format].setdefault(tr_algo, {})
            files_dict[baud_rate_and_mod_format][tr_algo][dpe_type] = os.path.join(folder_path, npz_file)

for baud_rate_and_mod_format, algo_dict in files_dict.items():
    for algo_type, dpe_dict in algo_dict.items():
        w_dpe_file = dpe_dict.get('w_dpe')
        wo_dpe_file = dpe_dict.get('wo_dpe')

        if w_dpe_file is None or wo_dpe_file is None:
            print(f"⚠️ Missing DPE variant for {baud_rate_and_mod_format} / {algo_type}")
            continue

        print(f"\nComparing {algo_type} / {baud_rate_and_mod_format}: With DPE vs. Without DPE")
        print(f"Current file: {os.path.basename(wo_dpe_file)}")

        with (
            np.load(wo_dpe_file, allow_pickle=True) as npz_off,
            np.load(w_dpe_file, allow_pickle=True) as npz_on
        ):
            data_off = dict(npz_off)
            data_on = dict(npz_on)

            # Determine modulation format
            is_qpsk = "QPSK" in baud_rate_and_mod_format
            bits_per_symbol = 2 if is_qpsk else 4
            const_cardinality = 4 if is_qpsk else 16
            title_label_for_plot_tot = "(DP-QPSK" if is_qpsk else "(DP-16QAM"

            # Compute SNR (linear) from measured OSNR points
            osnr_dB_vect = data_off[sweep_type]
            snr_lin = osnr_to_snr(osnr_dB_vect=osnr_dB_vect,
                                  symbol_rate=np.unique(data_off["symbol_rate"]))

            # Create a dense and extended OSNR range
            osnr_dB_dense = np.linspace(np.min(osnr_dB_vect) - 5, np.max(osnr_dB_vect) + 5, 1000)

            # Convert extended OSNR (dB) → linear SNR
            snr_lin_dense = osnr_to_snr(osnr_dB_vect=osnr_dB_dense, symbol_rate=np.unique(data_off["symbol_rate"]))
            ber_theory = theoretical_ber_vs_snr(
                snr=snr_lin_dense,
                M=const_cardinality
            )
            # Note: even if we see 'osnr', actually, it refers to the electrical OSNR
            # related to the BW of the receiver, what we call SNR
            evm_theory = theoretical_evm_vs_osnr(
                osnr=snr_lin_dense,
                bits_per_symbol=bits_per_symbol,
                M=const_cardinality
            )
            ber_evm_theory = theoretical_ber_from_evm(
                EVM_m=evm_theory,
                M=const_cardinality
            )

            # Sort OSNR axis
            off_data = data_off[sweep_type]
            on_data = data_on[sweep_type]

            x_idx_off = np.argsort(off_data)
            x_idx_on = np.argsort(on_data)

            x_values_off = off_data[x_idx_off]
            x_values_on = on_data[x_idx_on]

            # Apply plotting settings
            apply_plt_personal_settings()

            # for kind_of_plot in ["ber_evm"]:
            for kind_of_plot in ["ber", "evm", "ber_evm"]:
                theory_value = {
                    "ber": ber_theory,
                    "evm": evm_theory,
                    "ber_evm": ber_evm_theory
                }[kind_of_plot]

                for polarization in ["_tot", "_x", "_y"]:
                    key = kind_of_plot + polarization
                    if polarization == "_x":
                        temp = title_label_for_plot_tot + ", X Pol)"
                    elif polarization == "_y":
                        temp = title_label_for_plot_tot + ", Y Pol)"
                    else:
                        temp = title_label_for_plot_tot + ")"

                    # Plot DPE OFF vs DPE ON
                    plot_multiple_ber(
                        kind_of_plot=kind_of_plot,
                        algo_type=algo_type,
                        data_vectors=[data_off[key], data_on[key]],
                        filter_threshold=6e-1 if "ber" in kind_of_plot else 1.6,
                        # filter_threshold=4e-1 if "ber" in kind_of_plot else 0.7,
                        fec_threshold=2e-2,
                        filename=os.path.basename(wo_dpe_file),
                        x_values_sorted_indices_list=[x_idx_off, x_idx_on],
                        x_values_data_list=[x_values_off, x_values_on],
                        x_values_dense=osnr_dB_dense,
                        theory_value=theory_value,
                        extra_title_label=temp,
                        legend_labels=["DPE OFF", "DPE ON"],
                        save_plot=True,
                        directory_to_save_images=folder_to_store_images,
                        base_string_for_saving_image=f"{key}_vs_osnr",
                        alternative_plot="min"
                    )

# Filter outliers
# # if ((filter_threshold < 5e-1 and "ber" in kind_of_plot) or
# #     (filter_threshold < 1.5 and "ber" not in kind_of_plot)) and not alternative_plot:
# filtered_data_tot = np.array(
#     filter_outliers(
#         upper_threshold=filter_threshold,
#         lower_threshold=1e-20,
#         input_values=data_vector[x_idx]
#     )
# )
# # else:
# #     filtered_data_tot = np.array(data_vector[x_idx])