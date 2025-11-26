import os
import re
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from commun_utils.theoretical_formulas import (
    theoretical_ber_vs_snr,
    theoretical_evm_vs_osnr,
    theoretical_ber_from_evm,
    theoretical_evm_from_ber,
    osnr_to_snr
)
from commun_utils.utils import apply_plt_personal_settings, filter_outliers


def extract_osnr_from_filename(filename: str):
    """Extract OSNR value (e.g., 13, 40) from filename like '_13dB_OSNR_'."""
    match = re.search(r"_([0-9]+)dB_OSNR", filename)
    return int(match.group(1)) if match else None


def plot_multiple_ber(
        kind_of_plot: str,
        data_vectors: List[np.ndarray],
        filter_threshold: float,
        fec_threshold: float,
        filename: str,
        x_values_sorted_indices_list: List[np.ndarray],
        x_values_data_list: List[np.ndarray],
        extra_title_label: str,
        legend_labels: List[str],
        theory_value: float,
        save_plot: bool,
        directory_to_save_images: str,
        base_string_for_saving_image: str,
        plot_type: str
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
    linestyles = ['-', '--']
    linewidths = [2.5, 1.5]
    # colors = plt.cm.tab10(np.linspace(0.5, 0, len(data_vectors)))
    colors = ['black', 'red']

    vector_lengths = []
    for idx, (data_vector, x_idx, x_data, legend_label) in enumerate(zip(
            data_vectors, x_values_sorted_indices_list, x_values_data_list,
            legend_labels
    )):
        # Filter outliers
        if (((filter_threshold < 5e-1 and "ber" in kind_of_plot) or
             (filter_threshold < 1.5 and "ber" not in kind_of_plot)) and
                plot_type.lower() not in ["min", "max"]):
            filtered_data_tot = np.array(
                filter_outliers(
                    upper_threshold=filter_threshold,
                    input_values=data_vector[x_idx]
                )
            )
        else:
            filtered_data_tot = np.array(data_vector[x_idx])

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
        linestyle = linestyles[idx % len(linestyles)]
        linewidth = linewidths[idx % len(linewidths)]
        color = colors[idx]

        if 'ber' in kind_of_plot:
            if plot_type == "max":
                mask = (data_max <= filter_threshold) & (data_max > 1e-32)
                x_data_filtered = x_data_valid[mask]
                filtered_data_max = data_max[mask]
                if filtered_data_max.shape[0] > 2:
                    plt.semilogy(
                        x_data_filtered, filtered_data_max,
                        marker + linestyle,
                        # linestyle,
                        markersize=5,
                        linewidth=linewidth,
                        color=color, label=f"{legend_label}"
                    )
                    vector_lengths.append(x_data_filtered)
            elif plot_type == "min":
                mask = (data_min <= filter_threshold) & (data_min > 1e-32)
                x_data_filtered = x_data_valid[mask]
                filtered_data_min = data_min[mask]
                if filtered_data_min.shape[0] > 2:
                    plt.semilogy(
                        x_data_filtered, filtered_data_min,
                        marker + linestyle,
                        # linestyle,
                        markersize=5,
                        linewidth=linewidth,
                        color=color, label=f"{legend_label}"
                    )
                    vector_lengths.append(x_data_filtered)
            else:
                if data_mean.shape[0] > 2:
                    plt.semilogy(x_data_valid, data_mean,
                                 marker + linestyle,
                                 # linestyle,
                                 markersize=5,
                                 linewidth=linewidth,
                                 color=color, label=f"{legend_label}")
                    # plt.fill_between(x_data_valid, data_min, data_max, alpha=0.2, color=color)
                    vector_lengths.append(x_data_valid)
        else:
            if plot_type == "max":
                mask = (data_max <= filter_threshold) & (data_max > 1e-32)
                x_data_filtered = x_data_valid[mask]
                filtered_data_max = data_max[mask]
                if filtered_data_max.shape[0] > 2:
                    plt.plot(
                        x_data_filtered, filtered_data_max * 100,
                                         marker + linestyle,
                        # linestyle,
                        markersize=5,
                        linewidth=linewidth,
                        color=color, label=f"{legend_label}"
                    )
                    vector_lengths.append(x_data_filtered)
            elif plot_type == "min":
                mask = (data_min <= filter_threshold) & (data_min > 1e-32)
                x_data_filtered = x_data_valid[mask]
                filtered_data_min = data_min[mask]
                if filtered_data_min.shape[0] > 2:
                    plt.plot(
                        x_data_filtered, filtered_data_min * 100,
                                         marker + linestyle,
                        # linestyle,
                        markersize=5,
                        linewidth=linewidth,
                        color=color, label=f"{legend_label}"
                    )
                    vector_lengths.append(x_data_filtered)
            else:
                if data_mean.shape[0] > 2:
                    plt.plot(x_data_valid, data_mean * 100,
                             marker + linestyle,
                             # linestyle,
                             markersize=5,
                             linewidth=linewidth,
                             color=color, label=f"{legend_label}")
                    # plt.fill_between(x_data_valid, data_min * 100, data_max * 100, alpha=0.2, color=color)
                    vector_lengths.append(x_data_valid)

    temp_min = np.nanmin([np.nanmin(x) for x in vector_lengths if len(x) > 0])
    temp_max = np.nanmax([np.nanmax(x) for x in vector_lengths if len(x) > 0])
    plt.xticks(np.arange(temp_min, temp_max + 0.5, 0.5))

    # Reference lines
    if "ber" in kind_of_plot:
        if osnr_val < 50.0:
            plt.axhline(
                fec_threshold, color='darkred', linestyle=':', linewidth=2.5,
                label=f"FEC threshold={fec_threshold:.0e}"
            )
        if theory_value > 1e-30:
            plt.axhline(
                theory_value, color='darkblue', linestyle=':', linewidth=2.5,
                label=f"Theoretical {label_info['title']}={theory_value:.2e}"
            )
    else:
        if osnr_val < 50.0:
            plt.axhline(
                fec_threshold * 100, color='darkred', linestyle=':', linewidth=2.5,
                label=f"FEC threshold={fec_threshold * 100:.3g}%"
            )
        plt.axhline(
            theory_value * 100, color='darkblue', linestyle=':', linewidth=2.5,
            label=f"Theoretical EVM={theory_value * 100:.3g}%"
        )

    # Labels and title
    # plt.ylim([8, 13])
    # plt.ylim([6, 20])
    plt.ylim(top=3e-2)
    # plt.ylim(top=20)
    # plt.ylim(top=20)
    plt.xlabel("Frequency Offset [GHz]")
    plt.ylabel(label_info["ylabel"])
    plt.title(
        # f"{'Maximum' if plot_max else 'Average'} {label_info['title']}"
        f"{label_info['title']}"
        f" vs Frequency Offset {extra_title_label}"
    )
    plt.legend(loc="best")
    plt.grid(True, which="both")
    plt.tight_layout()

    # Save (after show() or before, both fine)
    if save_plot:
        image_name = filename.replace("PROCESSED_fo_sweep", "").replace(".npz", "").replace('_v1', '')
        if plot_type == "max":
            base_string_for_saving_image = "max_" + base_string_for_saving_image
        elif plot_type == "min":
            base_string_for_saving_image = "min_" + base_string_for_saving_image
        full_path = os.path.join(
            directory_to_save_images,
            base_string_for_saving_image + image_name + ".png"
        )
        plt.savefig(full_path, dpi=400)
        print(f"✅ Saved: {full_path}")

    plt.show()


root_folder = (r"C:\Users\39338\Politecnico Di Torino Studenti Dropbox\Simone Cambursano\Politecnico"
               r"\Tesi\Data-analysis\Lab results\v4 - Processed Datasets -- Final OPT")
tr_algo_list = ["Gardner", "Frequency Domain"]
# baud_rate_and_mod_format_list = ["30GBd QPSK"]
baud_rate_and_mod_format_list = ["30GBd 16QAM"]
# baud_rate_and_mod_format_list = ["30GBd QPSK", "30GBd 16QAM"]

sweep_type = 'fo'
folder_to_store_images = os.path.join(root_folder, "Final Plots", sweep_type.upper())

files_dict = {
    fmt: {} for fmt in baud_rate_and_mod_format_list
}

# Collect files
for baud_rate_and_mod_format in baud_rate_and_mod_format_list:
    for tr_algo in tr_algo_list:
        folder_path = os.path.join(root_folder, tr_algo, baud_rate_and_mod_format)
        files_in_current_folder = [f for f in os.listdir(folder_path) if f.endswith(".npz") and sweep_type in f]
        print(files_in_current_folder)
        for npz_file in files_in_current_folder:
            osnr_val = extract_osnr_from_filename(npz_file)
            if osnr_val is not None:
                if osnr_val not in files_dict[baud_rate_and_mod_format]:
                    files_dict[baud_rate_and_mod_format][osnr_val] = {}
                files_dict[baud_rate_and_mod_format][osnr_val][tr_algo] = os.path.join(folder_path, npz_file)

# Loop and plot
for baud_rate_and_mod_format, osnr_dict in files_dict.items():
    for osnr_val, algo_files in sorted(osnr_dict.items()):
        gardner_file = algo_files.get("Gardner")
        freqdom_file = algo_files.get("Frequency Domain")

        print(
            f"\n--- Gardner / {baud_rate_and_mod_format} --- vs --- Frequency Domain / {baud_rate_and_mod_format} ---")
        print(f"File: {os.path.basename(gardner_file)}")

        with (
            np.load(gardner_file, allow_pickle=True) as gardner_npz,
            np.load(freqdom_file, allow_pickle=True) as freqdom_npz
        ):
            data_gardner = dict(gardner_npz)
            data_freqdom = dict(freqdom_npz)

            # Determine modulation format
            is_qpsk = "QPSK" in baud_rate_and_mod_format
            bits_per_symbol = 2 if is_qpsk else 4
            const_cardinality = 4 if is_qpsk else 16
            title_label_for_plot_tot = "(DP-QPSK" if is_qpsk else "(DP-16QAM"

            # Compute theoretical values (assuming you already defined these functions)
            snr_lin = osnr_to_snr(osnr_val, np.unique(data_gardner['symbol_rate']))[0]
            ber_theory = theoretical_ber_vs_snr(snr_lin, const_cardinality)
            evm_theory = theoretical_evm_vs_osnr(
                bits_per_symbol=bits_per_symbol,
                M=const_cardinality,
                osnr=snr_lin,
            )
            ber_evm_theory = theoretical_ber_from_evm(
                EVM_m=evm_theory,
                M=const_cardinality
            )

            # Sort frequency offsets
            gardner_data = data_gardner[sweep_type]
            x_values_sorted_indices_gardner = np.argsort(gardner_data)
            x_values_data_gardner = gardner_data[x_values_sorted_indices_gardner]

            fd_data = data_freqdom[sweep_type]
            x_values_sorted_indices_fd = np.argsort(fd_data)
            x_values_data_fd = fd_data[x_values_sorted_indices_fd]

            # Apply personal matplotlib settings
            apply_plt_personal_settings()

            # for kind_of_plot in ['ber', 'evm', 'ber_evm']:
            for kind_of_plot in (
                    ['evm', 'ber_evm'] if "QPSK" in baud_rate_and_mod_format and osnr_val > 21 else ['ber', 'evm', 'ber_evm']
            ):
                if kind_of_plot == 'ber':
                    theory_value = ber_theory
                elif kind_of_plot == 'evm':
                    theory_value = evm_theory
                else:
                    theory_value = ber_evm_theory

                for polarization in ['_tot', '_x', '_y']:
                    key = kind_of_plot + polarization
                    if polarization == '_x':
                        temp = title_label_for_plot_tot + ', X Pol)'
                    elif polarization == '_y':
                        temp = title_label_for_plot_tot + ', Y Pol)'
                    else:
                        temp = title_label_for_plot_tot + ')'

                    # Add EVM fec threshold
                    # ber_filter = 3e-1
                    # ber_filter = 4.90e-1
                    ber_filter = 5e-1
                    evm_filter = theoretical_evm_from_ber(ber_filter, M=const_cardinality)
                    print(evm_filter)
                    if evm_filter < 0:
                        evm_filter = 150 / 100
                    ber_fec_threshold = 2e-2
                    evm_fec_threshold = theoretical_evm_from_ber(ber_fec_threshold, M=const_cardinality)
                    if evm_fec_threshold < 0:
                        evm_fec_threshold = 150 / 100
                    # Plot both algorithms
                    plot_multiple_ber(
                        kind_of_plot=kind_of_plot,
                        data_vectors=[data_gardner[key], data_freqdom[key]],
                        filter_threshold=ber_filter if 'ber' in kind_of_plot else evm_filter,
                        fec_threshold=ber_fec_threshold if 'ber' in kind_of_plot else evm_fec_threshold,
                        filename=os.path.basename(gardner_file),
                        x_values_sorted_indices_list=[x_values_sorted_indices_gardner, x_values_sorted_indices_fd],
                        x_values_data_list=[x_values_data_gardner, x_values_data_fd],
                        extra_title_label=temp,
                        legend_labels=["Gardner", "Fast square-timing"],
                        theory_value=theory_value,
                        save_plot=True,
                        directory_to_save_images=folder_to_store_images,
                        base_string_for_saving_image=f"{key}_vs_fo",
                        plot_type="min"
                    )

# tr_algo_list = ["Gardner", "Frequency Domain"]
# baud_rate_and_mod_format_list = ["30GBd QPSK", "34.28GBd QPSK", "40GBd QPSK", "30GBd 16QAM", "34.28GBd 16QAM"]

# tr_algo_list = ["Gardner"]
# tr_algo_list = ["Frequency Domain"]
# baud_rate_and_mod_format_list = ["30GBd QPSK"]
# baud_rate_and_mod_format_list = ["34.28GBd QPSK"]
# baud_rate_and_mod_format_list = ["30GBd 16QAM"]
# baud_rate_and_mod_format_list = ["34.28GBd 16QAM"]

# baud_rate_and_mod_format_list = ["40GBd QPSK"]
