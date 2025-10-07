import os
import re
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from commun_utils.theoretical_formulas import (
    theoretical_ber_vs_snr,
    theoretical_evm_vs_osnr,
    theoretical_ber_from_evm,
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
        plot_max: bool = False
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
    for idx, (data_vector, x_idx, x_data, legend_label) in enumerate(zip(
            data_vectors, x_values_sorted_indices_list, x_values_data_list,
            legend_labels
    )):
        # Filter outliers
        if ((filter_threshold < 5e-1 and "ber" in kind_of_plot) or
            (filter_threshold < 1.5 and "ber" not in kind_of_plot)) and not plot_max:
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
        color = colors[idx]

        if 'ber' in kind_of_plot:
            if plot_max:
                mask = (data_max <= filter_threshold) & (data_max > 1e-32)
                x_data_filtered = x_data_valid[mask]
                filtered_data_max = data_max[mask]
                if filtered_data_max.shape[0] > 2:
                    plt.semilogy(
                        x_data_filtered, filtered_data_max, marker + '-',
                        color=color, label=f"{legend_label}"
                    )
                    vector_lengths.append(x_data_filtered)
            else:
                if data_mean.shape[0] > 2:
                    plt.semilogy(x_data_valid, data_mean, marker + '-', color=color, label=f"{legend_label}")
                    plt.fill_between(x_data_valid, data_min, data_max, alpha=0.2, color=color)
                    vector_lengths.append(x_data_valid)
        else:
            if plot_max:
                mask = (data_max <= filter_threshold) & (data_max > 1e-32)
                x_data_filtered = x_data_valid[mask]
                filtered_data_max = data_max[mask]
                if filtered_data_max.shape[0] > 2:
                    plt.plot(
                        x_data_filtered, filtered_data_max * 100, marker + '-',
                        color=color, label=f"{legend_label}"
                    )
                    vector_lengths.append(x_data_filtered)
            else:
                if data_mean.shape[0] > 2:
                    plt.plot(x_data_valid, data_mean * 100, marker + '-', color=color, label=f"{legend_label}")
                    plt.fill_between(x_data_valid, data_min * 100, data_max * 100, alpha=0.2, color=color)
                    vector_lengths.append(x_data_valid)

    temp_min = np.min([np.min(x) for x in vector_lengths if len(x) > 0])
    temp_max = np.max([np.max(x) for x in vector_lengths if len(x) > 0])
    plt.xticks(np.arange(temp_min, temp_max + 0.5, 0.5))

    # Reference lines
    if "ber" in kind_of_plot:
        plt.axhline(
            fec_threshold, color='darkred', linestyle=':', linewidth=2.5,
            label=f"FEC threshold = {fec_threshold:.0e}"
        )
        if theory_value > 1e-30:
            plt.axhline(
                theory_value, color='darkblue', linestyle=':',
                linewidth=2.5, label=f"Theoretical {label_info['title']} = {theory_value:.2e}"
            )
    else:
        plt.axhline(
            theory_value * 100, color='darkblue', linestyle=':', linewidth=2.5,
            label=f"Theoretical EVM = {theory_value * 100:.3g}%"
        )

    # Labels and title
    plt.xlabel("Frequency Offset [GHz]")
    plt.ylabel(label_info["ylabel"])
    plt.title(
        f"{'Maximum filtered' if plot_max else 'Average'} {label_info['title']}"
        f" vs Frequency Offset {extra_title_label}"
    )
    plt.legend(loc="best")
    plt.grid(True, which="both")
    plt.tight_layout()

    # Save (after show() or before, both fine)
    if save_plot:
        image_name = filename.replace("PROCESSED_fo_sweep", "").replace(".npz", "")
        if plot_max:
            base_string_for_saving_image = ("max_filtered_" if plot_max else "") + base_string_for_saving_image
        full_path = os.path.join(
            directory_to_save_images,
            base_string_for_saving_image + image_name + ".png"
        )
        plt.savefig(full_path, dpi=400)
        print(f"✅ Saved: {full_path}")

    plt.show()


root_folder = r"C:\Users\39338\Politecnico Di Torino Studenti Dropbox\Simone Cambursano\Politecnico\Tesi\Data-analysis\Lab results\v4 - Processed Datasets -- Final OPT"
tr_algo_list = ["Gardner", "Frequency Domain"]
# baud_rate_and_mod_format_list = ["30GBd QPSK"]
# baud_rate_and_mod_format_list = ["34.28GBd QPSK"]
baud_rate_and_mod_format_list = ["30GBd 16QAM"]
# baud_rate_and_mod_format_list = ["34.28GBd 16QAM"]
# baud_rate_and_mod_format_list = ["40GBd QPSK"]

plot_type = 'fo'
folder_to_store_images = os.path.join(root_folder, "Final Plots", plot_type.upper())

files_dict = {
    fmt: {} for fmt in baud_rate_and_mod_format_list
}

# Collect files
for baud_rate_and_mod_format in baud_rate_and_mod_format_list:
    for tr_algo in tr_algo_list:
        folder_path = os.path.join(root_folder, tr_algo, baud_rate_and_mod_format)
        files_in_current_folder = [f for f in os.listdir(folder_path) if f.endswith(".npz") and plot_type in f]
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
            gardner_data = data_gardner[plot_type]
            x_values_sorted_indices_gardner = np.argsort(gardner_data)
            x_values_data_gardner = gardner_data[x_values_sorted_indices_gardner]

            fd_data = data_freqdom[plot_type]
            x_values_sorted_indices_fd = np.argsort(fd_data)
            x_values_data_fd = fd_data[x_values_sorted_indices_fd]

            # Apply personal matplotlib settings
            apply_plt_personal_settings()

            # for kind_of_plot in ['ber', 'evm', 'ber_evm']:
            for kind_of_plot in ['ber', 'evm']:
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

                    # Plot both algorithms
                    plot_multiple_ber(
                        kind_of_plot=kind_of_plot,
                        data_vectors=[data_gardner[key], data_freqdom[key]],
                        # filter_threshold=5e-1 if 'ber' in kind_of_plot else 1.6,
                        filter_threshold=3e-2 if 'ber' in kind_of_plot else 0.25,
                        fec_threshold=2e-2,
                        filename=os.path.basename(gardner_file),
                        x_values_sorted_indices_list=[x_values_sorted_indices_gardner, x_values_sorted_indices_fd],
                        x_values_data_list=[x_values_data_gardner, x_values_data_fd],
                        extra_title_label=temp,
                        legend_labels=["GardnerTimeRec", "FDTimeRec"],
                        theory_value=theory_value,
                        save_plot=False,
                        directory_to_save_images=folder_to_store_images,
                        base_string_for_saving_image=f"{key}_vs_fo",
                        plot_max=True
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
