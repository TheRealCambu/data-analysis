import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from commun_utils.utils import apply_plt_personal_settings, filter_outliers
from matplotlib.ticker import MultipleLocator


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
        save_plot: bool,
        directory_to_save_images: str,
        base_string_for_saving_image: str,
        alternative_plot: str,
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
    if kind_of_plot == 'evm':
        # Set y-axis tick interval to 5
        plt.gca().yaxis.set_major_locator(MultipleLocator(10))
    markers = ['o', 's', 'v', '^', 'd', 'x']
    colors = plt.cm.tab10(np.linspace(0, 1, len(data_vectors)))
    temp = ""
    vector_lengths = []
    for idx, (data_vector, x_idx, x_data, legend_label) in enumerate(zip(
            data_vectors, x_values_sorted_indices_list, x_values_data_list, legend_labels
    )):
        # Filter outliers
        if ((filter_threshold < 5e-1 and "ber" in kind_of_plot) or
            (filter_threshold < 1.5 and "ber" not in kind_of_plot)) and not alternative_plot:
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

        # Compute statistics
        data_mean = np.nanmean(filtered_data_valid, axis=1)
        data_min = np.nanmin(filtered_data_valid, axis=1)
        data_max = np.nanmax(filtered_data_valid, axis=1)

        # Plot each dataset
        marker = markers[idx % len(markers)]
        color = colors[idx]

        if 'ber' in kind_of_plot:
            if alternative_plot == 'max':
                mask = (data_max <= filter_threshold) & (data_max > 1e-32)
                x_data_filtered = x_data_valid[mask]
                filtered_data_max = data_max[mask]
                plt.semilogy(x_data_filtered, filtered_data_max, marker + '-', color=color, label=f"{legend_label}")
                temp = 'Maximum filtered'
                vector_lengths.append(x_data_filtered)
            elif alternative_plot == 'min':
                mask = (data_min <= filter_threshold) & (data_min > 1e-32)
                x_data_filtered = x_data_valid[mask]
                filtered_data_min = data_min[mask]
                plt.semilogy(x_data_filtered, filtered_data_min, marker + '-', color=color, label=f"{legend_label}")
                temp = 'Minimum filtered'
                vector_lengths.append(x_data_filtered)
            else:
                plt.semilogy(x_data_valid, data_mean, marker + '-', color=color, label=f"{legend_label}")
                plt.fill_between(x_data_valid, data_min, data_max, alpha=0.2, color=color)
                temp = 'Average'
                vector_lengths.append(x_data_valid)
        else:
            if alternative_plot == 'max':
                mask = (data_max <= filter_threshold) & (data_max > 1e-32)
                x_data_filtered = x_data_valid[mask]
                filtered_data_max = data_max[mask]
                plt.plot(x_data_filtered, filtered_data_max * 100, marker + '-', color=color, label=f"{legend_label}")
                temp = 'Maximum filtered'
                vector_lengths.append(x_data_filtered)
            elif alternative_plot == 'min':
                mask = (data_min <= filter_threshold) & (data_min > 1e-32)
                x_data_filtered = x_data_valid[mask]
                filtered_data_min = data_min[mask]
                plt.plot(x_data_filtered, filtered_data_min * 100, marker + '-', color=color, label=f"{legend_label}")
                temp = 'Minimum filtered'
                vector_lengths.append(x_data_filtered)
            else:
                plt.plot(x_data_valid, data_mean * 100, marker + '-', color=color, label=f"{legend_label}")
                plt.fill_between(x_data_valid, data_min * 100, data_max * 100, alpha=0.2, color=color)
                temp = 'Average'
                vector_lengths.append(x_data_valid)

    temp_min = np.min([np.min(x) for x in vector_lengths if len(x) > 0])
    temp_max = np.max([np.max(x) for x in vector_lengths if len(x) > 0])
    plt.xticks(np.arange(temp_min, temp_max + 1, 5))

    # Reference lines
    if "ber" in kind_of_plot:
        plt.axhline(
            fec_threshold, color='darkred', linestyle=':', linewidth=2.5,
            label=f"FEC threshold = {fec_threshold:.0e}"
        )

    # Labels and title
    plt.xlabel("ROP [dBm]")
    plt.ylabel(label_info["ylabel"])
    plt.title(f"{temp} {label_info['title']} vs ROP {extra_title_label}")
    plt.legend(loc="best")
    plt.grid(True, which="both")
    plt.tight_layout()

    if save_plot:
        image_name = filename.replace("PROCESSED_rop_sweep", "").replace(".npz", "")
        if alternative_plot in ["min", "max"]:
            base_string_for_saving_image = f"{alternative_plot}_filtered_" + base_string_for_saving_image
        full_path = os.path.join(
            directory_to_save_images,
            base_string_for_saving_image + image_name + ".png"
        )
        plt.savefig(full_path, dpi=400)
        print(f"✅ Saved: {full_path}")

    plt.show()


# --- File setup ---
root_folder = r"C:\Users\39338\Politecnico Di Torino Studenti Dropbox\Simone Cambursano\Politecnico\Tesi\Data-analysis\Lab results\v4 - Processed Datasets -- Final OPT"

plot_type = 'rop'
folder_to_store_images = os.path.join(root_folder, "Final Plots", plot_type.upper())

# Apply personal matplotlib settings
apply_plt_personal_settings()

# One .npz per algorithm and configuration
files_dict = {
    # "30GBd QPSK": {
    #     "Gardner": os.path.join(root_folder, "Gardner", "30GBd QPSK", "PROCESSED_rop_sweep_30GBd_DP_QPSK_w_dpe_v1.npz"),
    #     "Frequency Domain": os.path.join(root_folder, "Frequency Domain", "30GBd QPSK",
    #                                      "PROCESSED_rop_sweep_30GBd_DP_QPSK_w_dpe_v1.npz")
    # },
    "30GBd 16QAM": {
        "Gardner": os.path.join(root_folder, "Gardner", "30GBd 16QAM",
                                "PROCESSED_rop_sweep_30GBd_DP_16QAM_w_dpe_v1.npz"),
        "Frequency Domain": os.path.join(root_folder, "Frequency Domain", "30GBd 16QAM",
                                         "PROCESSED_rop_sweep_30GBd_DP_16QAM_w_dpe_v1.npz")
    },
    # "34.28GBd QPSK": {
    #     "Gardner": os.path.join(root_folder, "Gardner", "34.28GBd QPSK",
    #                             "PROCESSED_rop_sweep_34_28GBd_DP_QPSK_w_dpe_v1.npz"),
    #     "Frequency Domain": os.path.join(root_folder, "Frequency Domain", "34.28GBd QPSK",
    #                                      "PROCESSED_rop_sweep_34_28GBd_DP_QPSK_w_dpe_v1.npz")
    # },
    "34.28GBd 16QAM": {
        "Gardner": os.path.join(root_folder, "Gardner", "34.28GBd 16QAM",
                                "PROCESSED_rop_sweep_34_28GBd_DP_16QAM_w_dpe_v1.npz"),
        "Frequency Domain": os.path.join(root_folder, "Frequency Domain", "34.28GBd 16QAM",
                                         "PROCESSED_rop_sweep_34_28GBd_DP_16QAM_w_dpe_v1.npz")
    },
}

# --- Main plotting loop ---
for baud_rate_and_mod_format, algo_files in files_dict.items():
    gardner_file = algo_files["Gardner"]
    freqdom_file = algo_files["Frequency Domain"]

    print(f"\nGardner / {baud_rate_and_mod_format} vs. Frequency Domain / {baud_rate_and_mod_format}")

    with (
        np.load(gardner_file, allow_pickle=True) as gardner_npz,
        np.load(freqdom_file, allow_pickle=True) as freqdom_npz
    ):
        data_gardner = dict(gardner_npz)
        data_freqdom = dict(freqdom_npz)

        # Modulation info
        is_qpsk = "QPSK" in baud_rate_and_mod_format
        bits_per_symbol = 2 if is_qpsk else 4
        const_cardinality = 4 if is_qpsk else 16
        title_label_for_plot_tot = "(DP-QPSK" if is_qpsk else "(DP-16QAM"

        # Sort ROP (x-values)
        gardner_data = data_gardner[plot_type]
        x_values_sorted_indices_gardner = np.argsort(gardner_data)
        x_values_data_gardner = gardner_data[x_values_sorted_indices_gardner]

        fd_data = data_freqdom[plot_type]
        x_values_sorted_indices_fd = np.argsort(fd_data)
        x_values_data_fd = fd_data[x_values_sorted_indices_fd]

        # for kind_of_plot in ['ber', 'evm', 'ber_evm']:
        for kind_of_plot in ['evm']:
            for polarization in ['_tot', '_x', '_y']:
                key = kind_of_plot + polarization
                if polarization == '_x':
                    final_title_label = title_label_for_plot_tot + ', X Pol)'
                elif polarization == '_y':
                    final_title_label = title_label_for_plot_tot + ', Y Pol)'
                else:
                    final_title_label = title_label_for_plot_tot + ')'
                plot_multiple_ber(
                    kind_of_plot=kind_of_plot,
                    data_vectors=[data_gardner[key], data_freqdom[key]],
                    filter_threshold=6e-1 if 'ber' in kind_of_plot else 1.6,
                    # filter_threshold=3e-2 if 'ber' in kind_of_plot else 0.2,
                    fec_threshold=2e-2,
                    filename=os.path.basename(gardner_file),
                    x_values_sorted_indices_list=[x_values_sorted_indices_gardner, x_values_sorted_indices_fd],
                    x_values_data_list=[x_values_data_gardner, x_values_data_fd],
                    extra_title_label=final_title_label,
                    legend_labels=["GardnerTimeRec", "FDTimeRec"],
                    save_plot=True,
                    directory_to_save_images=folder_to_store_images,
                    base_string_for_saving_image=f"{key}_vs_rop",
                    alternative_plot="min"
                )
