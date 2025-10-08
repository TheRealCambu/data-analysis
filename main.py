import os
import re
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.interpolate import interp1d

from commun_utils.theoretical_formulas import (
    theoretical_evm_vs_osnr,
    theoretical_ber_vs_snr
)


def filter_outliers(threshold: float, input_values: List):
    return [
        [
            val if 1e-30 <= val <= threshold else np.nan for val in per_x_values_values
        ]
        for per_x_values_values in input_values
    ]


with open("process_and_plot_data_params.yaml") as file:
    plot_params = yaml.safe_load(file)

save_images = plot_params["save_images"]
plot_variance = plot_params["plot_variance"]
bits_per_symbol = plot_params["bits_per_symbol"]
symbol_rate_in_the_name = plot_params["symbol_rate_in_the_name"]
sweep_type = plot_params["sweep_type"]
plot_ber = plot_params["plot_ber"]
plot_ber_evm = plot_params["plot_ber_evm"]
plot_theo_ber = plot_params["plot_theo_ber"]
plot_theo_ber_evm = plot_params["plot_theo_ber_evm"]
plot_per_polarization = plot_params["plot_per_polarization"]
x_label = plot_params["x_label"]
y_label = plot_params["y_label"]
fec_threshold = plot_params["fec_threshold"]
filter_ber_threshold = plot_params["filter_ber_threshold"]
fec_label = f"FEC threshold = {fec_threshold:.0e}"
x_lim_upper = plot_params["x_lim_upper"]
x_lim_lower = plot_params["x_lim_lower"]
y_lim_upper = plot_params["y_lim_upper"]
y_lim_lower = plot_params["y_lim_lower"]
x_lim_vector = [x_lim_lower, x_lim_upper]
y_lim_vector = [y_lim_lower, y_lim_upper]
dataset_version = "v" + str(plot_params["version"])
eps = y_lim_lower

# Load data
constellation_cardinality = bits_per_symbol ** 2
mod_format_string = "QPSK" if bits_per_symbol == 2 else "16QAM"
files_basename = [
    # rf"PROCESSED_{sweep_type}_sweep_{symbol_rate_in_the_name}GBd_DP_{mod_format_string}_w_dpe_v1.npz",
    rf"PROCESSED_{sweep_type}_sweep_{symbol_rate_in_the_name}GBd_DP_{mod_format_string}_wo_dpe_v1.npz",
    # rf"PROCESSED_{sweep_type}_sweep_{symbol_rate_in_the_name}GBd_DP_{mod_format_string}_w_dpe_40dB_OSNR_v1.npz",
]

folder_processed_data = rf"D:\Repos\blind-coherent-receiver-python\Lab Measurements\Processed datasets\Gardner\{symbol_rate_in_the_name}GBd {mod_format_string}"
npz_processed_files = [os.path.join(folder_processed_data, filename) for filename in files_basename]

folder_to_store_images = rf"D:\blind-coherent-receiver-python\blind_dsp_lab\{dataset_version} - Plots\{symbol_rate_in_the_name}GBd {mod_format_string}"

####################### LOAD THE DATA ########################
# Load all data
loaded_data = [np.load(processed_data) for processed_data in npz_processed_files]
number_of_files = len(files_basename)

# Get symbol rate for each file
symbol_rate_vector = [np.unique(data["symbol_rate"])[0] for data in loaded_data]
same_rate = all(sr == symbol_rate_vector[0] for sr in symbol_rate_vector)

# Determine DPE status from filename
dpe_status_vector = []
for file in files_basename:
    if "w_dpe" in file:
        dpe_status_vector.append("DPE: on")
    elif "wo_dpe" in file:
        dpe_status_vector.append("DPE: off")
    else:
        dpe_status_vector.append("unknown")

same_dpe = all(status == dpe_status_vector[0] for status in dpe_status_vector)

# Initialize title and legend labels
legend_labels = ["" for _ in range(number_of_files)]
title_additional_str = ""

# Build title
if same_dpe and dpe_status_vector[0] != "unknown":
    title_additional_str += f"{dpe_status_vector[0]}"
else:
    for i, dpe in enumerate(dpe_status_vector):
        if dpe != "unknown":
            legend_labels[i] += dpe

if same_rate:
    title_additional_str += f", Symbol Rate: {symbol_rate_vector[0] / 1e9:.2f} GBd"
else:
    for i, symbol_rate in enumerate(symbol_rate_vector):
        if legend_labels[i]:
            legend_labels[i] += " - "
        legend_labels[i] += f"{symbol_rate / 1e9:.2f} GBd"

if sweep_type == "fo":
    s = files_basename[0]
    match = re.search(r'(\d+)dB_OSNR', s)
    if match:
        osnr = int(match.group(1))
        print("OSNR:", osnr)
    else:
        raise ValueError(f"OSNR not found in string: '{s}'")
    title_additional_str += f", OSNR: {osnr}dB"

# Plot setup
plt.figure(figsize=(8.5, 6))

# Generate a color vector using a colormap
start = 0.1
end = 0.9
cmap = plt.cm.viridis
number_of_intervals = np.linspace(start, end, number_of_files + 1)
intervals = [number_of_intervals[i:i + 2] for i in range(len(number_of_intervals) - 1)]

title_second_line = ""
if sweep_type == "osnr":
    # Compute OSNR range for theoretical curves (finer grain and extended lower bound)
    x_values_data_meas = loaded_data[0][sweep_type]
    x_values_sorted_indices = np.argsort(x_values_data_meas)
    x_values_data_meas = x_values_data_meas[x_values_sorted_indices]

    # Define finer and wider OSNR sweep for the theoretical curve
    osnr_min_theo = max(0.1, np.min(x_values_data_meas) - 0.5)  # Start 1 dB below the minimum, but not < 0
    osnr_max_theo = np.max(x_values_data_meas) + 0.5
    x_values_data_theo = np.linspace(osnr_min_theo, osnr_max_theo, 500)  # Finer resolution: 500 points

    # Linear OSNR → SNR conversion
    x_values_lin_vect = 10 ** (0.1 * x_values_data_theo)
    rx_bw_Hz = symbol_rate_vector[0] / 2
    snr_lin_dict = {
        "values": x_values_lin_vect * 12.5e9 / rx_bw_Hz / 2,
        "rate": symbol_rate_vector[0]
    }

    # Theoretical curves
    berTheory_dict = {
        "values": theoretical_ber_vs_snr(M=constellation_cardinality, snr=snr_lin_dict["values"]),
        "rate": snr_lin_dict["rate"]
    }
    EVMTheory = theoretical_evm_vs_osnr(
        bits_per_symbol=bits_per_symbol,
        M=constellation_cardinality,
        osnr=snr_lin_dict["values"]
    )
    berEVMTheory_dict = {
        "values": theoretical_ber_vs_evm(bits_per_symbol=bits_per_symbol, evm=EVMTheory),
        "rate": snr_lin_dict["rate"]
    }

    # Compute OSNR required to reach the FEC threshold from theoretical BER
    osnr_theo_at_fec = np.interp(
        fec_threshold,
        berTheory_dict["values"][::-1],
        x_values_data_theo[::-1]
    )

    # Compute OSNR penalty for each dataset (where BER is available)
    osnr_penalties = []
    for data in loaded_data:
        x_values_sorted_indices = np.argsort(data[sweep_type])
        x_values_data = data[sweep_type][x_values_sorted_indices]
        filtered_ber_tot = filter_outliers(filter_ber_threshold, data["ber_tot"][x_values_sorted_indices])
        ber_mean = np.nanmean(filtered_ber_tot, axis=1)
        interp_fn = interp1d(ber_mean[::-1], x_values_data[::-1], bounds_error=False)
        osnr_meas_at_fec = interp_fn(fec_threshold)
        penalty = osnr_meas_at_fec - osnr_theo_at_fec
        osnr_penalties.append(penalty)

    # Compose second line of title
    penalty_strs = [
        f"{label}= {penalty:.2f} dB" if not np.isnan(penalty) else f"{label}: N/A"
        for label, penalty in zip(legend_labels, osnr_penalties)
    ]
    title_second_line = ", OSNR penalty @ FEC threshold " + ", ".join(penalty_strs)
    x_lim_vector = [x_values_data_theo[0], x_lim_upper]

# Plotting
for i, (data, legend_label, interval) in enumerate(zip(loaded_data, legend_labels, intervals)):
    x_values_sorted_indices = np.argsort(data[sweep_type])
    x_values_data = data[sweep_type][x_values_sorted_indices]

    # Select colors from the colormap
    color_a = cmap(i / len(intervals))
    color_b = cmap((i + 0.5) / len(intervals))

    # Plot BER
    if plot_ber:
        filtered_ber_tot = filter_outliers(filter_ber_threshold, data["ber_tot"][x_values_sorted_indices])
        ber_mean = np.nanmean(filtered_ber_tot, axis=1)
        ber_variance = np.nanvar(filtered_ber_tot, axis=1)

        # Clip mean and variance to avoid log/semilogy problems
        ber_mean = np.clip(ber_mean, 1e-15, None)
        ber_variance = np.clip(ber_variance, 0, None)
        valid_mask = (ber_mean > 0) & np.isfinite(ber_variance)  # Mask invalid points (NaNs, zeros)

        plt.semilogy(
            x_values_data[valid_mask],
            ber_mean[valid_mask],
            'o-',
            color=color_a,
            label=f"BER {legend_label}"
        )

        if np.any(valid_mask) and plot_variance:
            var_over_mean_sq = ber_variance[valid_mask] / (ber_mean[valid_mask] ** 2)
            var_over_mean_sq = np.maximum(var_over_mean_sq, 0)  # protect against rounding errors
            ber_std_factor = np.power(10, np.sqrt(np.log10(1 + var_over_mean_sq)))
            lower_bound = ber_mean[valid_mask] / ber_std_factor
            upper_bound = ber_mean[valid_mask] * ber_std_factor
            plt.fill_between(
                x_values_data[valid_mask],
                lower_bound,
                upper_bound,
                color=color_a,
                alpha=0.3,
                label=f"±1 σ BER {legend_label}"
            )

    # Plot BER EVM
    if plot_ber_evm:
        filtered_ber_evm_tot = filter_outliers(filter_ber_threshold, data["ber_evm_tot"][x_values_sorted_indices])
        ber_evm_mean = np.nanmean(filtered_ber_evm_tot, axis=1)
        ber_evm_variance = np.nanvar(filtered_ber_evm_tot, axis=1)

        # Clip mean and variance to avoid log/semilogy problems
        ber_evm_mean = np.clip(ber_evm_mean, 1e-15, None)
        ber_evm_variance = np.clip(ber_evm_variance, 0, None)
        valid_mask = (ber_evm_mean > 0) & np.isfinite(ber_evm_variance)  # Mask invalid points

        plt.semilogy(
            x_values_data[valid_mask],
            ber_evm_mean[valid_mask],
            's--',
            color=color_b,
            label=f"BER EVM {legend_label}"
        )

        if np.any(valid_mask) and plot_variance:
            var_over_mean_sq = ber_evm_variance[valid_mask] / (ber_evm_mean[valid_mask] ** 2)
            var_over_mean_sq = np.maximum(var_over_mean_sq, 0)
            ber_std_factor = np.power(10, np.sqrt(np.log10(1 + var_over_mean_sq)))
            lower_bound = ber_evm_mean[valid_mask] / ber_std_factor
            upper_bound = ber_evm_mean[valid_mask] * ber_std_factor
            plt.fill_between(
                x_values_data[valid_mask],
                lower_bound,
                upper_bound,
                color=color_b,
                alpha=0.3,
                label=f"±1 σ BER EVM {legend_label}"
            )

# Theoretical curves if OSNR sweep
if sweep_type == "osnr":
    if plot_theo_ber:
        plt.semilogy(
            x_values_data_theo,
            berTheory_dict["values"],
            linestyle='-',
            color="black",
            linewidth=2,
            label=f"BER theoretical @{berTheory_dict['rate'] / 1e9:.2f}GBd"
        )
    if plot_theo_ber_evm:
        plt.semilogy(
            x_values_data_theo,
            berEVMTheory_dict["values"],
            linestyle='--',
            color="violet",
            linewidth=2,
            label=f"BER EVM theoretical @{berEVMTheory_dict['rate'] / 1e9:.2f}GBd"
        )

# Plot FEC Threshold
plt.axhline(fec_threshold, color='darkred', linestyle=':', linewidth=2.5, label=fec_label)

# Final plot adjustments
plt.xlabel(x_label, fontsize=14)
plt.ylabel(y_label, fontsize=14)
plt.title(
    f"Dual Polarization BER vs {sweep_type.upper()} \n{title_additional_str}{title_second_line}",
    fontsize=13
)
plt.xlim(x_lim_vector)
plt.ylim(y_lim_vector)
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.legend(fontsize=11, loc="best")
plt.tight_layout()

if save_images:
    image_name_with_time_stamp = add_timestamp_to_filename(
        f"TOTAL_{sweep_type}_sweep_{symbol_rate_in_the_name}GBd_DP_{mod_format_string}.png"
    )
    full_path = os.path.join(folder_to_store_images, image_name_with_time_stamp)
    plt.savefig(full_path)

###################################################################################################################

if plot_per_polarization:
    # --- X/Y Polarization Subplots ---
    fig, axs = plt.subplots(1, 2, figsize=(13, 7))
    fig.suptitle(f"X and Y Polarization BER vs OSNR {title_additional_str}", fontsize=16)

    # Plotting
    for i, (data, legend_label, interval) in enumerate(zip(loaded_data, legend_labels, intervals)):
        x_values_sorted_indices = np.argsort(data[sweep_type])
        x_values_data = data[sweep_type][x_values_sorted_indices]

        # Select colors from the colormap
        color_a = cmap(i / len(intervals))
        color_b = cmap((i + 0.5) / len(intervals))

        # --- Left: X Polarization ---
        if plot_ber:
            filtered_ber_x = filter_outliers(filter_ber_threshold, data["ber_x"][x_values_sorted_indices])
            ber_x_mean = np.nanmean(filtered_ber_x, axis=1)
            ber_x_variance = np.nanvar(filtered_ber_x, axis=1)
            axs[0].semilogy(x_values_data, ber_x_mean, 'o-', color=color_a, label=f"BER X {legend_label}")
            if len(filtered_ber_x) and plot_variance:
                axs[0].fill_between(
                    x_values_data,
                    ber_x_mean - np.sqrt(ber_x_variance),
                    ber_x_mean + np.sqrt(ber_x_variance),
                    color=color_a,
                    alpha=0.2,
                    label=f"±1 σ BER X {legend_label}"
                )

        if plot_ber_evm:
            filtered_ber_evm_x = filter_outliers(filter_ber_threshold, data["ber_evm_x"][x_values_sorted_indices])
            ber_evm_x_mean = np.nanmean(filtered_ber_evm_x, axis=1)
            ber_evm_x_variance = np.nanvar(filtered_ber_evm_x, axis=1)
            axs[0].semilogy(x_values_data, ber_evm_x_mean, 's--', color=color_b, label=f"BER EVM X {legend_label}")
            if len(filtered_ber_evm_x) and plot_variance:
                axs[0].fill_between(
                    x_values_data,
                    ber_evm_x_mean - np.sqrt(ber_evm_x_variance),
                    ber_evm_x_mean + np.sqrt(ber_evm_x_variance),
                    color=color_b,
                    alpha=0.2,
                    label=f"±1 σ BER EVM X {legend_label}"
                )

        # --- Right: Y Polarization ---
        if plot_ber:
            filtered_ber_y = filter_outliers(filter_ber_threshold, data["ber_y"][x_values_sorted_indices])
            ber_y_mean = np.nanmean(filtered_ber_y, axis=1)
            ber_y_variance = np.nanvar(filtered_ber_y, axis=1)
            axs[1].semilogy(x_values_data, ber_y_mean, 'o-', color=color_a, label=f"BER Y {legend_label}")
            if len(filtered_ber_y) and plot_variance:
                axs[1].fill_between(
                    x_values_data,
                    ber_y_mean - np.sqrt(ber_y_variance),
                    ber_y_mean + np.sqrt(ber_y_variance),
                    color=color_a,
                    alpha=0.2,
                    label=f"±1 σ BER Y {legend_label}"
                )

        if plot_ber_evm:
            filtered_ber_evm_y = filter_outliers(filter_ber_threshold, data["ber_evm_y"][x_values_sorted_indices])
            ber_evm_y_mean = np.nanmean(filtered_ber_evm_y, axis=1)
            ber_evm_y_variance = np.nanvar(filtered_ber_evm_y, axis=1)
            axs[1].semilogy(x_values_data, ber_evm_y_mean, 's--', color=color_b, label=f"BER EVM Y {legend_label}")
            if len(filtered_ber_evm_y) and plot_variance:
                axs[1].fill_between(
                    x_values_data,
                    ber_evm_y_mean - np.sqrt(ber_evm_y_variance),
                    ber_evm_y_mean + np.sqrt(ber_evm_y_variance),
                    color=color_b,
                    alpha=0.2,
                    label=f"±1 σ BER EVM Y {legend_label}"
                )

    if sweep_type == "osnr":
        # Plot theoretical curves
        if plot_theo_ber:
            axs[0].semilogy(
                x_values_data_theo,
                berTheory_dict["values"],
                linestyle='-',
                color="black",
                linewidth=2,
                label=f"BER theoretical @{berTheory_dict['rate'] / 1e9:.2f}GBd"
            )
            axs[1].semilogy(
                x_values_data_theo,
                berTheory_dict["values"],
                linestyle='-',
                color="black",
                linewidth=2,
                label=f"BER theoretical @{berTheory_dict['rate'] / 1e9:.2f}GBd"
            )
        if plot_theo_ber_evm:
            axs[0].semilogy(
                x_values_data_theo,
                berEVMTheory_dict["values"],
                linestyle='--',
                color="violet",
                linewidth=2,
                label=f"BER EVM theoretical @{berEVMTheory_dict['rate'] / 1e9:.2f}GBd"
            )
            axs[1].semilogy(
                x_values_data_theo,
                berEVMTheory_dict["values"],
                linestyle='--',
                color="violet",
                linewidth=2,
                label=f"BER EVM theoretical @{berEVMTheory_dict['rate'] / 1e9:.2f}GBd"
            )

        # Plot FEC Threshold
        axs[0].axhline(fec_threshold, color='darkred', linestyle=':', linewidth=2.5, label=fec_label)
        axs[1].axhline(fec_threshold, color='darkred', linestyle=':', linewidth=2.5, label=fec_label)

    axs[0].set_title("X Polarization", fontsize=14)
    axs[0].set_xlim(x_lim_vector)
    axs[0].set_ylim(y_lim_vector)
    axs[0].grid(True, which="both", linestyle="--", linewidth=0.4)
    axs[0].legend(fontsize=10, loc="best")
    axs[0].set_xlabel(x_label, fontsize=13)
    axs[0].set_ylabel(y_label, fontsize=13)

    axs[1].set_title("Y Polarization", fontsize=14)
    axs[1].set_xlim(x_lim_vector)
    axs[1].set_ylim(y_lim_vector)
    axs[1].grid(True, which="both", linestyle="--", linewidth=0.4)
    axs[1].legend(fontsize=10, loc="best")
    axs[1].set_xlabel(x_label, fontsize=13)

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))

    if save_images:
        image_name_with_time_stamp = add_timestamp_to_filename(
            f"XY_x_values_sweep_{symbol_rate_in_the_name}GBd_DP_{mod_format_string}.png"
        )
        full_path = os.path.join(folder_to_store_images, image_name_with_time_stamp)
        plt.savefig(full_path)

######################################## SHOW ALL THE PLOTS ########################################
plt.show()
