import itertools
import os

import matplotlib.pyplot as plt
import numpy as np

from commun_utils.theoretical_formulas import (
    theoretical_evm_from_ber
)
from commun_utils.utils import apply_plt_personal_settings

mod_format_vect = ["QPSK", "16QAM"]
# mod_format_vect = ["16QAM"]
pll_active_vect = ["True", "False"]
bps_bypassed_vect = ["True", "False"]
# PLL Active = False -- BPS bypassed = False ==> OK for QPSK and 16QAM
# PLL Active = False -- BPS bypassed = True ==> NOT OK SO SKIP IT
# PLL Active = True -- BPS bypassed = False ==> OK for QPSK and 16QAM
# PLL Active = True -- BPS bypassed = True ==> NOT OK FOR QPSK
linewidth_vect = ["100", "200", "300", "400", "500", "600", "700", "800", "900", "1000"]

# Generate all combinations except ("False", "True")
valid_combos = [
    (pll, bps)
    for pll, bps in itertools.product(pll_active_vect, bps_bypassed_vect)
    if not (pll == "False" and bps == "True")
]

folder_path = (r"C:\Users\39338\Politecnico Di Torino Studenti Dropbox\Simone Cambursano"
               r"\Politecnico\Tesi\Data-analysis\Simulation Sweeps\Linewidth Sweeps\First Batch")

# Apply personal matplotlib settings
apply_plt_personal_settings()

ber_fec_threshold = 2e-2
# ber_filter_threshold = 0.26
ber_filter_threshold = ber_fec_threshold
# ber_filter_threshold = 0.254

y_values_columns = {
    "berTot": {"title": "BER", "ylabel": "BER"},
    "EVMTot": {"title": "EVM", "ylabel": "EVM [%]"},
    # "berEVMTot": {"title": "BER$_{\mathrm{EVM}}$", "ylabel": r"BER$_{\mathrm{EVM}}$"}
}

markers = ['o', 's', 'v', '^', 'd', 'x']
colors = plt.cm.tab10(np.linspace(0, 1, len(valid_combos)))

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

# Prepare a dictionary to store all data
all_data = {}
for mod_format in mod_format_vect:
    all_data[mod_format] = {}
    for plot_type in y_values_columns.keys():
        data_collector_to_plot = {}
        for pll_active, bps_bypassed in valid_combos:
            if pll_active == "True" and bps_bypassed == "True" and mod_format == "QPSK":
                continue
            current_key = f"{pll_active} - {bps_bypassed}"
            data_collector_to_plot[current_key] = {}
            load_npz_vect = []
            y_values_list = []
            delta_nu_list = []
            symbol_rate_list = []
            bits_per_symbol_list = []
            osnr_dB_list = []

            for linewidth in linewidth_vect:
                current_npz_file = (
                    f"results_{mod_format}_32.0GBaud_CRAlgo_gardner_PLLActive_{pll_active}_"
                    f"BPSBypassed_{bps_bypassed}_Linewidth_{linewidth}.00kHz.npz"
                )
                try:
                    current_path = os.path.join(folder_path, current_npz_file)
                    loaded_data = dict(np.load(current_path, allow_pickle=True))
                    print(loaded_data)
                    if plot_type == "berTot":
                        y_valueTheory = loaded_data['berTheory']
                    elif plot_type == "EVMTot":
                        y_valueTheory = loaded_data['EVMTheory']
                    else:
                        y_valueTheory = loaded_data['berEVMTheory']

                    # Collect data
                    delta_nu_list.append(loaded_data['cfg_channel'].item()['laser_phase_noise']['delta_nu'])
                    y_values_list.append(loaded_data[plot_type].flatten())
                    symbol_rate_list.append(loaded_data['cfg_common'].item()['symbol_rate'])
                    bits_per_symbol_list.append(loaded_data['cfg_common'].item()['bits_per_symbol'])
                    osnr_dB_list.append(loaded_data['OSNR_dB_vect'].item())

                    # Store the theoretical value
                    data_collector_to_plot[current_key]['y_valueTheory'] = y_valueTheory
                except Exception as e:
                    print(f"Error loading {current_npz_file}: {e}")
                    continue

            # Convert to arrays for plotting
            data_collector_to_plot[current_key]['delta_nu'] = np.array(delta_nu_list) / 1000  # kHz
            data_collector_to_plot[current_key]['y_values'] = np.array(y_values_list)
            final_symbol_rate = np.unique(symbol_rate_list)
            assert len(final_symbol_rate) == 1
            data_collector_to_plot[current_key]['symbol_rate'] = final_symbol_rate.item()

            final_bits_per_symbol = np.unique(bits_per_symbol_list)
            assert len(final_bits_per_symbol) == 1
            data_collector_to_plot[current_key]['bits_per_symbol'] = final_bits_per_symbol.item()

            final_osnr_dB = np.unique(osnr_dB_list)
            assert len(final_osnr_dB) == 1
            data_collector_to_plot[current_key]['OSNR_dB'] = final_osnr_dB.item()

        all_data[mod_format][plot_type] = data_collector_to_plot

# ================= Plotting =================
for mod_format in mod_format_vect:
    for plot_type, plot_type_dict in y_values_columns.items():
        is_ber = 'ber' in plot_type
        first_key = next(iter(all_data[mod_format][plot_type]))
        symbol_rate = all_data[mod_format][plot_type][first_key]['symbol_rate'] / 1e9
        bits_per_symbol = all_data[mod_format][plot_type][first_key]['bits_per_symbol']
        OSNR_dB = all_data[mod_format][plot_type][first_key]['OSNR_dB']
        cardinality = mod_format_dict[bits_per_symbol]["cardinality"]
        modulation_format = mod_format_dict[bits_per_symbol]["mod_format_string"]
        if is_ber:
            scale = 1
            fec_threshold = ber_fec_threshold
            fec_threshold_label = f"FEC threshold={fec_threshold:.0e}"
        else:
            scale = 100
            fec_threshold = theoretical_evm_from_ber(ber=ber_fec_threshold, M=cardinality) * scale
            fec_threshold_label = f"FEC threshold = {fec_threshold:.2f}%"
        theory_value = all_data[mod_format][plot_type][first_key]['y_valueTheory'] * scale
        plt.figure()
        x = all_data[mod_format][plot_type][first_key]['delta_nu']
        if mod_format == "16QAM":
            for idx, (key, data_dict) in enumerate(all_data[mod_format][plot_type].items()):
                marker = markers[idx % len(markers)]
                color = colors[idx]
                y = np.mean(data_dict['y_values'], axis=1) * scale
                if key == "True - True":
                    plot_label = "PLL"
                elif key == "True - False":
                    plot_label = "PLL + BPS"
                else:
                    plot_label = "BPS"
                plt.plot(x, y, marker + '-', color=color, label=plot_label)
        else:
            # Extract all y_values arrays (list of lists/arrays)
            y_values = [data['y_values'] for data in all_data[mod_format][plot_type].values()]
            # Stack all corresponding sublists along a new axis for direct vectorized operations
            y_stacked = np.stack(y_values)
            # Mean across configurations (axis=0)
            y_mean = np.mean(y_stacked, axis=0).mean(axis=1)  # second mean collapses repeats if needed
            # Plot
            plt.plot(x, y_mean * scale, markers[0] + '-', color=colors[0], label="BPS")

        plt.axhline(
            fec_threshold, color='darkred', linestyle=':',
            linewidth=2.5, label=fec_threshold_label
        )
        if is_ber:
            theory_line_label = f"Theoretical {plot_type_dict['title']}={theory_value[0]:.2e}"
        else:
            theory_line_label = f"Theoretical {plot_type_dict['title']}={theory_value[0]:.2f}%"
        plt.axhline(theory_value, color='darkblue', linestyle=':', linewidth=2.5, label=theory_line_label)
        x = all_data[mod_format][plot_type][first_key]['delta_nu']
        plt.xlim(left=np.min(x), right=np.max(x))
        plt.yscale("log" if is_ber else "linear")
        plt.xlabel(r'$\Delta \nu$ [kHz]')
        plt.ylabel(plot_type_dict['ylabel'])
        plt.title(f"{plot_type_dict['title']} vs Linewidth ({modulation_format}@{symbol_rate:.0f}GBaud)")
        plt.legend(loc="best")
        plt.grid(True, which="both")
        plt.tight_layout()
        # plt.show()
        filename = (
            f"Linewidth_Sweep_{mod_format}_{plot_type}.png"
        )
        save_path = os.path.join(
            r"C:\Users\39338\Politecnico Di Torino Studenti Dropbox\Simone Cambursano"
            r"\Politecnico\Tesi\Data-analysis\Simulation Sweeps\Linewidth Sweeps\Final Plots",
            filename
        )
        plt.savefig(save_path, dpi=400, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
