import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from commun_utils.utils import apply_plt_personal_settings
from commun_utils.theoretical_formulas import theoretical_evm_from_ber

# === Load CSV ===
csv_path = (r"C:\Users\39338\Politecnico Di Torino Studenti Dropbox\Simone Cambursano"
            r"\Politecnico\Tesi\Data-analysis\Simulation Sweeps\CR Sweeps"
            r"\Second Batch (much more data, used the lab pc)\gardner_and_fd_cr_simulation_results.csv")
df = pd.read_csv(csv_path)

# Get the symbol rate since it is unique
symbol_rate = df.symbol_rate[0]

# Drop the symbol rate columns since it is unique
df.drop(['symbol_rate', 'repeat'], axis=1, inplace=True)

y_values_columns = {
    "berTot": {"title": "BER", "ylabel": "BER"},
    "EVMTot": {"title": "EVM", "ylabel": "EVM [%]"},
}

sweep_columns_dict = {
    "rolloff": {"idx": [0, 9], "xlabel": r'$\beta$'},
    "jitter_amp": {"idx": [9, 20], "xlabel": r'$\mathrm{jitter}_{\mathrm{amp}}$'},
    "jitter_df": {"idx": [20, 27], "xlabel": r'$\mathrm{jitter}_{\mathrm{df}}$'},
    "freq_off_ppm": {"idx": [27, 36], "xlabel": r'$\Delta f_{\mathrm{ppm}}$'},
    "samp_delay": {"idx": [36, 47], "xlabel": r'$\mathrm{sampling\ delay}$'}
}

sweep_columns_dict_keys = sweep_columns_dict.keys()

mod_order = {
    "DP-QPSK": 2,
    "DP-16QAM": 4
}

cr_algo = {
    "gardner": "GardnerTimeRec",
    "fd": "FDTimeRec"
}

ber_fec_threshold = 2e-2
evm_fec_threshold = {
    2: theoretical_evm_from_ber(ber_fec_threshold, 4),
    4: theoretical_evm_from_ber(ber_fec_threshold, 16)
}

markers = ['o', 's', 'v', '^', 'd', 'x']
colors = plt.cm.tab10(np.linspace(0, 1, 2))

# Group by modulation order and clock recovery algorithm
groups = df.groupby(['mod_order', 'clock_recovery_algo'])

# Apply personal matplotlib settings
apply_plt_personal_settings()

for modulation_label, bits_per_symbol in mod_order.items():
    for sweep_key, sweep_info in sweep_columns_dict.items():
        xlabel = sweep_info['xlabel']
        start_idx = sweep_info["idx"][0]
        stop_idx = sweep_info["idx"][1]

        for y_value_key, plot_dict in y_values_columns.items():
            plt.figure()
            for idx, (cr_algo_label, cr_label_for_plot) in enumerate(cr_algo.items()):
                print(f"Bits per symbol: {bits_per_symbol} -- Algorithm: {cr_algo_label}")

                # Plot each dataset
                marker = markers[idx % len(markers)]
                color = colors[idx]

                current_df = groups.get_group((bits_per_symbol, cr_algo_label)).copy().reset_index(drop=True)
                df_slice = current_df.iloc[start_idx:stop_idx]
                all_keys_but_current = [key for key in sweep_columns_dict_keys if key != sweep_key]
                substring_for_title = ", ".join(
                    f"{sweep_columns_dict[key]['xlabel']}: {np.unique(df_slice[key])[0]}"
                    for key in all_keys_but_current
                )
                # Filter only rows where sweep column is not NaN
                x_values = df_slice[sweep_key]
                # plt.xticks(np.arange(np.min(x_values), np.max(x_values) + 1, 0.5))
                plt.xlim(left=np.min(x_values) * 0.94, right=np.max(x_values) * 1.02)
                # Convert string arrays to numeric and take mean per row
                y_mean = df_slice[y_value_key].apply(
                    lambda s: np.mean([float(x) for x in s.replace('[', '').replace(']', '').split()])
                )
                y_min = df_slice[y_value_key].apply(
                    lambda s: np.min([float(x) for x in s.replace('[', '').replace(']', '').split()])
                )
                y_max = df_slice[y_value_key].apply(
                    lambda s: np.max([float(x) for x in s.replace('[', '').replace(']', '').split()])
                )
                if 'ber' in y_value_key:
                    mask = (y_max <= ber_fec_threshold)
                    x_values_filtered = x_values[mask]
                    y_mean_filtered = y_mean[mask]
                    y_min_filtered = y_min[mask]
                    y_max_filtered = y_max[mask]
                    plt.plot(
                        x_values_filtered, y_mean_filtered, marker + '-',
                        color=color, label=cr_label_for_plot
                    )
                    plt.fill_between(
                        x_values_filtered, y_min_filtered,
                        y_max_filtered, alpha=0.3, color=color
                    )
                else:
                    mask = (y_max <= evm_fec_threshold[bits_per_symbol])
                    x_values_filtered = x_values[mask]
                    y_mean_filtered = y_mean[mask] * 100
                    y_min_filtered = y_min[mask] * 100
                    y_max_filtered = y_max[mask] * 100
                    plt.plot(
                        x_values_filtered, y_mean_filtered, marker + '-',
                        color=color, label=cr_label_for_plot
                    )
                    plt.fill_between(
                        x_values_filtered, y_min_filtered,
                        y_max_filtered, alpha=0.3, color=color
                    )
            if 'ber' in y_value_key:
                plt.axhline(
                    ber_fec_threshold, color='darkred', linestyle=':', linewidth=2.5,
                    label=f"FEC threshold = {ber_fec_threshold:.0e}"
                )
            else:
                plt.axhline(
                    evm_fec_threshold[bits_per_symbol] * 100, color='darkred', linestyle=':', linewidth=2.5,
                    label=f"FEC threshold = {evm_fec_threshold[bits_per_symbol] * 100:.2f} %"
                )
            plt.xlabel(xlabel)
            plt.ylabel(plot_dict['ylabel'])
            plt.title(f"{plot_dict['title']} vs {xlabel} ({modulation_label})\n{substring_for_title}")
            plt.grid(True, which='both')
            plt.legend(loc='best')
            plt.tight_layout()
            plt.show()

# for modulation_order_label, bits_per_symbol in mod_order.items():
#     for cr_algo_label, label_for_plot in cr_algo.items():
#         current_df = groups.get_group((bits_per_symbol, cr_algo_label)).copy()
#         current_df = current_df.reset_index(drop=True)
#         for current_sweep_key, idx_and_label_dict in sweep_columns_dict.items():
#             start_idx = idx_and_label_dict["idx"][0]
#             stop_idx = idx_and_label_dict["idx"][1]
#             xlabel = idx_and_label_dict["xlabel"]
#             df_slice = current_df.iloc[start_idx:stop_idx]
#             x_values = df_slice[current_sweep_key]
#             all_keys_but_current = [key for key in sweep_columns_dict_keys if key != current_sweep_key]
#             substring_for_title = ", ".join(
#                 f"{sweep_columns_dict[key]['xlabel']}: {np.unique(df_slice[key])[0]}"
#                 for key in all_keys_but_current
#             )
#             for y_value_key, plot_dict in y_values_columns.items():
#                 plt.figure()
#                 y_mean = df_slice[y_value_key].apply(
#                     lambda s: np.mean([float(x) for x in s.replace('[', '').replace(']', '').split()])
#                 )
#                 y_min = df_slice[y_value_key].apply(
#                     lambda s: np.min([float(x) for x in s.replace('[', '').replace(']', '').split()])
#                 )
#                 y_max = df_slice[y_value_key].apply(
#                     lambda s: np.max([float(x) for x in s.replace('[', '').replace(']', '').split()])
#                 )
#                 if 'ber' in y_value_key:
#                     plt.semilogy(x_values, y_mean)
#                     plt.fill_between(x_values, y_min, y_max, alpha=0.2)
#                 else:
#                     plt.plot(x_values, y_mean * 100)
#                     plt.fill_between(x_values, y_min * 100, y_max * 100, alpha=0.2)
#                 plt.xlabel(xlabel)
#                 plt.ylabel(plot_dict["ylabel"])
#                 plt.title(f"{plot_dict['title']} vs. {sweep_columns_dict[current_sweep_key]['xlabel']}"
#                           f"\n{substring_for_title}")
#                 plt.grid(True, which="both")
#                 plt.tight_layout()
#                 plt.show()


# y_values_columns = {
#     "berTot": {"title": "BER", "ylabel": "BER"},
#     "berX": {"title": "BER (X Pol)", "ylabel": "BER"},
#     "berY": {"title": "BER (Y Pol)", "ylabel": "BER"},
#     "EVMTot": {"title": "EVM", "ylabel": "EVM [%]"},
#     "EVMX": {"title": "EVM (X Pol)", "ylabel": "EVM [%]"},
#     "EVMY": {"title": "EVM (Y Pol)", "ylabel": "EVM [%]"},
#     "berEVMTot": {"title": "BER$_{\mathrm{EVM}}", "ylabel": "BER$_{\mathrm{EVM}}"},
#     "berEVMX": {"title": "BER$_{\mathrm{EVM}} (X Pol)", "ylabel": "BER$_{\mathrm{EVM}}"},
#     "berEVMY": {"title": "BER$_{\mathrm{EVM}} (Y Pol)", "ylabel": "BER$_{\mathrm{EVM}}"}
# }

# y_values_columns = {
#     "berTot": {"title": "BER", "ylabel": "BER"},
#     "berX": {"title": "BER (X Pol)", "ylabel": "BER"},
#     "berY": {"title": "BER (Y Pol)", "ylabel": "BER"},
#     "EVMTot": {"title": "EVM", "ylabel": "EVM [%]"},
#     "EVMX": {"title": "EVM (X Pol)", "ylabel": "EVM [%]"},
#     "EVMY": {"title": "EVM (Y Pol)", "ylabel": "EVM [%]"},
# }