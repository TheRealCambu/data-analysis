import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from commun_utils.theoretical_formulas import theoretical_evm_from_ber
from commun_utils.utils import apply_plt_personal_settings


def merge_lists(series):
    combined = []
    for s in series:
        if s is None or (isinstance(s, float) and np.isnan(s)):
            continue
        if isinstance(s, str):
            s = s.strip()
            if s in ("[]", ""):
                continue
            s = [float(x) for x in s.replace('[', '').replace(']', '').split()]
        elif isinstance(s, (list, np.ndarray)):
            s = list(s)
        else:
            continue
        combined.extend(s)
    return combined


def get_df_slice(current_sweep, groups, sweep_columns_dict_keys, bits_per_symbol, cr_algo_label):
    # Get the group using bits per symbol and the clock recovery algorithm
    current_df = groups.get_group((bits_per_symbol, cr_algo_label)).copy().reset_index(drop=True)

    target_cols = [k for k in sweep_columns_dict_keys if k != current_sweep]
    same_value_rows = current_df[current_df.duplicated(subset=target_cols, keep=False)].copy()
    rows_to_join_mask = same_value_rows.duplicated(subset=sweep_columns_dict_keys, keep=False)
    rows_to_join = same_value_rows[rows_to_join_mask].copy()
    rows_to_keep = same_value_rows[~rows_to_join_mask].copy()

    # Detect merge columns automatically
    merge_cols = [col for col in rows_to_join.columns if 'EVM' in col or 'ber' in col]

    # The rest are considered grouping (unique identifier) columns
    group_cols = [col for col in rows_to_join.columns if col not in merge_cols]

    # Group by the unique columns and merge the numeric list columns
    merged_df = rows_to_join.groupby(group_cols, as_index=False).agg({col: merge_lists for col in merge_cols})

    # Concatenate the merged rows with the rows to keep
    final_df = pd.concat([merged_df, rows_to_keep], ignore_index=True)

    return final_df


csvs_folder = (r"C:\Users\39338\Politecnico Di Torino Studenti Dropbox\Simone Cambursano\Politecnico"
               r"\Tesi\Data-analysis\Simulation Sweeps\CR Sweeps\Second Batch (much more data, used the lab pc)")

# === Define output path ===
save_dir = (r"C:\Users\39338\Politecnico Di Torino Studenti Dropbox\Simone Cambursano"
            r"\Politecnico\Tesi\Data-analysis\Simulation Sweeps\CR Sweeps\Final Plots")

os.makedirs(save_dir, exist_ok=True)

# === Load CSV ===
csv_path_gardner_fd = os.path.join(csvs_folder, "gardner_and_fd_cr_simulation_results.csv")
df_gardner_fd = pd.read_csv(csv_path_gardner_fd)

csv_path_godard = os.path.join(csvs_folder, "godard_cr_simulation_results_no_sps_sweep.csv")
df_godard = pd.read_csv(csv_path_godard)

# Add the sps column in the gardner and fd
insert_loc_for_sps_column = df_gardner_fd.columns.get_loc("clock_recovery_algo") + 1
df_gardner_fd.insert(insert_loc_for_sps_column, "sps_cr", [2.0] * len(df_gardner_fd))

# Add the clock recovery algorithm in the godard
insert_loc_for_clock_recovery_algo_column = df_godard.columns.get_loc("samp_delay") + 1
df_godard.insert(insert_loc_for_clock_recovery_algo_column, "clock_recovery_algo", ["godard"] * len(df_godard))

df = pd.merge(df_gardner_fd, df_godard, how="outer")

symbol_rate = np.unique(df['symbol_rate'])
assert len(symbol_rate) == 1
symbol_rate = symbol_rate.item() / 1e9

# Drop the symbol rate columns since it is unique
df.drop(['symbol_rate', 'repeat'], axis=1, inplace=True)

y_values_columns = {
    "berTot": {"title": "BER", "ylabel": "BER"},
    "EVMTot": {"title": "EVM", "ylabel": "EVM [%]"},
}

# sweep_columns_dict = {
#     "rolloff": {"idx": [0, 9], "xlabel": r'$\beta$', "uom": ""},
#     "jitter_amp": {"idx": [9, 20], "xlabel": r'$\mathrm{jitter}_{\mathrm{amp}}$', "uom": "symbols"},
#     "jitter_df": {"idx": [20, 27], "xlabel": r'$\mathrm{jitter}_{\mathrm{df}}$', "uom": "MHz"},
#     "freq_off_ppm": {"idx": [27, 36], "xlabel": r'$\Delta f$', "uom": "ppm"},
#     "samp_delay": {"idx": [36, 47], "xlabel": r'$\Delta_{\mathrm{sym}}$', "uom": "symbols"}
# }

sweep_columns_dict = {
    "rolloff": {
        "idx": [0, 9],
        "xlabel": r"$\beta$",
        "uom": ""
    },
    "jitter_amp": {
        "idx": [9, 20],
        "xlabel": r"$A_{\mathrm{pp}}$",
        "uom": "symbols"
    },
    "jitter_df": {
        "idx": [20, 27],
        "xlabel": r"$f_{\mathrm{jitter}}$",
        "uom": "MHz"
    },
    "freq_off_ppm": {
        "idx": [27, 36],
        "xlabel": r"$\mathrm{SFO}_{\mathrm{ppm}}$",
        "uom": "ppm"
    },
    "samp_delay": {
        "idx": [36, 47],
        "xlabel": r"$\Delta_{\mathrm{sym}}$",
        "uom": "symbols"
    }
}

sweep_columns_dict_keys = sweep_columns_dict.keys()

mod_order = {
    "DP-QPSK": 2,
    "DP-16QAM": 4
}

cr_algo = {
    "gardner": "Gardner",
    "fd": "Fast square-timing",
    "godard": "Godard"
}

plot_fec_th_line = False
ber_fec_threshold = 2.01e-2
evm_fec_threshold = {
    2: theoretical_evm_from_ber(ber_fec_threshold, 4),
    4: theoretical_evm_from_ber(ber_fec_threshold, 16)
}

markers = ['o', 's', 'v', '^', 'd', 'x']
colors = plt.cm.tab10(np.linspace(0, 1, len(cr_algo)))

# Group by modulation order and clock recovery algorithm
groups = df.groupby(['mod_order', 'clock_recovery_algo'])

# Apply personal matplotlib settings
apply_plt_personal_settings()

for modulation_label, bits_per_symbol in mod_order.items():
    for sweep_key, sweep_info in sweep_columns_dict.items():
        xlabel = sweep_info['xlabel']
        start_idx, stop_idx = sweep_info["idx"]

        for y_value_key, plot_dict in y_values_columns.items():
            is_ber = 'ber' in y_value_key.lower()
            scale = 1 if is_ber else 100
            plt.figure()
            for idx, (cr_algo_label, cr_label_for_plot) in enumerate(cr_algo.items()):
                # print(f"Bits per symbol: {bits_per_symbol} -- Algorithm: {cr_algo_label}")

                # Plot each dataset
                marker = markers[idx % len(markers)]
                color = colors[idx]

                df_slice = get_df_slice(
                    current_sweep=sweep_key,
                    groups=groups,
                    sweep_columns_dict_keys=sweep_columns_dict_keys,
                    bits_per_symbol=bits_per_symbol,
                    cr_algo_label=cr_algo_label
                )

                all_keys_but_current = [k for k in sweep_columns_dict_keys if k != sweep_key]
                substring_for_title = ", ".join(
                    f"{sweep_columns_dict[k]['xlabel']}="
                    f"{(np.unique(df_slice[k])[0] / (1e6 if k == 'jitter_df' else 1.0)):.1f}"
                    f"{'MHz' if k == 'jitter_df' else ''}"
                    for k in all_keys_but_current
                )

                if sweep_key == "jitter_df":
                    x_values = df_slice[sweep_key] / 1e6
                else:
                    x_values = df_slice[sweep_key]
                plt.xlim(left=np.min(x_values) * 0.94, right=np.max(x_values) * 1.02)

                # Convert string arrays to numeric and take mean, min, and max per row
                y_array = df_slice[y_value_key].apply(
                    lambda s: np.array(s if isinstance(s, (list, np.ndarray)) else np.fromstring(s.strip('[]'), sep=' '))
                )
                y_mean = y_array.apply(np.mean)
                y_min = y_array.apply(np.min)
                y_max = y_array.apply(np.max)

                threshold = ber_fec_threshold if is_ber else evm_fec_threshold[bits_per_symbol]
                mask = (y_max <= threshold)

                # Apply mask consistently to all arrays to avoid misalignment warnings
                x_filtered = x_values[mask]
                y_mean_filtered = y_mean[mask] * scale
                y_min_filtered = y_min[mask] * scale
                y_max_filtered = y_max[mask] * scale

                # Plot
                plt.plot(
                    x_filtered, y_mean_filtered, marker + '-', color=color, label=cr_label_for_plot
                )
                plt.fill_between(
                    x_filtered, y_min_filtered, y_max_filtered, alpha=0.3, color=color
                )

                # Set scale: BER → semi logy; EVM → linear
                plt.yscale('log' if is_ber else 'linear')

            if plot_fec_th_line:
                if 'ber' in y_value_key:
                    plt.axhline(
                        ber_fec_threshold, color='darkred', linestyle=':', linewidth=2.5,
                        label=f"FEC threshold = {ber_fec_threshold:.0e}"
                    )
                else:
                    plt.axhline(
                        evm_fec_threshold[bits_per_symbol] * scale, color='darkred', linestyle=':', linewidth=2.5,
                        label=f"FEC threshold = {evm_fec_threshold[bits_per_symbol] * scale:.2f} %"
                    )
            extra_xlabel = '' if sweep_info['uom'] == '' else f" [{sweep_info['uom']}]"
            plt.xlabel(xlabel + extra_xlabel)
            plt.ylabel(plot_dict['ylabel'])
            plt.title(f"{plot_dict['title']} vs {xlabel} ({modulation_label}@{symbol_rate:.0f}Gbaud)"
                      f"\n{substring_for_title}")
            plt.grid(True, which='both')
            plt.legend(loc='best')
            plt.tight_layout()
            # plt.show()

            # === Save figure ===
            filename = (f"{modulation_label.replace(' ', '_')}_"
                        f"{sweep_key.replace(' ', '_')}_{y_value_key.replace(' ', '_')}.png")
            save_path = os.path.join(save_dir, filename)
            plt.tight_layout()
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"Saved plot to: {save_path}")
