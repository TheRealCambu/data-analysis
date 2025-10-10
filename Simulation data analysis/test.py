import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from commun_utils.utils import apply_plt_personal_settings

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
            for cr_algo_label, cr_label_for_plot in cr_algo.items():
                current_df = groups.get_group((bits_per_symbol, cr_algo_label)).copy().reset_index(drop=True)
                print(current_df.to_string())