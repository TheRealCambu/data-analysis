import os

import numpy as np

from plot_results_fo_sweep import plot_results_fo
from plot_results_rop_sweep import plot_results_rop

root_folder = r"C:\Users\39338\Politecnico Di Torino Studenti Dropbox\Simone Cambursano\Politecnico\Tesi\Codes for Analyzing data\Lab results\v4 - Processed Datasets -- Final OPT"

# tr_algo_list = ["Gardner", "Frequency Domain"]
# baud_rate_and_mod_format_list = ["30GBd QPSK", "34.28GBd QPSK", "40GBd QPSK", "30GBd 16QAM", "34.28GBd 16QAM"]

tr_algo_list = ["Gardner"]
# tr_algo_list = ["Frequency Domain"]
# baud_rate_and_mod_format_list = ["30GBd QPSK"]
# baud_rate_and_mod_format_list = ["34.28GBd QPSK"]
baud_rate_and_mod_format_list = ["30GBd 16QAM"]
# baud_rate_and_mod_format_list = ["34.28GBd 16QAM"]

# baud_rate_and_mod_format_list = ["40GBd QPSK"]

plot_type_wanted = "fo"
folder_to_store_images = root_folder + rf"\{tr_algo_list[0]}\{plot_type_wanted.upper()} Plots"

plot_dispatch = {
    "fo": plot_results_fo,
    "rop": plot_results_rop,
    "osnr": None
}

# Once we have iterated in each folder, we have the .npz files
for tr_algo in tr_algo_list:
    for baud_rate_and_mod_format in baud_rate_and_mod_format_list:
        folder_path = os.path.join(root_folder, tr_algo, baud_rate_and_mod_format)
        files_in_current_folder = [f for f in os.listdir(folder_path) if f.endswith(".npz")]
        print(f"\n--- {tr_algo} / {baud_rate_and_mod_format} ---")
        for npz_file in files_in_current_folder:
            if plot_type_wanted in npz_file:
                full_path = os.path.join(folder_path, npz_file)
                with np.load(full_path, allow_pickle=True) as current_npz:
                    data_dict = dict(current_npz)
                    print(f"File: {npz_file}, Keys: {list(current_npz.files)}")
                    func = plot_dispatch.get(plot_type_wanted)
                    if func is None:
                        raise ValueError(f"Plot type {plot_type_wanted} not supported")
                    func(
                        data_dict=data_dict,
                        filename=npz_file,
                        directory_to_save_images=folder_to_store_images
                    )
