import os
import re

import matplotlib.pyplot as plt
import numpy as np

from commun_utils.theoretical_formulas import (
    theoretical_ber_vs_snr,
    theoretical_evm_vs_osnr,
    theoretical_ber_from_evm,
    osnr_to_snr
)
from commun_utils.utils import apply_plt_personal_settings, filter_outliers


def plot_ber_vs_fo(
        ber_vect,
        ber_threshold: float,
        plot_ber_min_max: str,
        fec_threshold: float,
        filename: str,
        x_values_sorted_indices,
        x_values_data,
        extra_title_label: str,
        ber_theory,
        save_plot: bool,
        directory_to_save_images: str,
        base_string_for_saving_image: str
) -> None:
    if ber_threshold < 5e-1:
        filtered_ber_tot = filter_outliers(
            upper_threshold=ber_threshold,
            input_values=ber_vect[x_values_sorted_indices],
            lower_threshold=1e-10
        )
    else:
        filtered_ber_tot = ber_vect[x_values_sorted_indices]

    ber_mean = np.nanmean(filtered_ber_tot, axis=1)
    ber_min = np.nanmin(filtered_ber_tot, axis=1)
    ber_max = np.nanmax(filtered_ber_tot, axis=1)

    # Plot setup
    plt.figure()
    plt.semilogy(x_values_data, ber_mean, 'o-', label=f"Avg BER")
    if plot_ber_min_max is not None:
        if plot_ber_min_max == "type1":
            plt.semilogy(x_values_data, ber_min, 'x-', label=f"Min BER")
            plt.semilogy(x_values_data, ber_max, 'v-', label=f"Max BER")
        elif plot_ber_min_max == "type2":
            plt.fill_between(x_values_data, ber_min, ber_max, alpha=0.3, label="Min–Max BER")
        else:
            raise ValueError("Select between 'type1' and 'type2'!")
    plt.axhline(
        fec_threshold, color='darkred', linestyle=':',
        linewidth=2.5, label=f"FEC threshold = {fec_threshold:.0e}"
    )
    if ber_theory > 1e-30:
        plt.axhline(
            ber_theory, color='darkblue', linestyle=':', linewidth=2.5,
            label=f"Theoretical BER = {ber_theory:.2e}"
        )
    plt.xlabel("Frequency Offset [GHz]")
    plt.ylabel("BER")
    plt.title(f"BER vs Frequency Offset {extra_title_label}")
    plt.legend(loc="best")
    # plt.ylim([2.5e-3, 3e-2])
    plt.xticks(np.arange(np.min(x_values_data), np.max(x_values_data) + 0.5, 1))
    plt.grid(True, which="both")
    plt.tight_layout()

    if save_plot:
        image_name = filename.replace("PROCESSED_fo_sweep", "").replace(".npz", "")
        if plot_ber_min_max is not None:
            if plot_ber_min_max == "type1":
                image_name = "_avg_min_max" + image_name
            elif plot_ber_min_max == "type2":
                image_name = "_variance" + image_name
            else:
                raise ValueError("Select between 'type1' and 'type2'!")
        image_name = base_string_for_saving_image + image_name
        full_path = os.path.join(directory_to_save_images, image_name)
        plt.savefig(full_path)


def plot_evm_vs_fo(
        evm_vect,
        evm_threshold,
        plot_evm_min_max: str,
        filename: str,
        x_values_sorted_indices,
        x_values_data,
        extra_title_label: str,
        evm_theory,
        save_plot: bool,
        directory_to_save_images: str,
        base_string_for_saving_image: str
) -> None:
    if evm_threshold < 3.0:
        filtered_evm_tot = filter_outliers(
            upper_threshold=evm_threshold,
            input_values=evm_vect[x_values_sorted_indices]
        )
    else:
        filtered_evm_tot = evm_vect[x_values_sorted_indices]

    evm_mean = np.mean(filtered_evm_tot, axis=1) * 100
    evm_min = np.min(filtered_evm_tot, axis=1) * 100
    evm_max = np.max(filtered_evm_tot, axis=1) * 100

    # Plot setup
    plt.figure()
    plt.plot(x_values_data, evm_mean, 'o-', label=f"Avg EVM")
    if plot_evm_min_max is not None:
        if plot_evm_min_max == "type1":
            plt.plot(x_values_data, evm_min, 'x-', label=f"Min EVM")
            plt.plot(x_values_data, evm_max, 'v-', label=f"Max EVM")
        elif plot_evm_min_max == "type2":
            plt.fill_between(x_values_data, evm_min, evm_max, alpha=0.3, label="Min–Max EVM")
        else:
            raise ValueError("Select between 'type1' and 'type2'!")
    plt.axhline(
        evm_theory * 100, color='darkblue', linestyle=':', linewidth=2.5,
        label=f"Theoretical EVM = {evm_theory * 100:.3g}%"
    )
    # if evm_threshold < 2.5:
    #     if np.max(osnr_dB) > 20:
    #         plt.ylim([0, 115] if "16QAM" in filename else [8, 17])
    #     else:
    #         plt.ylim([10, 115] if "16QAM" in filename else [32.5, 46])
    plt.xlabel("Frequency Offset [GHz]")
    plt.ylabel(r"EVM$_{\mathrm{m}}$ [%]")
    plt.title(f"EVM vs Frequency Offset {extra_title_label}")
    plt.legend(loc="best")
    plt.xticks(np.arange(np.min(x_values_data), np.max(x_values_data) + 0.5, 1))
    plt.grid(True, which="both")
    plt.tight_layout()

    if save_plot:
        image_name = filename.replace("PROCESSED_fo_sweep", "").replace(".npz", "")
        if plot_evm_min_max is not None:
            if plot_evm_min_max == "type1":
                image_name = "_avg_min_max" + image_name
            elif plot_evm_min_max == "type2":
                image_name = "_variance" + image_name
            else:
                raise ValueError("Select between 'type1' and 'type2'!")
        image_name = base_string_for_saving_image + image_name
        full_path = os.path.join(directory_to_save_images, image_name)
        plt.savefig(full_path)


def plot_ber_evm_vs_fo(
        ber_evm_vect,
        ber_evm_threshold,
        plot_ber_evm_min_max: str,
        fec_threshold: float,
        filename: str,
        x_values_sorted_indices,
        x_values_data,
        extra_title_label: str,
        ber_evm_theory,
        save_plot: bool,
        directory_to_save_images: str,
        base_string_for_saving_image: str
) -> None:
    if ber_evm_threshold < 5e-1:
        filtered_ber_evm_tot = filter_outliers(
            upper_threshold=ber_evm_threshold,
            input_values=ber_evm_vect[x_values_sorted_indices]
        )
    else:
        filtered_ber_evm_tot = ber_evm_vect[x_values_sorted_indices]

    ber_evm_mean = np.mean(filtered_ber_evm_tot, axis=1)
    ber_evm_min = np.min(filtered_ber_evm_tot, axis=1)
    ber_evm_max = np.max(filtered_ber_evm_tot, axis=1)

    # Plot setup
    plt.figure()
    plt.semilogy(x_values_data, ber_evm_mean, 'o-', label=r"Avg BER$_{\mathrm{EVM}}$")
    if plot_ber_evm_min_max is not None:
        if plot_ber_evm_min_max == "type1":
            plt.semilogy(x_values_data, ber_evm_min, 'x-', label=r"Min BER$_{\mathrm{EVM}}$")
            plt.semilogy(x_values_data, ber_evm_max, 'v-', label=r"Max BER$_{\mathrm{EVM}}$")
        elif plot_ber_evm_min_max == "type2":
            plt.fill_between(
                x_values_data, ber_evm_min, ber_evm_max,
                alpha=0.3, label=r"Min–Max BER$_{\mathrm{EVM}}$"
            )
        else:
            raise ValueError("Select between 'type1' and 'type2'!")
    plt.axhline(
        fec_threshold, color='darkred', linestyle=':',
        linewidth=2.5, label=f"FEC threshold = {fec_threshold:.0e}"
    )
    if ber_evm_theory > 1e-30:
        plt.axhline(
            ber_evm_theory, color='darkblue', linestyle=':', linewidth=2.5,
            label=fr"Theoretical BER$_{{\mathrm{{EVM}}}}$ = {ber_evm_theory:.2e}"
        )
    plt.xlabel("Frequency Offset [GHz]")
    plt.ylabel(r"BER$_{\mathrm{EVM}}$")
    plt.title(fr"BER$_{{\mathrm{{EVM}}}}$ vs Frequency Offset {extra_title_label}")
    plt.legend(loc="best")
    plt.xticks(np.arange(np.min(x_values_data), np.max(x_values_data) + 0.5, 1))
    plt.grid(True, which="both")
    plt.tight_layout()

    if save_plot:
        image_name = filename.replace("PROCESSED_fo_sweep", "").replace(".npz", "")
        if plot_ber_evm_min_max is not None:
            if plot_ber_evm_min_max == "type1":
                image_name = "_avg_min_max" + image_name
            elif plot_ber_evm_min_max == "type2":
                image_name = "_variance" + image_name
            else:
                raise ValueError("Select between 'type1' and 'type2'!")
        image_name = base_string_for_saving_image + image_name
        full_path = os.path.join(directory_to_save_images, image_name)
        plt.savefig(full_path)


def plot_results_fo(
        data_dict: dict,
        filename: str,
        directory_to_save_images: str,

        fec_threshold: float = 2e-2,
        ber_filter_threshold: float = 5e-1,
        # ber_filter_threshold: float = 4e-1,

        evm_filter_threshold=4.0,
        # evm_filter_threshold=1.05,

        # plot_ber: bool = True,
        # save_plot_ber: bool = True,
        # plot_per_pol_ber: bool = True,
        # save_plot_per_pol_ber: bool = True,
        # plot_ber_min_max: str = "type1",

        plot_ber: bool = False,
        save_plot_ber: bool = True,
        plot_per_pol_ber: bool = False,
        save_plot_per_pol_ber: bool = True,
        plot_ber_min_max: str = "type2",

        # plot_evm: bool = True,
        # save_plot_evm: bool = True,
        # plot_per_pol_evm: bool = True,
        # save_plot_per_pol_evm: bool = True,
        # plot_evm_min_max: str = "type1",

        plot_evm: bool = False,
        save_plot_evm: bool = True,
        plot_per_pol_evm: bool = False,
        save_plot_per_pol_evm: bool = True,
        plot_evm_min_max: str = "type1",

        plot_ber_evm: bool = True,
        save_plot_ber_evm: bool = True,
        plot_per_pol_ber_evm: bool = True,
        save_plot_per_pol_ber_evm: bool = True,
        plot_ber_evm_min_max: str = "type2"

        # plot_ber_evm: bool = False,
        # save_plot_ber_evm: bool = True,
        # plot_per_pol_ber_evm: bool = False,
        # save_plot_per_pol_ber_evm: bool = True,
        # plot_ber_evm_min_max: str = "type1"
) -> None:
    osnr_dB = None
    # Extract the number before 'dB'
    match = re.search(r'_(\d+)dB_', filename)
    if match:
        osnr_dB = int(match.group(1))

    bits_per_symbol = 2 if "QPSK" in filename else 4
    const_cardinality = 4 if "QPSK" in filename else 16
    title_label_for_plot_tot = "(DP-QPSK)" if "QPSK" in filename else "(DP-16QAM)"
    title_label_for_plot_x = "(DP-QPSK, X Pol)" if "QPSK" in filename else "(DP-16QAM, X Pol)"
    title_label_for_plot_y = "(DP-QPSK, Y Pol)" if "QPSK" in filename else "(DP-16QAM, Y Pol)"

    if "osnr" not in data_dict.keys() and osnr_dB is not None:
        snr_lin = osnr_to_snr(osnr_dB, np.unique(data_dict['symbol_rate']))
        ber_theory = theoretical_ber_vs_snr(snr_lin, const_cardinality)[0]
        evm_theory = theoretical_evm_vs_osnr(
            bits_per_symbol=bits_per_symbol,
            M=const_cardinality,
            osnr=snr_lin,
        )[0]
        ber_evm_theory = theoretical_ber_from_evm(
            EVM_m=evm_theory,
            M=const_cardinality
        )
    else:
        snr_lin = osnr_to_snr(data_dict["osnr"], np.unique(data_dict['symbol_rate']))
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

    # Apply personal matplotlib settings
    apply_plt_personal_settings()
    fo = data_dict['fo']
    x_values_sorted_indices = np.argsort(fo)
    x_values_data = fo[x_values_sorted_indices]

    # Extract the data
    if "fo" not in data_dict.keys():
        raise ValueError("This function has been done to plot the frequency offset sweeps")

    if plot_ber:
        plot_ber_vs_fo(
            ber_vect=data_dict["ber_tot"],
            ber_threshold=ber_filter_threshold,
            plot_ber_min_max=plot_ber_min_max,
            fec_threshold=fec_threshold,
            filename=filename,
            x_values_sorted_indices=x_values_sorted_indices,
            x_values_data=x_values_data,
            extra_title_label=title_label_for_plot_tot,
            ber_theory=ber_theory,
            directory_to_save_images=directory_to_save_images,
            save_plot=save_plot_ber,
            base_string_for_saving_image="ber_tot_vs_fo"
        )

    if plot_per_pol_ber:
        plot_ber_vs_fo(
            ber_vect=data_dict["ber_x"],
            ber_threshold=ber_filter_threshold,
            plot_ber_min_max=plot_ber_min_max,
            fec_threshold=fec_threshold,
            filename=filename,
            x_values_sorted_indices=x_values_sorted_indices,
            x_values_data=x_values_data,
            extra_title_label=title_label_for_plot_x,
            ber_theory=ber_theory,
            directory_to_save_images=directory_to_save_images,
            save_plot=save_plot_per_pol_ber,
            base_string_for_saving_image="ber_x_vs_fo"
        )
        plot_ber_vs_fo(
            ber_vect=data_dict["ber_y"],
            ber_threshold=ber_filter_threshold,
            plot_ber_min_max=plot_ber_min_max,
            fec_threshold=fec_threshold,
            filename=filename,
            x_values_sorted_indices=x_values_sorted_indices,
            x_values_data=x_values_data,
            extra_title_label=title_label_for_plot_y,
            ber_theory=ber_theory,
            directory_to_save_images=directory_to_save_images,
            save_plot=save_plot_per_pol_ber,
            base_string_for_saving_image="ber_y_vs_fo"
        )

    if plot_evm:
        plot_evm_vs_fo(
            evm_vect=data_dict['evm_tot'],
            evm_threshold=evm_filter_threshold,
            plot_evm_min_max=plot_evm_min_max,
            filename=filename,
            x_values_sorted_indices=x_values_sorted_indices,
            x_values_data=x_values_data,
            extra_title_label=title_label_for_plot_tot,
            evm_theory=evm_theory,
            directory_to_save_images=directory_to_save_images,
            save_plot=save_plot_evm,
            base_string_for_saving_image="evm_tot_vs_fo"
        )

    if plot_per_pol_evm:
        plot_evm_vs_fo(
            evm_vect=data_dict['evm_x'],
            evm_threshold=evm_filter_threshold,
            plot_evm_min_max=plot_evm_min_max,
            filename=filename,
            x_values_sorted_indices=x_values_sorted_indices,
            x_values_data=x_values_data,
            extra_title_label=title_label_for_plot_x,
            evm_theory=evm_theory,
            directory_to_save_images=directory_to_save_images,
            save_plot=save_plot_per_pol_evm,
            base_string_for_saving_image="evm_x_vs_fo"
        )
        plot_evm_vs_fo(
            evm_vect=data_dict['evm_y'],
            evm_threshold=evm_filter_threshold,
            plot_evm_min_max=plot_evm_min_max,
            filename=filename,
            x_values_sorted_indices=x_values_sorted_indices,
            x_values_data=x_values_data,
            extra_title_label=title_label_for_plot_y,
            evm_theory=evm_theory,
            directory_to_save_images=directory_to_save_images,
            save_plot=save_plot_per_pol_evm,
            base_string_for_saving_image="evm_y_vs_fo"
        )

    if plot_ber_evm:
        plot_ber_evm_vs_fo(
            ber_evm_vect=data_dict['ber_evm_tot'],
            ber_evm_threshold=ber_filter_threshold,
            plot_ber_evm_min_max=plot_ber_evm_min_max,
            fec_threshold=fec_threshold,
            filename=filename,
            x_values_sorted_indices=x_values_sorted_indices,
            x_values_data=x_values_data,
            extra_title_label=title_label_for_plot_tot,
            ber_evm_theory=ber_evm_theory,
            directory_to_save_images=directory_to_save_images,
            save_plot=save_plot_ber_evm,
            base_string_for_saving_image="ber_evm_tot_vs_fo"
        )

    if plot_per_pol_ber_evm:
        plot_ber_evm_vs_fo(
            ber_evm_vect=data_dict['ber_evm_x'],
            ber_evm_threshold=ber_filter_threshold,
            plot_ber_evm_min_max=plot_ber_evm_min_max,
            fec_threshold=fec_threshold,
            filename=filename,
            x_values_sorted_indices=x_values_sorted_indices,
            x_values_data=x_values_data,
            extra_title_label=title_label_for_plot_x,
            ber_evm_theory=ber_evm_theory,
            directory_to_save_images=directory_to_save_images,
            save_plot=save_plot_per_pol_ber_evm,
            base_string_for_saving_image="ber_evm_x_vs_fo"
        )
        plot_ber_evm_vs_fo(
            ber_evm_vect=data_dict['ber_evm_y'],
            ber_evm_threshold=ber_filter_threshold,
            plot_ber_evm_min_max=plot_ber_evm_min_max,
            fec_threshold=fec_threshold,
            filename=filename,
            x_values_sorted_indices=x_values_sorted_indices,
            x_values_data=x_values_data,
            extra_title_label=title_label_for_plot_y,
            ber_evm_theory=ber_evm_theory,
            directory_to_save_images=directory_to_save_images,
            save_plot=save_plot_per_pol_ber_evm,
            base_string_for_saving_image="ber_evm_y_vs_fo"
        )
    #################################### SHOW ALL THE PLOTS #####################################
    # plt.show()
