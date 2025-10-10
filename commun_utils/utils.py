from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np


def apply_plt_personal_settings():
    # Axes labels and title
    plt.rc('axes', labelsize=17, titlesize=17, grid=True)

    # Ticks
    plt.rc('xtick', labelsize=13, direction='in', top=True)
    plt.rc('ytick', labelsize=13, direction='in', right=True)

    # Legend
    plt.rc('legend', fontsize=14)

    # Grid style
    plt.rc('grid', linestyle='--', linewidth=0.4, color='gray')

    # Figure size
    plt.rc('figure', figsize=(10, 6.5))


def filter_outliers(
        upper_threshold: float,
        input_values: Union[List, np.ndarray],
        lower_threshold: float = 1e-30,
):
    return [
        [
            val if lower_threshold <= val <= upper_threshold else np.nan
            for val in per_x_values
        ]
        for per_x_values in input_values
    ]

