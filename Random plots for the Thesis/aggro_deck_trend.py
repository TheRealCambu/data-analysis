# import numpy as np
# import matplotlib.pyplot as plt
# from commun_utils.utils import apply_plt_personal_settings
#
# # Apply personal matplotlib settings
# apply_plt_personal_settings()
#
# # Data
# years = np.linspace(2011, 2025, 15)
# percentage_aggro = np.array([49, 55, 43, 46, 50, 51, 52, 54, 50, 43, 49, 45, 47, 42, 46])
# percentage_control = np.array([23, 20, 29, 22, 18, 21, 22, 23, 25, 29, 23, 21, 20, 23, 21])
# percentage_combo = np.array([29, 25, 27, 32, 32, 28, 26, 23, 24, 28, 28, 34, 33, 35, 33])
# percentage_vector = [percentage_aggro, percentage_control, percentage_combo]
# percentage_labels = ["aggro", "control", "combo"]
# colors = ["blue", "red", "green"]
#
# # Future years to predict
# future_years = np.array([2026, 2027, 2028, 2029, 2030, 2031, 2032])
#
# plt.figure()
# for percentage_type, label, color in zip(percentage_vector, percentage_labels, colors):
#     coef = np.polyfit(years, percentage_type, 2)
#     model = np.poly1d(coef)
#
#     # Predictions
#     future_preds = model(future_years)
#
#     # Plot historical data
#     plt.plot(years, percentage_type, marker='o', color=color, label=f"Dati {label}")
#
#     # Plot predictions with darker color
#     plt.plot(
#         future_years, future_preds,
#         marker='x', markersize=12,
#         linestyle='none',
#         color=color,  # same base color…
#         label=f"Previsione {label} dal {future_years[0]} al {future_years[-1]}"
#     )
#
# # Final settings
# total_years = np.concatenate((years, future_years))
# plt.xlabel("Anno")
# plt.ylabel("Percentuale")
# plt.xticks(total_years, rotation=45)
# # plt.ylim([15, 65])
# plt.title(f"Deck in percentuale per anno dal {total_years[0]:.0f} al {total_years[-1]:.0f}, Format: Modern")
# plt.legend(loc="best", fontsize=12)
# plt.tight_layout()
# plt.grid(True)
# plt.show()

import numpy as np
import matplotlib.pyplot as plt
from commun_utils.utils import apply_plt_personal_settings

# Apply personal matplotlib settings
apply_plt_personal_settings()

# Data
years = np.linspace(2011, 2025, 15)
percentage_aggro = np.array([49, 55, 43, 46, 50, 51, 52, 54, 50, 43, 49, 45, 47, 42, 46])
percentage_control = np.array([23, 20, 29, 22, 18, 21, 22, 23, 25, 29, 23, 21, 20, 23, 21])
percentage_combo = np.array([29, 25, 27, 32, 32, 28, 26, 23, 24, 28, 28, 34, 33, 35, 33])
# percentage_aggro = np.array([49, 55, 43, 46, 50, 51, 52, 54, 50, 43, 49, 45, 47, 42])
# percentage_control = np.array([23, 20, 29, 22, 18, 21, 22, 23, 25, 29, 23, 21, 20, 23])
# percentage_combo = np.array([29, 25, 27, 32, 32, 28, 26, 23, 24, 28, 28, 34, 33, 35])

percentage_vector = [percentage_aggro, percentage_control, percentage_combo]
percentage_labels = ["aggro", "control", "combo"]
colors = ["blue", "red", "green"]

# Future years
future_years = np.array([2026, 2027, 2028])

# Store raw and normalized predictions
raw_preds = []

# Fit models and compute raw predictions
for percentage_type in percentage_vector:
    coef = np.polyfit(years, percentage_type, 4)
    model = np.poly1d(coef)
    raw_preds.append(model(future_years))

raw_preds = np.array(raw_preds)  # shape = (3, N_future)

# ---- NORMALIZATION TO 100% ---- #
norm_preds = 100 * raw_preds / np.sum(raw_preds, axis=0)

# Plot
plt.figure()
for idx, (percentage_type, label, color) in enumerate(zip(percentage_vector, percentage_labels, colors)):
    # Plot historical data
    plt.plot(years, percentage_type, marker='o', color=color, label=f"Dati {label}")

    # Plot normalized predictions
    plt.plot(
        future_years,
        norm_preds[idx],
        marker='x',
        markersize=12,
        linestyle='none',
        color=color,
        label=f"Previsione {label} ({future_years[0]}–{future_years[-1]})"
    )

# Final settings
total_years = np.concatenate((years, future_years))
plt.xlabel("Anno")
plt.ylabel("Percentuale (%)")
plt.ylim([13, 62])
plt.xticks(total_years, rotation=45)
plt.title(f"Deck in percentuale per anno dal {total_years[0]:.0f} al {total_years[-1]:.0f}, Format: Modern")
plt.legend(loc="best", fontsize=11)
plt.tight_layout()
plt.grid(True)
plt.show()
