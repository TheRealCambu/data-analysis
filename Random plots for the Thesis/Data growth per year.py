import matplotlib.pyplot as plt

from commun_utils.utils import apply_plt_personal_settings

# Apply personal matplotlib settings
apply_plt_personal_settings()

# Years
years = [
    "2011", "2012", "2013", "2014", "2015", "2016", "2017",
    "2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025*"
]

# Change over previous year (%)
change_percent = [
    150, 30, 38.46, 38.89, 24, 16.13, 44.44,
    26.92, 24.24, 56.59, 23.05, 22.78, 23.71, 22.5, 23.13
]

plt.figure()
plt.plot(years, change_percent, marker='o', linestyle='-', color='b')
# plt.title('Change in Data Volume Over Previous Year (%)')
plt.xlabel('Year')
plt.ylabel('Change (%)')
plt.grid(True, which="both")
plt.xticks(years, rotation=45)
plt.tight_layout()
plt.show()

# # Total data volume in zettabytes
# zettabytes = [
#     5, 6.5, 9, 12.5, 15.5, 18, 26,
#     33, 41, 64.2, 79, 97, 120, 147, 181
# ]
#
# plt.figure()
# plt.plot(years, zettabytes, marker='o', linestyle='-', color='green')
# # plt.title('Global Data Volume per Year')
# plt.xlabel('Year')
# plt.ylabel('Data Volume (Zettabytes)')
# plt.grid(True, which="both")
# plt.xticks(years, rotation=45)
# plt.tight_layout()
# plt.show()
