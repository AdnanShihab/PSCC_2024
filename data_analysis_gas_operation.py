
gree_h2_production_2023 = 36.46535319
gree_h2_production_2024 = 20.35497835
gree_h2_production_2025 = 22.60606029
gree_h2_production_2026 = 27.74301009
gree_h2_production_2027 = 55.91340161
gree_h2_production_2028 = 0


blue_h2_import_2023 = 1.254646815
blue_h2_import_2024 = 28.38502165
blue_h2_import_2025 = 35.66113971
blue_h2_import_2026 = 47.57698991
blue_h2_import_2027 = 35.87659839
blue_h2_import_2028 = 117.49


import matplotlib.pyplot as plt

plt.margins(x=0)
font = {'family': 'normal',
        'size': 14}
plt.rc('font', **font)

# Data
years = ['2023', '2024', '2025', '2026', '2027', '2028']

green_h2_production = [36.46535319, 20.35497835, 22.60606029, 27.74301009, 55.91340161, 0]
blue_h2_import = [1.254646815, 28.38502165, 35.66113971, 47.57698991, 35.87659839, 117.49]

# Create a bar plot for green H2 production stacked on top of blue H2 import
plt.bar(years, blue_h2_import, color='blue', alpha=0.8, label='Blue H2 import (MWh)')
plt.bar(years, green_h2_production, bottom=blue_h2_import, color='green', alpha=0.7, label='Green H2 production (MWh)')

plt.xlabel('Years', fontsize=12)
plt.ylabel('Hydrogen production and import (MWh)', fontsize=12)
# plt.title('Green and Blue Hydrogen Production and Import (2023-2028)')
plt.legend(fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()



