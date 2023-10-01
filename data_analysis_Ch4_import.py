gas_import_2023 = 119.85
gas_import_2024 = 105.95
gas_import_2025 = 116.76
gas_import_2026 = 118.60
gas_import_2027 = 120.23
gas_import_2028 = 140.74

CHP_production_2023 = 59.12
CHP_production_2024 = 47.90
CHP_production_2025 = 55.72
CHP_production_2026 = 57.60
CHP_production_2027 = 59.74
CHP_production_2028 = 70.74

HP_heat_production_2023 = 107.96
HP_heat_production_2024 = 106.70
HP_heat_production_2025 = 105.20
HP_heat_production_2026 = 107.68
HP_heat_production_2027 = 107.55
HP_heat_production_2028 = 108.59



import matplotlib.pyplot as plt
import numpy as np

data = {
    'Stage': ['Stage 1', 'Stage 2'],
    'Gas Import for Space Heating (kg)': [15, 19],  # Conversion: 1 MWh = 1000 kg
    'Heat Production CHP (MWh)': [162.84, 172.34],
    'Heat Production HP (MWh)': [222.70, 214.38]
}

# Extract the data
stages = data['Stage']
gas_import_kg = data['Gas Import for Space Heating (kg)']
heat_chp_mwh = data['Heat Production CHP (MWh)']
heat_hp_mwh = data['Heat Production HP (MWh)']

# Calculate the values in MWh
# gas_import_mwh = [kg / 1000 for kg in gas_import_kg]

# Set the width of the bars
bar_width = 0.25
bar_spacing = 0.2  # Space between the sets of bars
index = np.arange(len(stages))

# Create the figure and the first set of axes
fig, ax1 = plt.subplots()

# Create the bar plot for MWh on the first axis

ax1.bar(index, heat_hp_mwh, bar_width, label='HP heat (MWh)', color='blue', alpha=0.7)
ax1.bar(index + bar_width, heat_chp_mwh, bar_width, label='CHP heat (MWh)', color='green', alpha=0.7)

# Create a second set of axes (twin y-axis)
ax2 = ax1.twinx()

# Create the bar plot for kg on the second axis
ax2.bar(index + 2 * bar_width, gas_import_kg, bar_width, label='Gas Import (kg)', color='purple', alpha=0.7)

# Set the y-axis labels and titles for both axes
ax1.set_ylabel('MWh')
ax2.set_ylabel('kg (in thousands)')
ax1.set_xlabel('Stages')

# Set the x-axis ticks and labels
plt.xticks(index + bar_width * 1.0, stages)

# Add legends for both axes
# ax1.legend(loc='center', bbox_to_anchor=[0.4, 0.9], frameon=False)
# ax2.legend(loc='center', bbox_to_anchor=[0.5, 0.9], frameon=False)

ax1.legend(loc='upper left')
ax2.legend(loc=1)

# Show the plot
# plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import numpy as np

# Data
years = ['2023', '2024', '2025', '2026', '2027', '2028']

CHP_production = [59.12*0.15, 47.90*0.15, 55.72*0.15, 57.60*0.15, 59.74*0.15, 70.74*0.15]
HP_heat_production = [107.96*0.15, 106.70*0.15, 105.20*0.15, 107.68*0.15, 107.55*0.15, 108.59*0.15]
gas_import_kg = [119.85*0.15, 105.95*0.15, 116.76*0.15, 118.60*0.15, 120.23*0.15, 140.74*0.15]
# gas_import_kg = [(59.12*0.15)*2, 105.95*0.15, 116.76*0.15, 118.60*0.15, 120.23*0.15, 140.74*0.15]

# Create a bar plot for heat production (CHP and HP)
plt.figure(figsize=(10, 6))
bar_width = 0.35
index = np.arange(len(years))

plt.bar(index, CHP_production, bar_width, label='CHP heat production (MWh)', color='red', alpha=0.7)
plt.bar(index + bar_width, HP_heat_production, bar_width, label='HP heat production (MWh)', color='grey', alpha=0.7)

plt.xlabel('Years', fontsize=12)
plt.ylabel('Heat production (MWh)', fontsize=12)
# plt.title('Heat Production from CHP and HP (2023-2028)')
plt.xticks(index + bar_width / 2, years)
plt.legend(loc='upper left')

# Create a separate y-axis for gas import (CH4) in kg
plt.twinx()
plt.plot(index + bar_width / 2, gas_import_kg, marker='o', color='black', linestyle='-', label='Gas Import (kg)')

plt.ylabel('Gas import (Tsd. kg)', fontsize=12)
plt.legend(loc='upper right')

# Show the plot
plt.tight_layout()
plt.show()
