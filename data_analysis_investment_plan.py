year_2023 = 10437711.13
year_2024 = 8377623.30
year_2025 = 12847140.11

year_2026 = 12764446.35
year_2027 = 11585146.31
year_2028 = 4216592.47

stage_1_tot = (year_2023 + year_2024 + year_2025)
stage_2_tot = (year_2026 + year_2027 + year_2028)


import matplotlib.pyplot as plt

# Define the data
years = ['2023', '2024', '2025', '2026', '2027', '2028']
stage_1 = [year_2023, year_2024, year_2025, 0, 0, 0]  # Stage 1 data up to 2025
stage_2 = [0, 0, 0, year_2026, year_2027, year_2028]  # Stage 2 data from 2026 onwards

# Convert amounts to million Euros
stage_1_million = [amount / 1_000_000 for amount in stage_1]
stage_2_million = [amount / 1_000_000 for amount in stage_2]

# Calculate the accumulated amounts for each year
stage_1_accumulated = [sum(stage_1_million[:i+1]) for i in range(len(stage_1_million))]
stage_2_accumulated = [sum(stage_2_million[:i+1]) for i in range(len(stage_2_million))]

# Create a bar plot
plt.figure(figsize=(12, 6))

# Plot Stage 1 and Stage 2 investment costs as bars with different colors
plt.bar(years[:3], stage_1_million[:3], label='Stage 1', color='blue', alpha=0.7)
plt.bar(years[3:], stage_2_million[3:], label='Stage 2', color='green', alpha=0.7)

# Plot accumulated amounts as lines with markers
plt.plot(years[:3], stage_1_accumulated[:3], marker='o', linestyle='-', color='blue', label='Stage 1 Accumulated')
plt.plot(years[3:], stage_2_accumulated[3:], marker='o', linestyle='-', color='green', label='Stage 2 Accumulated')

# Increase font size to 15pt for labels, title, years, and amounts
plt.xlabel('Year', fontsize=15)
plt.ylabel('Amount in Million Euros', fontsize=15)
# plt.title('Stage 1 vs. Stage 2 Investment and Accumulated Amounts', fontsize=15)

# Increase font size for years and amounts on the ticks
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# Display amounts as integers on the y-axis ticks
plt.gca().get_yaxis().get_major_formatter().set_scientific(False)

plt.legend(fontsize=15)
plt.show()










