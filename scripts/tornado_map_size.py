import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Styling of the plot
plt.rcParams['figure.figsize'] = (12, 10)

# Read the dataset
data_path = '../Data/clean_tornado_tx_1950_2021.csv'
df = pd.read_csv(data_path)

# Filter out invalid coordinates (0.0, 0.0), invalid widths, and very small tornadoes (< 30 yards)
# This reduces clutter and focuses on more significant tornadoes
min_width = 30   # Median is 33 yards for all tornadoes in Texas
df_clean = df[(df['slat'] != 0.0) & (df['slon'] != 0.0) & (df['wid'] >= min_width)].copy()

# Create the scatter map
fig, ax = plt.subplots(figsize=(12, 10))

# Calculate marker sizes proportional to width
marker_sizes = 10 + (df_clean['wid'] / df_clean['wid'].max()) * 200

# Calculate alpha values proportional to width
# Small tornadoes: very transparent (0.2), large tornadoes: fully opaque (1.0)
# Normalize width to 0-1 range, then map to alpha range
width_normalized = (df_clean['wid'] - df_clean['wid'].min()) / (df_clean['wid'].max() - df_clean['wid'].min())
alpha_values = 0.2 + width_normalized * 0.8  # Range from 0.2 to 1.0

# Apply log transformation to width values for better color distribution
width_log = np.log10(df_clean['wid'] + 1)  # +1 to avoid log(0) issues
width_log_normalized = (width_log - width_log.min()) / (width_log.max() - width_log.min())

# Create scatter plot colored by tornado width (size) with log scale
# Using log-transformed values for better color distribution across wide range
scatter = ax.scatter(df_clean['slon'], df_clean['slat'], 
                     c=width_log_normalized, 
                     cmap='viridis', 
                     s=marker_sizes, 
                     alpha=alpha_values, 
                     edgecolors='black', 
                     linewidths=0.2)

# Customizations to the plot
cbar = plt.colorbar(scatter, ax=ax)
# Create custom tick labels for log scale
width_min = df_clean['wid'].min()
width_max = df_clean['wid'].max()
# Generate log-spaced tick positions
log_ticks = np.logspace(np.log10(width_min + 1), np.log10(width_max + 1), num=6, base=10) - 1
log_tick_positions = (np.log10(log_ticks + 1) - width_log.min()) / (width_log.max() - width_log.min())
cbar.set_ticks(log_tick_positions)
cbar.set_ticklabels([f'{int(t)}' for t in log_ticks])
cbar.set_label('Tornado Width (yards, log scale)', 
               fontsize=12, fontweight='bold')
ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
ax.set_title(f'Tornado Locations in Texas Colored by Size (Width ≥ {min_width} yards) (1950-2021)', 
             fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--')

# Set aspect ratio to maintain geographic proportions
ax.set_aspect('equal', adjustable='box')

# Save the plot
output_dir = '../plots'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'tornado_map_size.png'), 
            dpi=300, bbox_inches='tight')
print(f"Plot saved to {os.path.join(output_dir, 'tornado_map_size.png')}")

plt.show()

# Print summary statistics
print(f"\nFiltering: Only showing tornadoes with width ≥ {min_width} yards")
print(f"Total tornadoes plotted: {len(df_clean)}")
print(f"Width range: {df_clean['wid'].min():.0f} to {df_clean['wid'].max():.0f} yards")
print(f"Mean width: {df_clean['wid'].mean():.2f} yards")
print(f"Median width: {df_clean['wid'].median():.2f} yards")
