import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Styling of the plot
plt.rcParams['figure.figsize'] = (12, 10)

# Read the dataset
data_path = '../Data/cleaned/clean_tornado_tx_1950_2021.csv'
df = pd.read_csv(data_path)

# Filter out invalid coordinates (0.0, 0.0) and invalid magnitudes
# Note: mag = -9 indicates unknown rating, so we filter those out (mag >= 0)
df_clean = df[(df['slat'] != 0.0) & (df['slon'] != 0.0) & (df['mag'] >= 0)].copy()

# Create the scatter map
fig, ax = plt.subplots(figsize=(12, 10))

# Calculate alpha values with exponential scaling based on magnitude
# Small magnitude tornadoes: very transparent (0.1), large magnitude: very opaque (0.9)
# Normalize magnitude to 0-1 range, then apply exponential scaling
# Magnitude is already on 0-5 scale, so we normalize it
mag_normalized = (df_clean['mag'] - df_clean['mag'].min()) / (df_clean['mag'].max() - df_clean['mag'].min())
# Exponential scaling: using power of 3 for exponential curve
# This makes low magnitudes very transparent, high magnitudes very opaque
alpha_values = 0.1 + 0.8 * (mag_normalized ** 3)  # Range from 0.1 to 0.9 with exponential scaling

# Calculate marker sizes proportional to magnitude
marker_sizes = 10 + (mag_normalized * 150)  # Range from 10 to 160

# Create scatter plot colored by magnitude
scatter = ax.scatter(df_clean['slon'], df_clean['slat'],
                     c=df_clean['mag'],
                     cmap='YlOrRd',
                     s=marker_sizes,
                     alpha=alpha_values,
                     edgecolors='black',
                     linewidths=0.3)

# Customizations to the plot
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Tornado Magnitude (Enhanced Fujita Scale)',
               fontsize=12, fontweight='bold')
ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
ax.set_title('Tornado Locations in Texas Colored by Magnitude (1950-2021)',
             fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--')

# Set aspect ratio to maintain geographic proportions
ax.set_aspect('equal', adjustable='box')

# Save the plot into the maps subfolder
output_dir = '../plots/maps'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'tornado_map_magnitude.png'),
            dpi=300, bbox_inches='tight')
print(f"Plot saved to {os.path.join(output_dir, 'tornado_map_magnitude.png')}")

plt.show()

# Print summary statistics
print(f"\nFiltering: Excluding tornadoes with unknown magnitude (mag = -9)")
print(f"Total tornadoes plotted: {len(df_clean)}")
print(f"Magnitude distribution:")
print(df_clean['mag'].value_counts().sort_index())
