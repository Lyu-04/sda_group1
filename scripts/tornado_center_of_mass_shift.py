import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

# Styling
plt.rcParams['figure.figsize'] = (14, 10)
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('default')

ALPHA = 0.05  # Significance level for hypothesis testing

# Read the dataset
data_path = '../Data/clean_tornado_tx_1950_2021.csv'
df = pd.read_csv(data_path)

# Filter out invalid coordinates (0.0, 0.0)
df_clean = df[(df['slat'] != 0.0) & (df['slon'] != 0.0)].copy()

print("=" * 80)
print("TEXAS TORNADO CENTER OF MASS SHIFT ANALYSIS (1950-2021)")
print("=" * 80)
print(f"\nTotal tornadoes analyzed: {len(df_clean)}")
print(f"Year range: {df_clean['yr'].min()} - {df_clean['yr'].max()}")
print(f"Latitude range: {df_clean['slat'].min():.2f}° - {df_clean['slat'].max():.2f}°")
print(f"Longitude range: {df_clean['slon'].min():.2f}° - {df_clean['slon'].max():.2f}°")

# ============================================================================
# STEP 1: Calculate Center of Mass by Different Time Periods
# ============================================================================

def calculate_center_of_mass(group, weight_col=None):
    """Calculate weighted center of mass for a group of tornadoes."""
    if weight_col is None:
        # Unweighted: simple average
        com_lat = group['slat'].mean()
        com_lon = group['slon'].mean()
        count = len(group)
    else:
        # Weighted by specified column (e.g., magnitude, width)
        weights = group[weight_col].values
        com_lat = np.average(group['slat'].values, weights=weights)
        com_lon = np.average(group['slon'].values, weights=weights)
        count = len(group)
    return pd.Series({
        'com_lat': com_lat,
        'com_lon': com_lon,
        'count': count
    })

# Calculate by decade
df_clean['decade'] = (df_clean['yr'] // 10) * 10
com_by_decade = df_clean.groupby('decade').apply(
    lambda x: calculate_center_of_mass(x)
).reset_index()

# Calculate by 5-year periods
df_clean['period_5yr'] = (df_clean['yr'] // 5) * 5
com_by_5yr = df_clean.groupby('period_5yr').apply(
    lambda x: calculate_center_of_mass(x)
).reset_index()

# Calculate by year
com_by_year = df_clean.groupby('yr').apply(
    lambda x: calculate_center_of_mass(x)
).reset_index()

# Calculate weighted by magnitude (if magnitude >= 0)
df_mag = df_clean[df_clean['mag'] >= 0].copy()
com_by_year_weighted = df_mag.groupby('yr').apply(
    lambda x: calculate_center_of_mass(x, weight_col='mag')
).reset_index()

print("\n" + "=" * 80)
print("CENTER OF MASS BY DECADE")
print("=" * 80)
print(com_by_decade[['decade', 'com_lat', 'com_lon', 'count']].to_string(index=False))

# ============================================================================
# STEP 2: Statistical Analysis - Test for Significant Shift
# ============================================================================

print("\n" + "=" * 80)
print("STATISTICAL ANALYSIS: Testing for Significant Shift")
print("=" * 80)

# Linear regression on latitude over time
slope_lat, intercept_lat, r_value_lat, p_value_lat, std_err_lat = stats.linregress(
    com_by_year['yr'], com_by_year['com_lat']
)

# Linear regression on longitude over time
slope_lon, intercept_lon, r_value_lon, p_value_lon, std_err_lon = stats.linregress(
    com_by_year['yr'], com_by_year['com_lon']
)

print(f"\nLATITUDE SHIFT:")
print(f"  Slope: {slope_lat:.6f} degrees/year")
print(f"  Total shift (1950-2021): {slope_lat * 71:.4f} degrees")
print(f"  R-squared: {r_value_lat**2:.4f}")
print(f"  P-value: {p_value_lat:.6f}")
print(f"  Significant? {'YES' if p_value_lat < ALPHA else 'NO'} (α={ALPHA})")

print(f"\nLONGITUDE SHIFT:")
print(f"  Slope: {slope_lon:.6f} degrees/year")
print(f"  Total shift (1950-2021): {slope_lon * 71:.4f} degrees")
print(f"  R-squared: {r_value_lon**2:.4f}")
print(f"  P-value: {p_value_lon:.6f}")
print(f"  Significant? {'YES' if p_value_lon < ALPHA else 'NO'} (α={ALPHA})")

# Convert degrees to approximate distance (1 degree ≈ 111 km)
lat_shift_km = slope_lat * 71 * 111
lon_shift_km = slope_lon * 71 * 111 * np.cos(np.radians(com_by_year['com_lat'].mean()))
total_shift_km = np.sqrt(lat_shift_km**2 + lon_shift_km**2)

print(f"\nAPPROXIMATE DISTANCE SHIFT (1950-2021):")
print(f"  North-South: {lat_shift_km:.2f} km")
print(f"  East-West: {lon_shift_km:.2f} km")
print(f"  Total distance: {total_shift_km:.2f} km")

# ============================================================================
# STEP 3: Create Visualizations
# ============================================================================

output_dir = '../plots'
os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# Figure 1: Center of Mass Trajectory Plots (Grouped Together)
# ============================================================================

# Calculate bounds for center of mass region (with padding)
# Combine all center of mass points to determine zoom region
all_com_lat = pd.concat([com_by_decade['com_lat'], com_by_5yr['com_lat']])
all_com_lon = pd.concat([com_by_decade['com_lon'], com_by_5yr['com_lon']])

lat_min, lat_max = all_com_lat.min(), all_com_lat.max()
lon_min, lon_max = all_com_lon.min(), all_com_lon.max()

# Add padding (15% on each side)
lat_range = lat_max - lat_min
lon_range = lon_max - lon_min
lat_padding = lat_range * 0.15
lon_padding = lon_range * 0.15

lat_lim = [lat_min - lat_padding, lat_max + lat_padding]
lon_lim = [lon_min - lon_padding, lon_max + lon_padding]

print(f"\nZoom region for trajectory plots:")
print(f"  Latitude: {lat_lim[0]:.4f}° to {lat_lim[1]:.4f}° (range: {lat_range:.4f}°)")
print(f"  Longitude: {lon_lim[0]:.4f}° to {lon_lim[1]:.4f}° (range: {lon_range:.4f}°)")

# Create color gradients
decade_colors = plt.cm.plasma(np.linspace(0, 1, len(com_by_decade)))
period_colors = plt.cm.plasma(np.linspace(0, 1, len(com_by_5yr)))

# Helper function to plot trajectory
def plot_trajectory(ax, com_data, colors, title, colorbar_label, zoom=False, label_every=1):
    """Plot center of mass trajectory with color gradient."""
    ax.scatter(df_clean['slon'], df_clean['slat'],
               alpha=0.1 if zoom else 0.05, s=1, c='lightgray', label='All tornadoes')

    # Plot line segments with color gradient
    for i in range(len(com_data) - 1):
        ax.plot([com_data.iloc[i]['com_lon'], com_data.iloc[i+1]['com_lon']],
                [com_data.iloc[i]['com_lat'], com_data.iloc[i+1]['com_lat']],
                color=colors[i], linewidth=3, alpha=0.8)

    # Plot markers with color gradient
    for idx, (i, row) in enumerate(com_data.iterrows()):
        marker_size = 150 if 'decade' in com_data.columns else 120
        ax.scatter(row['com_lon'], row['com_lat'],
                   s=marker_size, c=[colors[idx]], edgecolors='black',
                   linewidths=1.5, zorder=5, alpha=0.9)

        # Add labels
        if 'decade' in com_data.columns:
            label = f"{int(row['decade'])}s"
            fontsize = 8
        else:
            if idx % label_every == 0:
                label = f"{int(row['period_5yr'])}"
                fontsize = 7
            else:
                label = None
                fontsize = 7

        if label:
            ax.annotate(label,
                       (row['com_lon'], row['com_lat']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=fontsize, fontweight='bold')

    # Add colorbar
    if 'decade' in com_data.columns:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma,
                                    norm=plt.Normalize(vmin=com_data['decade'].min(),
                                                       vmax=com_data['decade'].max()))
    else:
        sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma,
                                    norm=plt.Normalize(vmin=com_data['period_5yr'].min(),
                                                       vmax=com_data['period_5yr'].max()))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(colorbar_label, fontsize=10, fontweight='bold')

    ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')

    if zoom:
        ax.set_xlim(lon_lim)
        ax.set_ylim(lat_lim)

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Left: Decade trajectory
plot_trajectory(ax1, com_by_decade, decade_colors,
                'Trajectory by Decade', 'Decade', zoom=False)

# Right: 5-year trajectory
plot_trajectory(ax2, com_by_5yr, period_colors,
                'Trajectory by 5-Year Periods', '5-Year Period Start',
                zoom=False, label_every=3)

plt.tight_layout()
output_path1 = os.path.join(output_dir, 'tornado_center_of_mass_trajectories.png')
plt.savefig(output_path1, dpi=300, bbox_inches='tight')
print(f"\nTrajectory plots saved to {output_path1}")

# ============================================================================
# Figure 2: Time Series Plots (Latitude and Longitude - Grouped Together)
# ============================================================================

fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(16, 6))

# Plot 2: Time series of latitude shift
ax2.scatter(com_by_year['yr'], com_by_year['com_lat'],
           alpha=0.6, s=50, color='blue', label='Yearly Center of Mass')
ax2.plot(com_by_year['yr'],
        intercept_lat + slope_lat * com_by_year['yr'],
        'r--', linewidth=2, label=f'Linear Trend (p={p_value_lat:.4f})')
ax2.set_xlabel('Year', fontsize=12, fontweight='bold')
ax2.set_ylabel('Center of Mass Latitude (°)', fontsize=12, fontweight='bold')
ax2.set_title('Latitude Shift Over Time', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Time series of longitude shift
ax3.scatter(com_by_year['yr'], com_by_year['com_lon'],
           alpha=0.6, s=50, color='green', label='Yearly Center of Mass')
ax3.plot(com_by_year['yr'],
        intercept_lon + slope_lon * com_by_year['yr'],
        'r--', linewidth=2, label=f'Linear Trend (p={p_value_lon:.4f})')
ax3.set_xlabel('Year', fontsize=12, fontweight='bold')
ax3.set_ylabel('Center of Mass Longitude (°)', fontsize=12, fontweight='bold')
ax3.set_title('Longitude Shift Over Time', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
output_path2 = os.path.join(output_dir, 'tornado_center_of_mass_timeseries.png')
plt.savefig(output_path2, dpi=300, bbox_inches='tight')
print(f"Time series plots saved to {output_path2}")

# Also save a combined figure for convenience
fig_combined, axes = plt.subplots(2, 2, figsize=(16, 12))

# Top left: Decade trajectory
plot_trajectory(axes[0, 0], com_by_decade, decade_colors,
                'Trajectory of Tornado Center of Mass (by Decade)', 'Decade', zoom=False)

axes[0, 1].scatter(com_by_year['yr'], com_by_year['com_lat'],
                   alpha=0.6, s=50, color='blue', label='Yearly Center of Mass')
axes[0, 1].plot(com_by_year['yr'],
                intercept_lat + slope_lat * com_by_year['yr'],
                'r--', linewidth=2, label=f'Linear Trend (p={p_value_lat:.4f})')
axes[0, 1].set_xlabel('Year', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Center of Mass Latitude (°)', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Latitude Shift Over Time', fontsize=14, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

axes[1, 0].scatter(com_by_year['yr'], com_by_year['com_lon'],
                   alpha=0.6, s=50, color='green', label='Yearly Center of Mass')
axes[1, 0].plot(com_by_year['yr'],
                intercept_lon + slope_lon * com_by_year['yr'],
                'r--', linewidth=2, label=f'Linear Trend (p={p_value_lon:.4f})')
axes[1, 0].set_xlabel('Year', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Center of Mass Longitude (°)', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Longitude Shift Over Time', fontsize=14, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Bottom right: 5-year trajectory
plot_trajectory(axes[1, 1], com_by_5yr, period_colors,
                'Trajectory of Tornado Center of Mass (5-Year Periods)',
                '5-Year Period Start', zoom=False, label_every=3)

plt.tight_layout()
output_path_combined = os.path.join(output_dir, 'tornado_center_of_mass_shift.png')
plt.savefig(output_path_combined, dpi=300, bbox_inches='tight')
print(f"Combined plot saved to {output_path_combined}")

# ============================================================================
# STEP 4: Save Results to CSV
# ============================================================================

# Save center of mass data
results_dir = '../Data'
os.makedirs(results_dir, exist_ok=True)

# Save by decade
com_by_decade.to_csv(os.path.join(results_dir, 'center_of_mass_by_decade.csv'), index=False)
print(f"Results saved to {os.path.join(results_dir, 'center_of_mass_by_decade.csv')}")

# Save by year
com_by_year.to_csv(os.path.join(results_dir, 'center_of_mass_by_year.csv'), index=False)
print(f"Results saved to {os.path.join(results_dir, 'center_of_mass_by_year.csv')}")

# Save by 5-year periods
com_by_5yr.to_csv(os.path.join(results_dir, 'center_of_mass_by_5yr.csv'), index=False)
print(f"Results saved to {os.path.join(results_dir, 'center_of_mass_by_5yr.csv')}")

# Save summary statistics
summary = {
    'metric': ['Latitude Slope (deg/year)', 'Longitude Slope (deg/year)',
               'Latitude R-squared', 'Longitude R-squared',
               'Latitude P-value', 'Longitude P-value',
               'Total Lat Shift (deg)', 'Total Lon Shift (deg)',
               'Total Shift Distance (km)'],
    'value': [slope_lat, slope_lon, r_value_lat**2, r_value_lon**2,
              p_value_lat, p_value_lon, slope_lat * 71, slope_lon * 71,
              total_shift_km]
}
summary_df = pd.DataFrame(summary)
summary_df.to_csv(os.path.join(results_dir, 'center_of_mass_shift_summary.csv'), index=False)
print(f"Summary statistics saved to {os.path.join(results_dir, 'center_of_mass_shift_summary.csv')}")

print("\n" + "=" * 80)
print("STATISTICAL CONCLUSION")
print("=" * 80)
print("\nNull Hypothesis (H0): The center of mass of tornado occurrences has not changed over time.")
print(f"Alternative Hypothesis (H1): The center of mass of tornado occurrences has changed over time.")
print(f"Significance level (α): {ALPHA}")

if p_value_lat < ALPHA or p_value_lon < ALPHA:
    print("\n" + "-" * 80)
    print("RESULT: We REJECT the null hypothesis.")
    print("-" * 80)
    print("There is sufficient statistical evidence to conclude that the center of mass")
    print("of tornado occurrences in Texas has changed significantly over time (1950-2021).")
    if p_value_lat < ALPHA:
        direction_lat = "northward" if slope_lat > 0 else "southward"
        print(f"\n  - Latitude: Significant shift {direction_lat} (p = {p_value_lat:.4f})")
        print(f"    Rate: {abs(slope_lat):.6f} degrees/year")
    if p_value_lon < ALPHA:
        direction_lon = "eastward" if slope_lon > 0 else "westward"
        print(f"\n  - Longitude: Significant shift {direction_lon} (p = {p_value_lon:.4f})")
        print(f"    Rate: {abs(slope_lon):.6f} degrees/year")
else:
    print("\n" + "-" * 80)
    print("RESULT: We FAIL TO REJECT the null hypothesis.")
    print("-" * 80)
    print("There is insufficient statistical evidence to conclude that the center of mass")
    print("of tornado occurrences in Texas has changed significantly over time (1950-2021).")
    print(f"\n  - Latitude: p = {p_value_lat:.4f} (not significant)")
    print(f"  - Longitude: p = {p_value_lon:.4f} (not significant)")
    print("\nNote: This does not prove that no shift has occurred, but rather that")
    print("we cannot detect a statistically significant linear trend in the data.")
    print("The shift may be too subtle, non-linear, or require a longer time period to detect.")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)

plt.show()
