import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os

# Read the dataset
data_path = '../Data/clean_tornado_tx_1950_2021.csv'
df = pd.read_csv(data_path)

# Only valid magnitudes and lengths
df = df[df['mag'] >= 0]         # mag: 0 to 5
df = df[df['len'] > 0]          # len in miles
df = df[df['wid'] > 0]          # wid in yards

# Compute yearly averages
yearly = df.groupby('yr').agg({
    'mag': 'mean',
    'len': 'mean',
    'wid': 'mean',
    'yr': 'count'
}).rename(columns={'yr': 'count'}).reset_index()

print(yearly)

# Linear regression function
def run_regression(x, y):
    slope, intercept, r, p, _ = stats.linregress(x, y)
    return slope, intercept, r**2, p


# Plot the magnitude trend
plt.figure(figsize=(10,6))
plt.scatter(yearly['yr'], yearly['mag'], s=50, alpha=0.6)
slope_mag, intercept_mag, r2_mag, p_mag = run_regression(yearly['yr'], yearly['mag'])
plt.plot(yearly['yr'], intercept_mag + slope_mag * yearly['yr'], 'r--', linewidth=2)

plt.title(f"Tornado Magnitude Trend Over Time (p={p_mag:.4f})")
plt.xlabel("Year")
plt.ylabel("Average Magnitude (F-scale)")
plt.grid(True, alpha=0.3)
plt.savefig("../plots/magnitude_trend.png", dpi=300)

# Plot the length trend
plt.figure(figsize=(10,6))
plt.scatter(yearly['yr'], yearly['len'], s=50, alpha=0.6)
slope_len, intercept_len, r2_len, p_len = run_regression(yearly['yr'], yearly['len'])
plt.plot(yearly['yr'], intercept_len + slope_len * yearly['yr'], 'r--', linewidth=2)

plt.title(f"Tornado Length Trend Over Time (p={p_len:.4f})")
plt.xlabel("Year")
plt.ylabel("Average Length (miles)")
plt.grid(True, alpha=0.3)
plt.savefig("../plots/length_trend.png", dpi=300)

# Plot the width trend
plt.figure(figsize=(10,6))
plt.scatter(yearly['yr'], yearly['wid'], s=50, alpha=0.6)
slope_wid, intercept_wid, r2_wid, p_wid = run_regression(yearly['yr'], yearly['wid'])
plt.plot(yearly['yr'], intercept_wid + slope_wid * yearly['yr'], 'r--', linewidth=2)

plt.title(f"Tornado Width Trend Over Time (p={p_wid:.4f})")
plt.xlabel("Year")
plt.ylabel("Average Width (yards)")
plt.grid(True, alpha=0.3)
plt.savefig("../plots/width_trend.png", dpi=300)

# Summary dataframe
summary_df = pd.DataFrame({
    'metric': ['mag_slope', 'mag_r2', 'mag_p',
               'len_slope', 'len_r2', 'len_p',
               'wid_slope', 'wid_r2', 'wid_p'],
    'value': [slope_mag, r2_mag, p_mag,
              slope_len, r2_len, p_len,
              slope_wid, r2_wid, p_wid]
})

summary_df.to_csv("../Data/tornado_intensity_trend_summary.csv", index=False)
