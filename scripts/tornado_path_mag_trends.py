import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os

from linear_assumption_tests import test_all_metrics

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


# Plot the trends
def plot_trend(df, xcol, ycol, ylabel, title_prefix, outfile):
    plt.figure(figsize=(10,6))

    # Scatter
    plt.scatter(df[xcol], df[ycol], s=50, alpha=0.6)

    # Regression
    slope, intercept, r2, p = run_regression(df[xcol], df[ycol])
    plt.plot(df[xcol], intercept + slope * df[xcol], 'r--', linewidth=2)

    # Labels & Style
    plt.title(f"{title_prefix} (p={p:.4f})")
    plt.xlabel(xcol.capitalize())
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)

    # Save
    plt.savefig(outfile, dpi=300)
    plt.close()

    return slope, intercept, r2, p

# Calculate metrics
metrics = {
    'mag': plot_trend(yearly, 'yr', 'mag', "Average Magnitude (F-scale)", "Tornado Magnitude Trend Over Time", "../plots/magnitude_trend.png"),
    'len': plot_trend(yearly, 'yr', 'len', "Average Length (miles)", "Tornado Length Trend Over Time", "../plots/length_trend.png"),
    'wid': plot_trend(yearly, 'yr', 'wid', "Average Width (yards)", "Tornado Width Trend Over Time", "../plots/width_trend.png")
}

# Summary dataframe
summary_df = pd.DataFrame({
    'metric': ['mag_slope', 'mag_r2', 'mag_p',
               'len_slope', 'len_r2', 'len_p',
               'wid_slope', 'wid_r2', 'wid_p'],
    'value': [metrics['mag'][0], metrics['mag'][2], metrics['mag'][3],
              metrics['len'][0], metrics['len'][2], metrics['len'][3],
              metrics['wid'][0], metrics['wid'][2], metrics['wid'][3]]
})

summary_df.to_csv(os.path.join('../Data', 'tornado_path_shift_summary.csv'), index=False)
print(f"Summary statistics saved to {os.path.join('../Data', 'tornado_path_shift_summary.csv')}")


# Test all the metrics
test_all_metrics(yearly, metrics, metric_list=['mag', 'len', 'wid'], outdir='../plots')
