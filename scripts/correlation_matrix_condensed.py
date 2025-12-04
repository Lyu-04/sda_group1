import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Read the dataset
data_path = '../Data/final_data1.csv'
df = pd.read_csv(data_path)

# You can change cols_to_drop to drop other desired columns if you want
cols_to_drop = ['event_id', 'st', 'date', 'yr', 'mo', 'dy', 'latitude', 'longitude', 'slat', 'slon', 'elat', 'elon', 'dist2']
clean_df = df.drop(columns=cols_to_drop)

# Select only numeric columns
numeric_df = clean_df.select_dtypes(include=['float64', 'int64'])

# Compute correlation matrix
corr_matrix = numeric_df.corr(method="pearson")

# Plot
plt.figure(figsize=(12,10))
sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)
plt.title("Correlation Matrix: Texas Tornado Meteorological Factors")

# Save the plot
output_dir = '../plots'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'correlation_matrix_condensed.png'))
print(f"Plot saved to {os.path.join(output_dir, 'correlation_matrix_condensed.png')}")

plt.show()
