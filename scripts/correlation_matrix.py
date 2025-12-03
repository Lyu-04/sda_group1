import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Read the dataset
data_path = '../Data/final_data.csv'
df = pd.read_csv(data_path)

# Select only numeric columns
numeric_df = df.select_dtypes(include=['float64', 'int64'])

# Compute correlation matrix
corr_matrix = numeric_df.corr(method="pearson")

# Plot
plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, cmap="coolwarm", annot=False)
plt.title("Correlation Matrix: Texas Tornado Meteorological Factors")

# Save the plot
output_dir = '../plots'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
print(f"Plot saved to {os.path.join(output_dir, 'correlation_matrix.png')}")

plt.show()
