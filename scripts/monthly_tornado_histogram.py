import pandas as pd
import matplotlib.pyplot as plt
import os

# Styling of the plot
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'

# Read the dataset
data_path = '../Data/clean_tornado_tx_1950_2021.csv'
df = pd.read_csv(data_path)

# Count tornadoes by month
monthly_counts = df['mo'].value_counts().sort_index()

# Month names for labels
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Create the histogram
plt.figure(figsize=(12, 6))
bars = plt.bar(monthly_counts.index, monthly_counts.values, 
               color='skyblue', edgecolor='black', alpha=0.7)

# Customize the plot
plt.xlabel('Month', fontsize=12, fontweight='bold')
plt.ylabel('Number of Tornadoes', fontsize=12, fontweight='bold')
plt.title('Number of Tornadoes in Texas by Month (1950-2021)', 
          fontsize=14, fontweight='bold', pad=20)

# Set x-axis labels to month names
plt.xticks(monthly_counts.index, [month_names[i-1] for i in monthly_counts.index])

# Value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}',
             ha='center', va='bottom', fontsize=9)

plt.grid(axis='y', alpha=0.3, linestyle='--')

# Save the plot
output_dir = '../plots'
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'monthly_tornado_histogram.png'))
print(f"Plot saved to {os.path.join(output_dir, 'monthly_tornado_histogram.png')}")

plt.show()

# Print summary statistics
print(f"\nTotal tornadoes: {monthly_counts.sum()}")
