import pandas as pd
import matplotlib.pyplot as plt

input_file = r"Data\clean_tornado_tx_1970_2021.csv"
df = pd.read_csv(input_file)

counts = df["date"].value_counts().sort_index()

plt.hist(counts, bins = [i-0.5 for i in range(1, max(counts))], label = "count per storm", edgecolor = "k")
plt.xticks(range(1, max(counts) - 1))
plt.xlabel("storm duration (# tornado's)")
plt.ylabel("frequencies")
plt.legend(loc="best")
plt.show()
plt.savefig(r"plots\storm_lengths_hist.png")
