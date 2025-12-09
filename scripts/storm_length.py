import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

input_file = r"Data\cleaned\clean_tornado_tx_1970_2021.csv"
df = pd.read_csv(input_file)

counts = df["date"].value_counts().sort_index()

# make new dataframe for storms
storm_df = pd.DataFrame(columns = ["storm_length","avg_lat", "avg_lon"])
# check storm length and calculate average values
# set u with values for first entry
lat_storage = [df["slat"][0]]
lon_storage = [df["slon"][0]]
counter = 1
# loop back to beginning to save last storm 
for i in range(len(df["date"]) + 1):
    # check if streak continues or not
    if df.iloc[i % len(df)]["date"] == df.iloc[i-1]["date"]:
        lat_storage.append(df.iloc[i % len(df)]["slat"])
        lon_storage.append(df.iloc[i % len(df)]["slon"])
        counter += 1

    else :
        storm_df.loc[len(storm_df)] = {"storm_length" : counter, "avg_lat" : np.mean(lat_storage), "avg_lon" : np.mean(lon_storage)}
        # clean values again
        counter = 1
        lat_storage = [df.iloc[i % len(df)]["slat"]]
        lon_storage = [df.iloc[i % len(df)]["slon"]]

# save df to csv
storm_df.to_csv(r"Data\processed\unique_storms.csv", encoding="utf-8", index=False)

plt.hist(counts, bins = [i-0.5 for i in range(1, max(counts))], label = "count per storm", edgecolor = "k")
plt.xticks(range(1, max(counts) - 1))
plt.xlabel("storm duration (# tornado's)")
plt.ylabel("frequencies")
plt.legend(loc="best")
plt.savefig(r"plots\storm_lengths_hist.png")
#plt.show()

