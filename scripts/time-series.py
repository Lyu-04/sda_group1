import pandas as pd
import csv
import matplotlib.pyplot as plt

input_file = r"Data\final_data1.csv"
df = pd.read_csv(input_file)

lat_list = []
lat_min_1 = []
lon_list = []
lon_min_1 = []

for i in range(1, len(df["slat"])):
    lat_list.append(df["slat"][i])
    lat_min_1.append(df["slat"][i-1])
    lon_list.append(df["slon"][i])
    lon_min_1.append(df["slon"][i-1])

fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.scatter(lat_min_1, lat_list, alpha = 0.1)
ax1.set_xlabel("latitude (-1)")
ax1.set_ylabel("latitude")

ax2.scatter(lon_min_1, lon_list, alpha = 0.1)
ax2.set_xlabel("longitude (-1)")
ax2.set_ylabel("longitude")

plt.show()