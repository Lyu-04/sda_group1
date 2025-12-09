import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

input_file = r"Data\final\final_data1.csv"
df = pd.read_csv(input_file)

# extracting all needed data
lat_list = []
lat_min_1 = []
lon_list = []
lon_min_1 = []

for i in range(1, len(df["slat"])):
    lat_list.append(df["slat"][i])
    lat_min_1.append(df["slat"][i-1])
    lon_list.append(df["slon"][i])
    lon_min_1.append(df["slon"][i-1])

# data extraction for total storm data
input_file = r"Data\processed\unique_storms.csv"
storm_df = pd.read_csv(input_file)
avg_lat_list = []
avg_lat_min_1 = []
avg_lon_list = []
avg_lon_min_1 = []
for i in range(1, len(storm_df)):
    avg_lat_list.append(storm_df["avg_lat"][i])
    avg_lat_min_1.append(storm_df["avg_lat"][i-1])
    avg_lon_list.append(storm_df["avg_lon"][i])
    avg_lon_min_1.append(storm_df["avg_lon"][i-1])


# plotting
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

ax1.scatter(lat_min_1, lat_list, alpha = 0.1)
#ax1.plot(x_lat, lat_fit, label= "lineair fit", color = "black")
ax1.set_xlabel("latitude (-1)")
ax1.set_ylabel("latitude")
ax1.set_ylim(25, 37)
ax1.set_xlim(25, 37)

ax2.scatter(lon_min_1, lon_list, alpha = 0.1)
#ax2.plot(x_lon, lon_fit, label= "lineair fit", color = "black")
ax2.set_xlabel("longitude (-1)")
ax2.set_ylabel("longitude")
ax2.set_ylim(-107, -93)
ax2.set_xlim(-107, -93)

ax3.scatter(avg_lat_min_1, avg_lat_list, alpha = 0.1)
ax3.set_xlabel("avg latitude (-1)")
ax3.set_ylabel("avg latitude")
ax3.set_ylim(25, 37)
ax3.set_xlim(25, 37)

ax4.scatter(avg_lon_min_1, avg_lon_list, alpha = 0.1)
ax4.set_xlabel("avg longitude (-1)")
ax4.set_ylabel("avg longitude")
ax4.set_ylim(-107, -93)
ax4.set_xlim(-107, -93)

plt.savefig(r"plots\tornado_time_series.png")
plt.show()
