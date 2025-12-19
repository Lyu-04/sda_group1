import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

input_file = r"Data\final\final_data1.csv"
df = pd.read_csv(input_file)

#cross corelation
i_list = []
lon_correlation = []
lat_correlation = []
lon_p = []
lat_p = []
for i in range(1, 41):
    # extract needed "minus x" data and calculate correlation
    lat = []
    lat_m_x = []
    lon = []
    lon_m_x = []
    for j in range(i, len(df["slat"])):
        lat.append(df["slat"][j])
        lat_m_x.append(df["slat"][j-i])
        lon.append(df["slon"][j])
        lon_m_x.append(df["slon"][j-i])

    lat_r, lat_pv = stats.pearsonr(lat_m_x, lat)
    lon_r, lon_pv = stats.pearsonr(lon_m_x, lon)
    i_list.append(i)
    lat_correlation.append(lat_r)
    lat_p.append(lat_pv)
    lon_correlation.append(lon_r)
    lon_p.append(lon_pv)

# plot results
fig, (ax1, ax2) = plt.subplots(1, 2)

ax1.scatter(i_list, lat_correlation, label = "latitude", color = "blue")
ax1.set_ylabel("correlation coefficient")
ax1.set_xlabel("time steps back")

ax2.scatter(i_list, lon_correlation, label = "longitude", color = "red")
ax2.set_ylabel("correlation coefficient")
ax2.set_xlabel("time steps back")

plt.savefig(r"plots\cross_correlation_tornadoes.png")
plt.show()