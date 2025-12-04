import csv
import matplotlib.pyplot as plt

input_file = r"Data\cleaned\clean_tornado_tx_1970_2021.csv"


# all daat in easier to work with lists
year_list = []
month_list = []
day_list = []
mag_list = []
inj_list = []
fat_list = []
slat_list = []
slon_list = []
elat_list = []
elon_list = []
len_list = []
wid_list = []

with open(input_file, newline="", encoding="utf-8") as file:
    reader = csv.DictReader(file)

    for row in reader:
        year_list.append(row.get("yr"))
        month_list.append(row.get("mo"))
        day_list.append(row.get("dy"))
        mag_list.append(row.get("mag"))
        inj_list.append(row.get("inj"))
        fat_list.append(row.get("fat"))
        slat_list.append(row.get("slat"))
        slon_list.append(row.get("slon"))
        elat_list.append(row.get("elat"))
        elon_list.append(row.get("elon"))
        len_list.append(row.get("len"))
        wid_list.append(row.get("wid"))

# plot some things to give us an overview of the data



def month_plot(month_list):
    for i in range(len(month_list)):
        month_list[i] = int(month_list[i])

    plt.hist(month_list, bins = [i - 0.5 for i in range(1, 14)], label = "month", edgecolor="k")
    plt.xticks(range(1, 13))
    plt.xlabel("month")
    plt.ylabel("freq")
    plt.legend(loc = "best")
    plt.show()

def year_plot(year_list):
    for i in range(len(year_list)):
        year_list[i] = int(year_list[i])

    plt.hist(year_list, bins = [i - 0.5 for i in range(1950, 2022)], label = "year", edgecolor = "k")
    plt.xticks(range(1950, 2022, 5), rotation = 45)
    plt.xlabel("year")
    plt.ylabel("freq")
    plt.legend(loc = "best")
    plt.show()

def magnitude_plot(mag_list):
    for i in range(len(mag_list)):
        mag_list[i] = int(mag_list[i])

    plt.hist(mag_list, bins = [i-0.5 for i in range(1, 6)], label = "magnitude", edgecolor = "k")
    plt.xticks(range(1, 5))
    plt.xlabel("magnitude")
    plt.ylabel("freq")
    plt.legend(loc = "best")
    plt.show()

def arrow(slat_list, slon_list, elat_list, elon_list):
    # arrow for each tornado (lat = y, lon = x)
    for i in range(len(slat_list)):
        slat = float(slat_list[i])
        slon = float(slon_list[i])
        elat = float(elat_list[i])
        elon = float(elon_list[i])
        # exclude entries with unknown locations
        plt.plot(slon, slat, '.', color = "green")
        if elon != 0.0 and elat != 0.0 :
            plt.plot(elon, elat, '.', color = "red")
            plt.plot([slon, elon], [slat, elat], color = "black")
    plt.show()

def len_mag_scatter(len_list, mag_list):
    for i in range(len(len_list)):
        len_list[i] = float(len_list[i])
        mag_list[i] = int(mag_list[i])
    plt.scatter(mag_list, len_list, alpha = 0.25)
    plt.xlabel("magnitude")
    plt.ylabel("length (miles)")
    plt.show()

def len_wid_scatter(len_list, wid_list):
    for i in range(len(len_list)):
        len_list[i] = float(len_list[i])
        wid_list[i] = int(wid_list[i])
    plt.scatter(len_list, wid_list, alpha = 0.25)
    plt.xlabel("lentgh (miles)")
    plt.ylabel("width (yards)")
    plt.xscale("log")
    plt.yscale("log")
    plt.show()

#magnitude_plot(mag_list)
arrow(slat_list, slon_list, elat_list, elon_list)
#year_plot(year_list)
#month_plot(month_list)
#len_mag_scatter(len_list, mag_list)
#len_wid_scatter(len_list, wid_list)