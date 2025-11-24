# Joost Luijten
# 24-11-2025
import csv

# data file to clean and cleaned file
input_file = r"Data\us_tornado_dataset_1950_2021.csv"
output_file = r"Data\clean_tornado_tx_1950_2021.csv"


with open(input_file, newline="", encoding="utf-8") as infile, \
     open(output_file, "w", newline="", encoding="utf-8") as outfile:

    reader = csv.DictReader(infile)
    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)

    writer.writeheader()

    #select only data where state == texas and write to cleaned file
    for row in reader:
        if row.get("st") == "TX":
            writer.writerow(row)