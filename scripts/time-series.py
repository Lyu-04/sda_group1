import pandas
import csv
import matplotlib.pyplot as plt

input_file = r"Data\final_data.csv"
df = pd.read_csv(input_file)

lat_list = []
lat_min_1 = []
lon_list = []
lon_min_1 = []