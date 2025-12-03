# SDA Group 1 - Tornado Analysis

Analysis of tornado data in Texas (1950-2021) to predict tornado size and magnitude.

## Dataset

- **Data**: `Data/clean_tornado_tx_1950_2021.csv` - Texas tornado records (1950-2021)
- **Source**: Filtered from US tornado dataset, containing 9,149 tornado events

## Plotting Scripts

All plotting scripts are located in `scripts/`. To run:

```bash
cd scripts/
python3 <filename>.py
```

### Available Plots

- `monthly_tornado_histogram.py` - Monthly tornado frequency distribution
- `tornado_map_magnitude.py` - Geographic map colored by tornado magnitude
- `tornado_map_size.py` - Geographic map colored by tornado width
- `tornado_center_of_mass_shift.py` - **Center of mass shift analysis** - Analyzes whether the geographic center of tornadoes in Texas is shifting over time (1950-2021), with statistical significance testing
- `tornado_path_mag_trends.py` - **Path and magnitude shift analysis** - Analyzes whether the path and magnitude of tornadoes in Texas is shifting over time (1950-2021)

Generated plots are saved to `plots/`.

## Requirements

- pandas
- matplotlib
- numpy
- scipy
- seaborn

### small note

For the plots of the trajectories of the center of mass of tornadoes, when running `tornado_center_of_mass_shift.py` yourself, you can zoom in using the lens to get a better, more readable view of the trajectories.
