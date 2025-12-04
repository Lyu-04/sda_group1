# SDA Group 1 - Tornado Analysis (Texas, 1950-2021)

This project analyzes tornadoes in Texas from 1950-2021, focusing on:

- Spatial shifts in the **center of mass** of tornado occurrence over time
- Temporal trends in **magnitude**, **path length**, and **path width**
- Descriptive distributions (like monthly frequency) and spatial maps

## Data overview

- **Raw source**: `Data/raw/us_tornado_dataset_1950_2021.csv`
- **Cleaned Texas subset**: `Data/cleaned/`
- **Current final analysis dataset (tornado + weather match)**: `Data/final/final_data1.csv`
- **Processed summaries** (e.g. center of mass by year/period): `Data/processed/`
- **Results tables** (regression summaries): `Data/results/`

Data-prep scripts (run from the `Data/` folder):

- `tornado_cleaner.py` – filters the US dataset to Texas (1970-2021) and removes outliers
- `CIproject_combining_datasets.py` – matches tornado events to gridded weather data and writes `final/final_data1.csv`

## Main analysis scripts

All analysis/plotting scripts live in `scripts/`. Run them from inside `scripts/`:

```bash
cd scripts/
python3 <filename>.py
```

Some important analyses:

- `tornado_center_of_mass_shift.py`
  - Computes yearly, 5‑year, and decadal centers of mass for tornado locations
  - Fits simple linear regressions to test for **latitudinal and longitudinal shifts over time**
  - Outputs processed center-of-mass (COM) tables to `Data/processed/` and summary stats to `Data/results/center_of_mass_shift_summary.csv`

- `tornado_path_mag_trends.py`
  - Computes yearly averages of **magnitude**, **path length**, and **path width**
  - Fits linear trends over time and saves summary statistics to `Data/results/tornado_path_shift_summary.csv`

- `linear_assumption_tests.py`  
  - Provides diagnostics used by the main scripts to check linear model assumptions

## Plots

Generated figures are saved into subfolders under `plots/`:

- `plots/trends/` – time‑trend figures (center of mass, magnitude, length, width)
- `plots/maps/` – spatial maps of tornado locations and center-of-mass trajectories
  - `tornado_map_magnitude.py` – map colored by tornado magnitude
  - `tornado_map_size.py` – map colored by tornado width
  - `tornado_center_of_mass_shift.py` – saves center-of-mass trajectory maps here
- `plots/distributions/` – distribution-style plots (e.g. `monthly_tornado_histogram.py`)
- `plots/correlations/` – correlation matrices from `correlation_matrix.py` and `correlation_matrix_condensed.py`
- `plots/residuals/` – regression diagnostic plots from the linear assumption tests

## Requirements

- pandas
- matplotlib
- numpy
- scipy
- seaborn

### small note

For the plots of the trajectories of the center of mass of tornadoes, when running `tornado_center_of_mass_shift.py` yourself, you can zoom in using the lens to get a better, more readable view of the trajectories.
