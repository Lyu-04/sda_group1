# SDA Group 1 - Tornado Analysis (Texas, 1950-2021)

This project analyzes tornadoes in Texas from 1950-2021, focusing on:

- Spatial shifts in the **center of mass** of tornado occurrence over time
- Temporal trends in **magnitude**, **path length**, and **path width**
- Descriptive distributions (like monthly frequency) and spatial maps

## Data overview
- The weather grid  dataset( not fitered by taking only days with tornadoes) (`grid_all_vars.csv`, ~5 GB) is not stored in this repository due to GitHub size limitations.
  The dataset is available at: https://1drv.ms/f/c/e4eaff840ac28774/IgDnssY3ichfRKtKSv3cXirpAWiEJBxR0BiOC1g1OGcd3uc?e=Acvr9b
- **Raw source**: `Data/raw/us_tornado_dataset_1950_2021.csv`
- **Cleaned Texas subset**: `Data/cleaned/`
- **Current final analysis dataset (tornado + weather match)**: `Data/final/final_data1.csv`
- **Processed summaries** (e.g. center of mass by year/period): `Data/processed/`
- **Results tables** (regression summaries): `Data/results/`

  Data-prep scripts (run from the `Data/` folder):

- `tornado_cleaner.py` - filters the US dataset to Texas (1970-2021) and removes outliers
- `CIproject_combining_datasets.py` - matches tornado events to gridded weather data and writes `final/final_data1.csv`

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
- `scripts/regresion anlysis` - where is 3 files, main one 'regression_tornado.py' is there we tried to forecast tornadoes, calculated probabilities of tornado happening based on weather data, calculated expected fataliies based on weather data. Other two files are for tuning parameters.
## Plots

Generated figures are saved into subfolders under `plots/`:

- `plots/trends/` - time‑trend figures (center of mass, magnitude, length, width)
- `plots/maps/` - spatial maps of tornado locations and center-of-mass trajectories
  - `tornado_map_magnitude.py` - map colored by tornado magnitude
  - `tornado_map_size.py` - map colored by tornado width
  - `tornado_center_of_mass_shift.py` - saves center-of-mass trajectory maps here
- `plots/distributions/` - distribution-style plots (e.g. `monthly_tornado_histogram.py`)
- `plots/correlations/` - correlation matrices from `correlation_matrix.py` and `correlation_matrix_condensed.py`
- `plots/residuals/` - regression diagnostic plots from the linear assumption tests
- `plots/regression_analysis/` maps to show our predictions and variables importance: 2 heatmaps(tornado mean probabilities per grid, maximum fatalities per grid through 2015-2021), feature importance for XGBOOST model, daily expeccted fatalities through 2015-2021, average expected fatalities by month through 2015-2021.
## Requirements

- pandas
- matplotlib
- numpy
- scipy
- seaborn
- sklearn
- xarray

### small note

For the plots of the trajectories of the center of mass of tornadoes, when running `tornado_center_of_mass_shift.py` yourself, you can zoom in using the lens to get a better, more readable view of the trajectories.


### Use of AI
Some parts of code were written with the help of AI, lets go through them:
1) converting data file from .grib to .csv. (sqitchgribtocsv.py in data folder)

For the regression:
2) we haven't worked with XGBOOST and forest model for a while so helped   a little bit to code the model and to use the new libraries.
3) in the file regression_tornado.py we were overestimating the expected fatalities so with the advice from google an AI we decided to calibrate the probabilities calculated by XGBOOST (lines 273-283).
4) Wrote a code to tune the hyperparamaters of our models, never done this before so tune_models.py mostly written by AI.
5) Some code were just hard to write so for the time puposes and not to get stuck, AI helped in some lines( for example taking data in chunks because dataset was too large) but nothing significant.