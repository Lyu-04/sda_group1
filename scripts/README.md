# Scripts Overview

Quick guide to what each script does, how to run it, and required packages.

## Setup
- Run commands from `scripts/` with

```bash
python3 <filename>.py
```

so the relative `../Data` and `../plots` paths resolve.

## Data locations
- `../Data/final/final_data1.csv`
- `../Data/cleaned/clean_tornado_tx_1950_2021.csv`
- `../Data/cleaned/clean_tornado_tx_1970_2021.csv`

## Scripts
- `correlation_matrix.py` - Pearson correlation of all numeric columns in `final_data1.csv`. Saves heatmap to `../plots/correlations/correlation_matrix.png`.

- `correlation_matrix_condensed.py` - Drops non-meteorological columns first, then builds a correlation heatmap. Output `../plots/correlations/correlation_matrix_condensed.png`.

- `cross_correlation.py` - calculates the Pearson correlation for lat and lon to lat - x and lon - x. Saves plot of 40 steps back to `../plots/cross_correlation_tornadoes.png`

- `linear_assumption_tests.py` - Helper functions to plot residuals vs year for regression metrics. Used by other scripts. Usage: import `test_linearity_all_metrics(...)` or `plot_residuals_vs_year(...)`.

- `monthly_tornado_histogram.py` - Histogram of Texas tornado counts by month (1950-2021). Saves `../plots/distributions/monthly_tornado_histogram.png` and prints totals.

- `storm_length.py` - Script runs through all tornado's and groups together storms on the same day and location. saves this to a similar csv to the tornado data but with the average data for the storm, saves at `../Data/processed/unique_storms.csv`

- `time-series.py` - plots lat and lon for tornado occurances against previous data points. also does this for the grouped storm data. outputs two plots. `../plots/tornado_series.png` for tornadoes and `../plots/tornado_time_series.png` for tornadoes and storms combined. Prints r and p values for the storm data. 

- `tornado_center_of_mass_shift.py` - Tracks and visualizes the spatial center of mass of tornadoes over decades/5-year periods/years. Saves multiple maps and trend plots to `../plots` and CSV summaries to `../Data/processed` and `../Data/results`. Uses SciPy for regressions and `linear_assumption_tests` for residual checks.

- `tornado_map_magnitude.py` - Maps tornado locations colored/scaled by magnitude (filters unknown magnitudes). Saves `../plots/maps/tornado_map_magnitude.png` and prints magnitude counts.

- `tornado_map_size.py` - Maps tornado locations colored/scaled by path width (filters width ≥ 30 yards). Saves `../plots/maps/tornado_map_size.png` and prints width stats.

- `tornado_path_mag_trends.py` - Computes yearly mean magnitude/length/width trends with linear fits. Saves trend plots to `../plots/trends/`, summary CSV to `../Data/results/tornado_path_shift_summary.csv`, and residual plots via `linear_assumption_tests`.

- `tornado_visualiser.py` - script contains functions to display values from the tornado data
  
- `inj_fat_correlation_data_prep.py` - creating target variables, scaling features, and handling class weights
  
- `inj_fat_unweighted_model.py` - unweighted logistic regression to model injury and fatality predictions
  
- `inj_fat_weighted_model.py` - logistic regression models with weighted classes
  
-  `fat_inj_visualization.py` - create plots related to coefficient comparisons, distributions and odds ratio comparisons
  
-  `injury_fatality_correlation_analysis.py` - main script for analyzing association between various predictors and injury/fatality, integrates all previous int_fat files on data preparation, modeling, and visualization into a comprehensive pipeline. Saves `boxplot_by_fatality_target.png`, `boxplot_by_injury_target.png`, `class_imbalance.png`, `coefficient_comparison.png`, `odd_ration_comparison.png`, `pct_change_odds_ratio.png` to `../plots/class_imbalance/`

## Requirements
- Python 3.x
### Packages
- pandas
- matplotlib
- numpy
- scipy
- seaborn
