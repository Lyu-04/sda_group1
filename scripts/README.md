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

- `linear_assumption_tests.py` - Helper functions to plot residuals vs year for regression metrics. Used by other scripts. Usage: import `test_linearity_all_metrics(...)` or `plot_residuals_vs_year(...)`.

- `monthly_tornado_histogram.py` - Histogram of Texas tornado counts by month (1950-2021). Saves `../plots/distributions/monthly_tornado_histogram.png` and prints totals.

- `storm_length.py` - [EXPLANATION!!!]

- `time-series.py` - [EXPLANATION!!!]

- `tornado_center_of_mass_shift.py` - Tracks and visualizes the spatial center of mass of tornadoes over decades/5-year periods/years. Saves multiple maps and trend plots to `../plots` and CSV summaries to `../Data/processed` and `../Data/results`. Uses SciPy for regressions and `linear_assumption_tests` for residual checks.

- `tornado_map_magnitude.py` - Maps tornado locations colored/scaled by magnitude (filters unknown magnitudes). Saves `../plots/maps/tornado_map_magnitude.png` and prints magnitude counts.

- `tornado_map_size.py` - Maps tornado locations colored/scaled by path width (filters width ≥ 30 yards). Saves `../plots/maps/tornado_map_size.png` and prints width stats.

- `tornado_path_mag_trends.py` - Computes yearly mean magnitude/length/width trends with linear fits. Saves trend plots to `../plots/trends/`, summary CSV to `../Data/results/tornado_path_shift_summary.csv`, and residual plots via `linear_assumption_tests`.

- `tornado_visualiser.py` - [EXPLANATION!!!]

## Requirements
- Python 3.x
### Packages
- pandas
- matplotlib
- numpy
- scipy
- seaborn
