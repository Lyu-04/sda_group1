## Data Directory Overview

This `Data/` folder contains all datasets and data-preparation scripts used in the Texas tornado analysis.  
It holds the **raw tornado records**, intermediate **cleaned / processed tables**, the **final matched tornado-weather dataset**, and **results tables** produced by the analysis scripts.

Where relevant, run the scripts in this folder from inside `Data/` so the relative paths resolve correctly.

---

## Folder structure

### `raw/`
- **Purpose**: Original input files as obtained from external sources.
- **Contents**:
  - `us_tornado_dataset_1950_2021.csv`: Full U.S. tornado dataset (1950-2021), including all states.
- **Notes**: This file is **not** modified in-place. All cleaning and subsetting is written to `cleaned/`.

### `cleaned/`
- **Purpose**: Texas-only and quality-controlled tornado event tables derived from the raw dataset.
- **Contents** (current):
  - `clean_tornado_tx_1950_2021.csv`: Cleaned subset of tornadoes in **Texas** for the full 1950-2021 period.
  - `clean_tornado_tx_1970_2021.csv`: Texas subset from 1970 onward, with a small number of outlier points removed.
- **Used script**:
  - `tornado_cleaner.py`
    - Reads `raw/us_tornado_dataset_1950_2021.csv`.
    - Filters rows where `st == "TX"` and `yr > 1969` (i.e. Texas tornadoes from 1970 onward).
    - Removes a few spatial outliers based on `slat`/`slon` bounds.
    - Writes the cleaned subset to `cleaned/clean_tornado_tx_1970_2021.csv`.

### `final/`
- **Purpose**: Final, analysis-ready dataset combining tornado events with gridded weather variables.
- **Contents**:
  - `final_data1.csv`: Each row corresponds to a **tornado event**, enriched with:
    - Tornado attributes.
    - Weather variables from the nearest ERA-style grid point on the same date.
- **Used script**:
  - `CIproject_combining_datasets.py`
    - Loads `cleaned/clean_tornado_tx_1950_2021.csv` and keeps events from **1986 onward**.
    - Loads a large gridded weather file `grid_all_vars.csv` in chunks to avoid memory issues.
    - Drops metadata columns (if present) and builds a date-only column from `time`.
    - Filters grid rows to those dates that actually appear in the tornado data.
    - For each tornado event, finds the **nearest grid point** (minimum squared distance in latitude/longitude) on that date.
    - Saves the matched, one-row-per-event dataset to `final/final_data1.csv`.

### `processed/`
- **Purpose**: Derived data products and aggregated tables used by downstream analysis scripts.
- **Typical contents**:
  - `center_of_mass_by_year.csv`: Yearly center of mass of tornado locations (lat/lon) in Texas.
  - `center_of_mass_by_5yr.csv`: 5-year aggregated center of mass.
  - `center_of_mass_by_decade.csv`: Decadal center of mass.
  - `unique_storms.csv`: Storm-level table where multiple tornadoes belonging to the same storm (same date/location) are grouped and summarized.
- **Produced by**: Various scripts in `scripts/`, with the ability to be used by other analysis and plotting scripts.

### `derived/`
- **Purpose**: Intermediate exploratory or transformed datasets used for specific analyses.
- **Example contents**:
  - `filtered.csv`: Filtered subset of tornado or matched data for a particular task.
  - `final_data.csv`: Alternative or intermediate version of the final dataset used during development.
- **Notes**: These files may reflect earlier stages or side analyses. They are kept for reproducibility and quick re-use.

### `results/`
- **Purpose**: Machine-readable outputs summarizing important analysis results.
- **Top-level contents**:
  - `center_of_mass_shift_summary.csv`: Summary statistics and trend estimates for how the tornado center of mass shifts over time.
  - `tornado_path_shift_summary.csv`: Summary of temporal trends in path **magnitude**, **length**, and **width**.

#### `results/class_imbalance/`
- **Purpose**: Tables related to injury/fatality class-imbalance analyses.
- **Contents** (examples):
  - `coefficient_comparison_fatality.csv` / `coefficient_comparison_injury.csv`: Regression-style coefficient summaries comparing models with different weighting or setups.
  - `odds_ratio_comparison_fatality.csv` / `odds_ratio_comparison_injury.csv`: Odds ratios and confidence intervals for injury/fatality outcomes under different conditions.
  - `pct_change_fatality.csv` / `pct_change_injury.csv`: Percentage change metrics for selected variables or model outputs.
  - `significant_changes_fatality.csv` / `significant_changes_injury.csv`: Filtered tables of statistically significant changes.
- **Produced by**: The class-imbalance / injury-fatality scripts in `scripts/` (see `scripts/README.md`).

---

## Data dictionary and variable explanations

For information about every weather and tornado variable used, refer to `Weather_variables_explanation.md`.

---

## Usage notes

- Run `tornado_cleaner.py` and `CIproject_combining_datasets.py` from inside the `Data/` directory:

  ```bash
  cd Data/
  python3 tornado_cleaner.py
  python3 CIproject_combining_datasets.py
  ```
