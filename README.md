## SDA Group 1 – Tornado Analysis in Texas (1950-2021)

This repository contains an end‑to‑end analysis of tornadoes in Texas from 1950-2021.  
We investigate **where**, **how strong**, and **how large** tornadoes have been over time, with a focus on:

- **Spatial shifts** in the center of mass of tornado occurrence
- **Temporal trends** in magnitude, path length, and path width
- **Descriptive patterns** in time (e.g. monthly frequency) and space (maps)
- **Relationships with environmental variables** (correlations and regression-style summaries)

Our end goal is to try and forecast tornadoes given our research and analyses on the above topics.

For a detailed description of a given folder, see their respective `README.md`, located in their own folders.

---

## Data and processing pipeline

- **Raw source data**
  - `Data/raw/us_tornado_dataset_1950_2021.csv`: CONUS tornado records (SPC-like format).

- **Cleaning and Texas subset**
  - `Data/cleaned/clean_tornado_tx_1950_2021.csv`
  - `Data/cleaned/clean_tornado_tx_1970_2021.csv`
  - Generated via scripts in `Data/` (e.g. `tornado_cleaner.py`).

- **Final analysis dataset**
  - `Data/final/final_data1.csv`: matched tornado events with gridded weather variables, used by most scripts in `scripts/`.

- **Processed summaries**
  - `Data/processed/` contains aggregated tables such as:
    - Center of mass by **year**, **5‑year period**, and **decade**
    - Aggregated storm‑level datasets (e.g. `unique_storms.csv`)

- **Results tables**
  - `Data/results/` holds machine‑readable outputs of the main analyses, e.g.:
    - `center_of_mass_shift_summary.csv`
    - `tornado_path_shift_summary.csv`
    - Class‑imbalance and odds‑ratio summaries for injuries/fatalities in `Data/results/class_imbalance/`

---

## Analyses performed

All analysis and visualization code is in `scripts/` (see its `README.md` for script‑by‑script details).  
At a high level, we perform:

- **Center of mass and spatial shifts**
- **Trends in magnitude and path geometry**
- **Descriptive distributions and time series**
- **Correlation and class imbalance analyses**

---

## Figures and outputs

All plots are stored under `plots/`, with similar plots being located in their own subfolders:

- **Trends** (`plots/trends/`): center‑of‑mass trajectories and trends in magnitude, length, and width.
- **Maps** (`plots/maps/`): spatial distributions of tornado locations, sizes, magnitudes, and center‑of‑mass paths.
- **Distributions** (`plots/distributions/`): e.g. monthly tornado counts.
- **Correlations** (`plots/correlations/`): full and condensed correlation matrices.
- **Residuals/diagnostics** (`plots/residuals/`): linear model assumption checks.
- **Other**: additional time‑series and storm‑length histograms at the top level of `plots/`.

For interpretation notes and textual conclusions, see the `conclusions/` folder.
