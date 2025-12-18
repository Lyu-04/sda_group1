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

## Data folder

The `Data/` folder contains all datasets and data-preparation scripts for the project. It includes:

- **Raw data** (`raw/`): Original U.S. tornado dataset (1950-2021).
- **Cleaned data** (`cleaned/`): Texas-only subsets with quality control applied.
- **Final dataset** (`final/`): Tornado events matched with gridded weather variables (the primary analysis dataset).
- **Processed summaries** (`processed/`): Aggregated tables.
- **Results** (`results/`): Machine-readable outputs from analyses.
- **Data preparation scripts**: Scripts to clean tornado data and match it with weather grids.

For detailed information about each subfolder, scripts, and data files, see `Data/README.md`.

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
