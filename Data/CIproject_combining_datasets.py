
import pandas as pd

# 1. Load data
# Weather / coordinates data
df = pd.read_csv("output_with_coords.csv", parse_dates=["time"])

# Tornado data (full period)
dt_raw = pd.read_csv("clean_tornado_tx_1950_2021.csv", parse_dates=["date"])

# 2. Clean coordinate dataset
cols_to_drop = ["number", "step", "surface", "valid_time"]
df = df.drop(columns=cols_to_drop, errors="ignore")

# keep only whole-degree lat/lon to narrow the dataset
df = df[
    (df["latitude"] % 1 == 0) &
    (df["longitude"] % 1 == 0)
].copy()

# Add a date column (date only) from time
df["date"] = df["time"].dt.normalize()

# 3. Clean tornado dataset
# Keep only events from 1970 onward
dt = dt_raw[dt_raw["yr"] >= 1970].copy()

# Make sure dt["date"] is datetime and normalized (no time part)
dt["date"] = pd.to_datetime(dt["date"]).dt.normalize()


# 4. Filter df by dates that appear in tornado data
valid_dates = dt["date"].unique()
df = df[df["date"].isin(valid_dates)].copy()

# We no longer need the full timestamp column
df = df.drop(columns=["time"], errors="ignore")

# 5. Match each tornado event to nearest grid point (lat/lon)
# Give each event a unique id
dt = dt.reset_index().rename(columns={"index": "event_id"})

# Merge on date → all combinations of grid points & events for same day
merged = df.merge(dt, on="date", how="inner")

# Squared distance in (lat, lon) space
merged["dist2"] = (
    (merged["latitude"] - merged["slat"])**2 +
    (merged["longitude"] - merged["slon"])**2
)

# For each event, keep only the closest grid point
idx = merged.groupby("event_id")["dist2"].idxmin()
nearest_matches = merged.loc[idx].reset_index(drop=True)

# 6. Save final result
nearest_matches.to_csv("final_data.csv", index=False)

print("Done. Saved final_data.csv with", len(nearest_matches), "rows.")
