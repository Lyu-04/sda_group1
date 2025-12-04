
import pandas as pd

# ============================================================
# 1. LOAD TORNADO DATA
# ============================================================

# Tornado data (full period)
dt_raw = pd.read_csv("cleaned/clean_tornado_tx_1950_2021.csv", parse_dates=["date"])

# Keep only events from 1986 onward
dt = dt_raw[dt_raw["yr"] >= 1986].copy()

# Normalize date (remove time, keep only yyyy-mm-dd)
dt["date"] = dt["date"].dt.normalize()

# Get unique dates we care about
valid_dates = set(dt["date"].unique())

print(f"Tornado events kept: {len(dt)}")
print(f"Unique tornado dates: {len(valid_dates)}")

# ============================================================
# 2. LOAD GRID DATA IN CHUNKS (TO AVOID MEMORY ERROR)
# ============================================================

grid_file = "grid_all_vars.csv"

# All numeric weather variables as float32 to save RAM
float_cols = [
    "latitude", "longitude",
    "u10", "v10",
    "d2m", "t2m",
    "msl", "tcc",
    "tp", "e", "pev"
]

dtype_map = {col: "float32" for col in float_cols}

chunks = []
chunk_index = 0

for chunk in pd.read_csv(
    grid_file,
    parse_dates=["time"],
    chunksize=1_000_000,   # adjust if needed based on RAM
    dtype=dtype_map
):
    chunk_index += 1
    print(f"Processing chunk {chunk_index}...")

    # Drop metadata columns if present
    drop_cols = ["number", "step", "surface", "valid_time"]
    chunk = chunk.drop(columns=[c for c in drop_cols if c in chunk.columns], errors="ignore")

    # Add date-only column from time
    chunk["date"] = chunk["time"].dt.normalize()

    # Keep only rows whose date appears in tornado data
    chunk = chunk[chunk["date"].isin(valid_dates)]

    # If nothing left in this chunk, skip
    if chunk.empty:
        continue

    chunks.append(chunk)

# Concatenate all filtered chunks
if chunks:
    df = pd.concat(chunks, ignore_index=True)
else:
    raise RuntimeError("No matching dates between grid_all_vars.csv and tornado dataset.")

print(f"Grid rows after filtering by tornado dates: {len(df)}")
print("Grid columns:", df.columns.tolist())

# We no longer need the full timestamp column
df = df.drop(columns=["time"], errors="ignore")

# ============================================================
# 3. MATCH EACH TORNADO EVENT TO NEAREST GRID POINT (BY LAT/LON)
# ============================================================

# Give each event a unique id
dt = dt.reset_index().rename(columns={"index": "event_id"})

# Merge on date → all combinations of grid points & events for same day
print("Merging grid and tornado data on date...")
merged = df.merge(dt, on="date", how="inner")

print(f"Rows after date merge (all combos for same day): {len(merged)}")

# Squared distance in (lat, lon) space
merged["dist2"] = (
    (merged["latitude"] - merged["slat"])**2 +
    (merged["longitude"] - merged["slon"])**2
)

# For each event, keep only the closest grid point
print("Selecting nearest grid point for each tornado event...")
idx = merged.groupby("event_id")["dist2"].idxmin()
nearest_matches = merged.loc[idx].reset_index(drop=True)

print(f"Final matched rows (one per tornado event): {len(nearest_matches)}")

# ============================================================
# 4. SAVE FINAL RESULT
# ============================================================

output_file = "final_data1.csv"
nearest_matches.to_csv(output_file, index=False)

print(f"Done. Saved {output_file} with {len(nearest_matches)} rows.")
