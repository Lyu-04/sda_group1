import pandas as pd
import numpy as np
'''
# Weather grid
grid_file = "grid_all_vars.csv"
df = pd.read_csv(grid_file, parse_dates=["time"])

# Add date without time
df["date"] = df["time"].dt.normalize()


# Tornado data
tornado_file = "clean_tornado_tx_1970_2021.csv" 
dt = pd.read_csv(tornado_file, parse_dates=["date"])

dt = dt[dt["yr"] >= 1986].copy()

# Normalize tornado dates
dt["date"] = dt["date"].dt.normalize()

print("Tornado rows kept:", len(dt))


# We only need these columns from tornado data for matching
needed_cols = ["date", "slat", "slon"]
for col in needed_cols:
    if col not in dt.columns:
        raise ValueError(f"Column '{col}' not found in tornado file. Available: {dt.columns.tolist()}")

dt_match = dt[needed_cols].copy()

# Add event_id so each tornado is unique
dt_match = dt_match.reset_index().rename(columns={"index": "event_id"})

# Merge on date: all grid points for the same day as each tornado
merged = df.merge(dt_match, on="date", how="left")

print("Rows after grid–tornado date merge:", len(merged))


#FIND NEAREST GRID CELL FOR EACH TORNADO

has_tornado = merged["slat"].notna() & merged["slon"].notna()
merged_torn = merged[has_tornado].copy()

if merged_torn.empty:
    raise RuntimeError("No overlapping dates between grid and tornado data for the chosen period.")

# Squared distance in (lat, lon)
merged_torn["dist2"] = (
    (merged_torn["latitude"] - merged_torn["slat"])**2 +
    (merged_torn["longitude"] - merged_torn["slon"])**2
)

# For each tornado event, keep only the closest grid point
idx = merged_torn.groupby("event_id")["dist2"].idxmin()
nearest = merged_torn.loc[idx, ["date", "latitude", "longitude"]].copy()

# Mark these as tornado=1
nearest["tornado"] = 1

print("Nearest grid cells (one per tornado):", len(nearest))

# Remove duplicates just in case multiple tornadoes share a cell/date
nearest = nearest.drop_duplicates(subset=["date", "latitude", "longitude"])

#BUILD FINAL LOGISTIC DATASET (FULL GRID + LABEL)
# Start with full grid (drop raw time column if not needed)
df_logit = df.drop(columns=["time"], errors="ignore").copy()

# Default label: no tornado
df_logit["tornado"] = 0

# Merge in tornado=1 flags
df_logit = df_logit.merge(
    nearest,
    on=["date", "latitude", "longitude"],
    how="left",
    suffixes=("", "_from_nearest")
)

# If tornado_from_nearest is 1, overwrite tornado=1
df_logit.loc[df_logit["tornado_from_nearest"] == 1, "tornado"] = 1

# Clean up helper column
df_logit = df_logit.drop(columns=["tornado_from_nearest"], errors="ignore")

# Ensure binary int label
df_logit["tornado"] = df_logit["tornado"].fillna(0).astype(int)

print("Final logistic dataset shape:", df_logit.shape)
print("Class counts:\n", df_logit["tornado"].value_counts())

#save file
out_file = "logit_data.csv"
df_logit.to_csv(out_file, index=False)
print(f"Saved logistic regression dataset to: {out_file}")

df = pd.read_csv("logit_data.csv")
df = df.drop(columns=["number", "step", "surface", "valid_time"], errors="ignore")
#df["wind_speed"] = np.sqrt(df["u10"]**2 + df["v10"]**2)
sample_rows = []
chunksize = 500_000

for chunk in pd.read_csv("logit_data.csv", chunksize=chunksize):
    # Keep all tornado=1 rows
    pos = chunk[chunk["tornado"] == 1]
    
    # Sample some tornado=0 rows
    neg = chunk[chunk["tornado"] == 0].sample(
        n=min(5000, (chunk["tornado"] == 0).sum()),  # adjust 5000 if you want more
        random_state=42
    )
    
    sample_rows.append(pd.concat([pos, neg], ignore_index=True))

df_small = pd.concat(sample_rows, ignore_index=True)

df_small = df_small.sample(frac=1, random_state=42).reset_index(drop=True)

print(df_small["tornado"].value_counts())
df_small.to_csv("logit_data_small.csv", index=False)







grid_file = "grid_all_vars.csv"
df = pd.read_csv(grid_file, parse_dates=["time"])

# Add date without time
df["date"] = df["time"].dt.normalize()

# Precompute wind speed
df["wind_speed"] = np.sqrt(df["u10"]**2 + df["v10"]**2)

# Group grid rows by date for fast lookup
grid_by_date = {d: subdf for d, subdf in df.groupby("date")}



# CLEAN TORNADO DATA
tornado_file = "clean_tornado_tx_1970_2021.csv"
dt = pd.read_csv(tornado_file, parse_dates=["date"])

#only years where grid data exists
dt = dt[dt["yr"] >= 1986].copy()
dt["date"] = dt["date"].dt.normalize()

print("Tornado rows kept:", len(dt))

# Tornado variables we keep
tornado_cols = [
    "date", "yr", "mo", "dy", "st", "mag",
    "inj", "fat", "slat", "slon", "elat", "elon",
    "len", "wid"
]

# Keep only these columns
dt = dt[tornado_cols].copy()

#match each tornado to nearest grid point
rows = []

for idx, ev in dt.iterrows():
    d = ev["date"]

    # Ensure grid data exists for this date
    if d not in grid_by_date:
        continue

    grid_day = grid_by_date[d]
    slat, slon = ev["slat"], ev["slon"]

    # Compute squared distance to all gridpoints that day
    dist2 = (grid_day["latitude"] - slat)**2 + (grid_day["longitude"] - slon)**2
    nearest_idx = dist2.idxmin()

    # Extract weather for the nearest grid cell
    weather = grid_day.loc[nearest_idx].copy()

    # Add all tornado metadata
    for col in tornado_cols:
        weather[col] = ev[col]

    rows.append(weather)


tornado_weather = pd.DataFrame(rows)

print("Final tornado_weather shape:", tornado_weather.shape)
print(tornado_weather.head())

tornado_weather.to_csv("tornado_weather.csv", index=False)
print("Saved tornado_weather.csv")
'''




import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    mean_absolute_error,
    r2_score
)

from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV

##The whole prediction


#datasets
WEATHER_FILE = "grid_all_vars.csv"
LOGIT_FILE = "logit_data_small.csv"        
TORNADO_WEATHER_FILE = "tornado_weather.csv" 
TORNADO_FILE = "clean_tornado_tx_1970_2021.csv"
RISK_OUT_FILE = "risk_2015_2020_hurdle.csv"

START_YEAR = 2015
END_YEAR = 2020

FEATURES = ["wind_speed", "d2m", "t2m", "msl", "tcc", "tp", "e", "pev"]

print("\n=== Training P(Tornado) model (XGBoost, calibrated) ===")

df_logit = pd.read_csv(LOGIT_FILE)

# Drop useless columns
df_logit = df_logit.drop(columns=["number", "step", "surface", "valid_time"], errors="ignore")


df_logit["wind_speed"] = np.sqrt(df_logit["u10"]**2 + df_logit["v10"]**2)

X_torn = df_logit[FEATURES]
y_torn = df_logit["tornado"]

# split into train / test
X_train_t, X_temp_t, y_train_t, y_temp_t = train_test_split(
    X_torn, y_torn, test_size=0.4, random_state=42, stratify=y_torn
)

X_calib_t, X_test_t, y_calib_t, y_test_t = train_test_split(
    X_temp_t, y_temp_t, test_size=0.5, random_state=42, stratify=y_temp_t
)
# 60% train, 20% calib, 20% test

# Compute class imbalance ratio from training set
neg, pos = (y_train_t == 0).sum(), (y_train_t == 1).sum()
scale_pos_weight = neg / pos
print("neg:", neg, "pos:", pos, "scale_pos_weight:", scale_pos_weight)

#base XGBoost model (your tuned params)
xgb_base = XGBClassifier(
    n_estimators=330,
    max_depth=7,
    learning_rate=0.021035,
    subsample=0.60823,
    min_child_weight=4,
    colsample_bytree=0.66239,
    objective="binary:logistic",
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

xgb_base.fit(X_train_t, y_train_t)

#calibrate probabilities on validation (calib) set
calib_model = CalibratedClassifierCV(
    xgb_base,        # estimator passed positionally
    method="isotonic",
    cv="prefit"
)

calib_model.fit(X_calib_t, y_calib_t)

# evaluate raw vs calibrated on TEST set 
y_prob_raw = xgb_base.predict_proba(X_test_t)[:, 1]
y_prob_cal = calib_model.predict_proba(X_test_t)[:, 1]

print("\nRaw XGBoost ROC-AUC:", roc_auc_score(y_test_t, y_prob_raw))
print("Calibrated XGBoost ROC-AUC:", roc_auc_score(y_test_t, y_prob_cal))
print("Mean raw p_tornado:", y_prob_raw.mean())
print("Mean calibrated p_tornado:", y_prob_cal.mean())


y_pred_cal = (y_prob_cal > 0.5).astype(int)

print("\nXGBoost Confusion Matrix (calibrated P(tornado)):")
print(confusion_matrix(y_test_t, y_pred_cal))
print("\nXGBoost Classification Report (calibrated):")
print(classification_report(y_test_t, y_pred_cal, digits=4))

xgb_model = calib_model

# TRAIN P(FATAL | TORNADO) MODEL FROM tornado_weather.csv
print("\n Training P(Fatal | Tornado) model")

tw = pd.read_csv(TORNADO_WEATHER_FILE)

# Only rows with valid fatalities
tw = tw[tw["fat"].notna()].copy()


tw["wind_speed"] = np.sqrt(tw["u10"]**2 + tw["v10"]**2)

tw["fatal_flag"] = (tw["fat"] > 0).astype(int)
print("Fatal vs non-fatal tornado counts:")
print(tw["fatal_flag"].value_counts())

X_flag = tw[FEATURES]
y_flag = tw["fatal_flag"]

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_flag, y_flag, test_size=0.25, random_state=42, stratify=y_flag
)

fatal_clf = RandomForestClassifier(
    n_estimators=684,
    max_depth=14,
    max_features=0.8,
    min_samples_leaf=1,
    min_samples_split=4,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

fatal_clf.fit(X_train_f, y_train_f)


y_pred_f = fatal_clf.predict(X_test_f)
y_prob_f = fatal_clf.predict_proba(X_test_f)[:, 1]

print("\n Fatal Tornado Classifier (P(fatal | tornado))")
print("Confusion matrix:")
print(confusion_matrix(y_test_f, y_pred_f))
print("\nClassification report:")
print(classification_report(y_test_f, y_pred_f, digits=4))
print("ROC-AUC:", roc_auc_score(y_test_f, y_prob_f))

# E[FAT | FATAL, TORNADO]
print("n\ Training model (E[fat | fatal, tornado]) ")

tw_fatal = tw[tw["fatal_flag"] == 1].copy()

X_sev = tw_fatal[FEATURES]
y_sev = np.log1p(tw_fatal["fat"])

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_sev, y_sev, test_size=0.25, random_state=42
)

rf_sev = RandomForestRegressor(
    n_estimators=387,
    max_depth=10,
    min_samples_leaf=2,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1
)

rf_sev.fit(X_train_s, y_train_s)

pred_log_s = rf_sev.predict(X_test_s)
y_true_s = np.expm1(y_test_s)
y_pred_s = np.expm1(pred_log_s)

print("\n Model (Given fatal tornado)")
print("MAE:", mean_absolute_error(y_true_s, y_pred_s))
print("R²:", r2_score(y_true_s, y_pred_s))

print("\n Example severity predictions:")
print(pd.DataFrame({"Actual": y_true_s[:10], "Predicted": y_pred_s[:10]}))


#APPLY HURDLE MODEL TO GRID DATA 2015–2020
print("\n=== Running hurdle risk model on 2015–2020 grid data ===")
#file is too big so we split it into chunks
chunksize = 100_000
first_chunk = True

for chunk in pd.read_csv(WEATHER_FILE, chunksize=chunksize, parse_dates=["time"]):
    # filter to 2015–2020
    mask = (chunk["time"] >= f"{START_YEAR}-01-01") & (chunk["time"] < f"{END_YEAR+1}-01-01")
    chunk = chunk[mask]
    if chunk.empty:
        continue

    # date
    chunk["date"] = chunk["time"].dt.normalize()

    chunk["wind_speed"] = np.sqrt(chunk["u10"]**2 + chunk["v10"]**2)

    X_future = chunk[FEATURES]

    # P(tornado)
    p_tornado = xgb_model.predict_proba(X_future)[:, 1]

    # P(fatal | tornado)
    p_fatal_given_tornado = fatal_clf.predict_proba(X_future)[:, 1]

    # E[fat | fatal, tornado]
    log_fat_sev = rf_sev.predict(X_future)
    e_fat_given_fatal = np.expm1(log_fat_sev)

    # Hurdle expected fatalities
    expected_fat = p_tornado * p_fatal_given_tornado * e_fat_given_fatal

    chunk["p_tornado"] = p_tornado
    chunk["p_fatal_given_tornado"] = p_fatal_given_tornado
    chunk["e_fat_given_fatal_tornado"] = e_fat_given_fatal
    chunk["expected_fatalities"] = expected_fat

    # write out
    if first_chunk:
        chunk.to_csv(RISK_OUT_FILE, index=False, mode="w")
        first_chunk = False
    else:
        chunk.to_csv(RISK_OUT_FILE, index=False, mode="a", header=False)

print("Saved hurdle risk results to:", RISK_OUT_FILE)

#YEARLY COMPARISON: MODEL VS REAL FATALITIES (2015–2020)
print("\n Comparing predicted vs real fatalities (per year)")

# REAL fatalities
real = pd.read_csv(TORNADO_FILE, parse_dates=["date"])
real = real[(real["yr"] >= START_YEAR) & (real["yr"] <= END_YEAR)]

real_yearly = real.groupby("yr")["fat"].sum().reset_index()
real_yearly.columns = ["year", "real_fatalities"]

# PREDICTED fatalities (using daily max risk as region summary)
pred = pd.read_csv(RISK_OUT_FILE, parse_dates=["time"])
pred["date"] = pred["time"].dt.date
# Sort by date and p_tornado (highest risk first)
pred_sorted = pred.sort_values(["date", "p_tornado"], ascending=[True, False])

K = 1

# For each date, take top K highest p_tornado cells
topK = (
    pred_sorted
    .groupby("date")
    .head(K)
    .copy()
)

# Daily expected fatalities 
daily = (
    topK
    .groupby("date")["expected_fatalities"]
    .mean()
    .reset_index()
)

# Add year column
daily["year"] = pd.to_datetime(daily["date"]).dt.year

pred_yearly = (
    daily
    .groupby("year")["expected_fatalities"]
    .sum()
    .reset_index()
)
pred_yearly.columns = ["year", "predicted_fatalities"]

print(pred_yearly)

compare = real_yearly.merge(pred_yearly, on="year", how="left")
print(compare)













####PLOTING
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates


df = pd.read_csv("risk_2015_2020_hurdle.csv", parse_dates=["time"])
df["date"] = df["time"].dt.date

#HEATMAP: MEAN TORNADO PROBABILITY (2015–2020)
risk_map = (
    df.groupby(["latitude", "longitude"])["p_tornado"]
      .mean()
      .reset_index()
)

heatmap_data = risk_map.pivot(
    index="latitude",
    columns="longitude",
    values="p_tornado"
)

plt.figure(figsize=(12, 7))

vmax = np.nanpercentile(heatmap_data.values, 99)

sns.heatmap(
    heatmap_data,
    cmap="hot",
    cbar_kws={"label": "Mean Tornado Probability"},
    vmax=vmax
)

plt.title("Heatmap of Tornado Probability (2015–2020)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# HEATMAP: TOTAL EXPECTED FATALITIES (2015–2020)
# Here we take maximum of each grid expected fatalities over 2015–2020
fatal_map = (
    df.groupby(["latitude", "longitude"])["expected_fatalities"]
      .max()
      .reset_index()
)

fatal_data = fatal_map.pivot(
    index="latitude",
    columns="longitude",
    values="expected_fatalities"
)

plt.figure(figsize=(12, 7))

vmax = np.nanpercentile(fatal_data.values, 99)

sns.heatmap(
    fatal_data,
    cmap="Reds",
    cbar_kws={"label": "Maximum Expected Fatalities"},
    vmax=vmax
)

plt.title("Maximum Expected Tornado Fatality Risk per Grid Cell (2015–2020)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


#DAILY EXPECTED FATALITIES (2015–2020)
df["time"] = pd.to_datetime(df["time"])
df["date"] = df["time"].dt.date

# sort highest-risk grids come first
df_sorted = df.sort_values(["date", "p_tornado"], ascending=[True, False])

K = 1  #number of top grids per day

# take top-K most dangerous grids per day
topK = df_sorted.groupby("date").head(K)

# sum expected fatalities for these grids only
daily = topK.groupby("date")["expected_fatalities"].sum().reset_index()
daily["date"] = pd.to_datetime(daily["date"])

plt.figure(figsize=(16, 6))
plt.plot(daily["date"], daily["expected_fatalities"], linewidth=1)

plt.title("Daily Expected Fatalities (2015–2020)")
plt.xlabel("Date")
plt.ylabel("Expected Fatalities")
plt.grid(True)

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.YearLocator())  # tick every year
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

import calendar
# expected fatalities mean by month
daily["month"] = daily["date"].dt.month

monthly = daily.groupby("month")["expected_fatalities"].mean().reset_index()

plt.figure(figsize=(10, 5))
plt.plot(monthly["month"], monthly["expected_fatalities"], marker="o")
plt.xticks(
    monthly["month"],
    [calendar.month_abbr[m] for m in monthly["month"]]
)
plt.title("Average Expected Fatalities by Month (2015–2020)")
plt.xlabel("Month")
plt.ylabel("Mean Daily Expected Fatalities")
plt.grid(True)
plt.tight_layout()
#plt.show()


FEATURES = ["wind_speed", "d2m", "t2m", "msl", "tcc", "tp", "e", "pev"]
feature_names = {
    "wind_speed": "10 m Wind Speed",
    "d2m": "2 m Dewpoint Temperature",
    "t2m": "2 m Air Temperature",
    "msl": "Mean Sea Level Pressure",
    "tcc": "Total Cloud Cover",
    "tp": "Total Precipitation",
    "e": "Evaporation",
    "pev": "Potential Evaporation"
}

try:
    xgb_raw = calib_model.base_estimator        # older sklearn versions
except AttributeError:
    xgb_raw = calib_model.estimator            # newer sklearn versions


xgb_importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": xgb_base.feature_importances_
}).sort_values("importance", ascending=True)

xgb_importance["feature_label"] = xgb_importance["feature"].map(feature_names)


plt.figure(figsize=(8,6))
plt.barh(xgb_importance["feature_label"], xgb_importance["importance"])
plt.title("Feature Importance – XGBoost Tornado Probability Model")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()


