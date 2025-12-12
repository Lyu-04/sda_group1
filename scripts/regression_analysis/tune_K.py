import pandas as pd
import numpy as np

risk_file = "risk_2015_2020_hurdle.csv"
df = pd.read_csv(risk_file)

# make sure time/date exist
df["time"] = pd.to_datetime(df["time"])
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
else:
    df["date"] = df["time"].dt.normalize()

# REAL TORNADO FATALITIES
t = pd.read_csv("clean_tornado_tx_1970_2021.csv", parse_dates=["date"])
t = t[t["yr"].between(2015, 2020)]
real_yearly = t.groupby(t["date"].dt.year)["fat"].sum().reset_index()
real_yearly.columns = ["year", "real_fatalities"]

print("Real yearly fatalities (2015–2020):")
print(real_yearly)


# FUNCTION TO EVALUATE K
def evaluate_K(K, df, real_yearly):
    """
    For a given K:
      - take top-K grids per day by p_tornado
      - sum expected_fatalities over those K
      - aggregate to yearly totals
      - return MAE and comparison table
    """
    # sort so highest-risk grids come first each day
    df_sorted = df.sort_values(["date", "p_tornado"], ascending=[True, False])

    # top-K per day
    topK = df_sorted.groupby("date").head(K)

    # daily totals
    daily = (
        topK.groupby("date")["expected_fatalities"]
            .sum()
            .reset_index()
    )
    daily["year"] = daily["date"].dt.year

    # yearly totals
    pred_yearly = (
        daily.groupby("year")["expected_fatalities"]
             .sum()
             .reset_index()
    )
    pred_yearly.columns = ["year", "predicted_fatalities"]

    # compare with real
    comp = real_yearly.merge(pred_yearly, on="year", how="left").fillna(0)

    mae = np.mean(np.abs(comp["real_fatalities"] - comp["predicted_fatalities"]))
    return mae, comp


results = []

for K in range(1, 51):
    mae, _ = evaluate_K(K, df, real_yearly)
    results.append((K, mae))

# results table
results_df = pd.DataFrame(results, columns=["K", "MAE"]).sort_values("MAE")
print("\nMAE for K=1..50 (sorted):")
print(results_df)

# best K
best_K = results_df.iloc[0]["K"]
print(f"\nBest K based on MAE = {int(best_K)}")

#SHOW COMPARISON FOR BEST K
best_mae, best_comp = evaluate_K(int(best_K), df, real_yearly)
print("\nComparison for best K:")
print(best_comp)
print("\nMAE for best K:", best_mae)