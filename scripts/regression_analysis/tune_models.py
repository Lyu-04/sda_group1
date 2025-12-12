import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier
from scipy.stats import randint, uniform

FEATURES = ["wind_speed", "d2m", "t2m", "msl", "tcc", "tp", "e", "pev"]


# TUNE P(TORNADO) MODEL (XGBoost, logit_data_small.csv)

print("\n=== Tuning P(Tornado) model (XGBoost) ===")

df_logit = pd.read_csv("logit_data_small.csv")
df_logit = df_logit.drop(columns=["number", "step", "surface", "valid_time"], errors="ignore")

if "wind_speed" not in df_logit.columns:
    df_logit["wind_speed"] = np.sqrt(df_logit["u10"]**2 + df_logit["v10"]**2)

X_torn = df_logit[FEATURES]
y_torn = df_logit["tornado"]

X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
    X_torn, y_torn, test_size=0.25, random_state=42, stratify=y_torn
)

neg, pos = (y_train_t == 0).sum(), (y_train_t == 1).sum()
scale_pos_weight = neg / pos
print("Class balance (tornado): neg =", neg, "pos =", pos, "ratio =", scale_pos_weight)

xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_jobs=-1,
    random_state=42,
    scale_pos_weight=scale_pos_weight,
)

param_dist_xgb = {
    "n_estimators": randint(200, 800),
    "max_depth": randint(3, 8),
    "learning_rate": uniform(0.01, 0.19),   # 0.01–0.2
    "subsample": uniform(0.6, 0.4),         # 0.6–1.0
    "colsample_bytree": uniform(0.6, 0.4),  # 0.6–1.0
    "min_child_weight": randint(1, 10),
}

search_xgb = RandomizedSearchCV(
    xgb,
    param_distributions=param_dist_xgb,
    n_iter=20,
    scoring="roc_auc",
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1,
)

search_xgb.fit(X_train_t, y_train_t)

best_xgb = search_xgb.best_estimator_
print("\nBest XGBoost params (P(tornado)):")
print(search_xgb.best_params_)

y_prob_t = best_xgb.predict_proba(X_test_t)[:, 1]
y_pred_t = best_xgb.predict(X_test_t)

print("\nXGBoost performance (P(tornado)):")
print("ROC-AUC:", roc_auc_score(y_test_t, y_prob_t))
print("Confusion matrix:")
print(confusion_matrix(y_test_t, y_pred_t))
print("Classification report:")
print(classification_report(y_test_t, y_pred_t, digits=4))


# TUNE P(FATAL | TORNADO) MODEL (RandomForestClassifier)
print("\n=== Tuning P(Fatal | Tornado) model (RandomForestClassifier) ===")

tw = pd.read_csv("tornado_weather.csv")
tw = tw[tw["fat"].notna()].copy()

if "wind_speed" not in tw.columns:
    tw["wind_speed"] = np.sqrt(tw["u10"]**2 + tw["v10"]**2)

tw["fatal_flag"] = (tw["fat"] > 0).astype(int)
print("\nFatal vs non-fatal counts:")
print(tw["fatal_flag"].value_counts())

X_flag = tw[FEATURES]
y_flag = tw["fatal_flag"]

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_flag, y_flag, test_size=0.25, random_state=42, stratify=y_flag
)

rf_clf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced")

param_dist_rf_clf = {
    "n_estimators": randint(200, 800),
    "max_depth": [None, 6, 10, 14],
    "min_samples_split": randint(2, 15),
    "min_samples_leaf": randint(1, 6),
    "max_features": ["sqrt", "log2", 0.5, 0.8],
}

search_rf_clf = RandomizedSearchCV(
    rf_clf,
    param_distributions=param_dist_rf_clf,
    n_iter=20,
    scoring="roc_auc",
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1,
)

search_rf_clf.fit(X_train_f, y_train_f)

fatal_clf_best = search_rf_clf.best_estimator_
print("\nBest RF params (P(fatal | tornado)):")
print(search_rf_clf.best_params_)

y_prob_f = fatal_clf_best.predict_proba(X_test_f)[:, 1]
y_pred_f = fatal_clf_best.predict(X_test_f)

print("\nRF performance (P(fatal | tornado)):")
print("ROC-AUC:", roc_auc_score(y_test_f, y_prob_f))
print("Confusion matrix:")
print(confusion_matrix(y_test_f, y_pred_f))
print("Classification report:")
print(classification_report(y_test_f, y_pred_f, digits=4))


# TUNE SEVERITY MODEL: E[FAT | FATAL, TORNADO]
print("\n=== Tuning Severity model (E[fat | fatal, tornado]) ===")

tw_fatal = tw[tw["fatal_flag"] == 1].copy()

X_sev = tw_fatal[FEATURES]
y_sev = np.log1p(tw_fatal["fat"])

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_sev, y_sev, test_size=0.25, random_state=42
)

rf_reg = RandomForestRegressor(random_state=42, n_jobs=-1)

param_dist_rf_reg = {
    "n_estimators": randint(200, 800),
    "max_depth": [None, 6, 10, 14],
    "min_samples_split": randint(2, 15),
    "min_samples_leaf": randint(1, 6),
    "max_features": ["sqrt", "log2", 0.5, 0.8],
}

search_rf_reg = RandomizedSearchCV(
    rf_reg,
    param_distributions=param_dist_rf_reg,
    n_iter=20,
    scoring="neg_mean_absolute_error",
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1,
)

search_rf_reg.fit(X_train_s, y_train_s)

rf_sev_best = search_rf_reg.best_estimator_
print("\nBest RF params (severity model):")
print(search_rf_reg.best_params_)

pred_log_s = rf_sev_best.predict(X_test_s)
y_true_s = np.expm1(y_test_s)
y_pred_s = np.expm1(pred_log_s)

print("\nSeverity model performance:")
print("MAE:", mean_absolute_error(y_true_s, y_pred_s))
print("R² :", r2_score(y_true_s, y_pred_s))

print("\nExample severity predictions:")
print(pd.DataFrame({"Actual": y_true_s[:10], "Predicted": y_pred_s[:10]}))


print("\n=== SUMMARY ===")
print("Use these trained models in your risk pipeline:")
print(" - best_xgb          -> P(tornado)")
print(" - fatal_clf_best    -> P(fatal | tornado)")
print(" - rf_sev_best       -> E[fat | fatal, tornado]")
