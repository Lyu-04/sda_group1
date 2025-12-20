"""
Fat and Injury Correlation Analysis Pipeline
Runs data preparation, model fitting (unweighted and "weighted"), comparisons,
and visualizations using the functions defined in this repository's modules.
"""

import sys
from pathlib import Path

# Data prep
from scripts.inj_fat_correlation_data_prep import (
    data_path as DEFAULT_DATA_PATH,
    load_data,
    create_target_variables,
    get_target_distribution,
    prepare_features,
    calculate_class_weights,
    apply_class_weights,
    predictors,
)

# Unweighted models
from scripts.inj_fat_unweighted_model import (
    fit_unweighted_injury_model,
    fit_unweighted_fatality_model,
    calculate_odds_ratios as calculate_odds_ratios_unweighted,
    print_model_summary as print_model_summary_unweighted,
)

# Weighted" model
from scripts.inj_fat_weighted_model import (
    fit_weighted_injury_model,
    fit_weighted_fatality_model,
    calculate_weighted_odds_ratios,
    print_weighted_model_summary,
)

# Visualize plots
from int_faj_visualizations import (
    create_output_directory,
    plot_coefficient_comparison,
    plot_odds_ratio_comparison,
    plot_pct_change_odds_ratio,
    plot_class_distribution,
    create_boxplot_by_injury_target,
    create_boxplot_for_fatality_target,
)

# Comparison 
from weighted_unweighted_comparison import (
    create_coefficient_comparison,
    create_odds_ratio_comparison,
    create_pct_change_odds_ratio,
    identify_significant_changes,
)


def main():

    # Load data 
    data_csv = DEFAULT_DATA_PATH # path defined in inj_fat_correlation_data_prep.py
    df = load_data(data_csv)
    df = create_target_variables(df)

    # Class distributions
    target_distribution_inj = get_target_distribution(df, "injury_target")
    target_distribution_fat = get_target_distribution(df, "fatality_target")
    print(target_distribution_inj)
    print(target_distribution_fat)

    # Prepare features
    X_scaled, scaler, pred_list, y_inj, y_fat = prepare_features(df)

    # Compute weights
    df = apply_class_weights(df, "injury_target", "injury_weights")
    df = apply_class_weights(df, "fatality_target", "fatality_weights")
    weights_inj = df["injury_weights"].values
    weights_fat = df["fatality_weights"].values

    # fit unweighted model
    res_inj_unw, mod_inj_unw = fit_unweighted_injury_model(X_scaled, y_inj)
    res_fat_unw, mod_fat_unw = fit_unweighted_fatality_model(X_scaled, y_fat)
    print_model_summary_unweighted(res_inj_unw, "Unweighted Injury Model")
    print_model_summary_unweighted(res_fat_unw, "Unweighted Fatality Model")

    # Odds ratios (unweighted)
    or_inj_unw_summary, _, _, _, _ = calculate_odds_ratios_unweighted(res_inj_unw)
    or_fat_unw_summary, _, _, _, _ = calculate_odds_ratios_unweighted(res_fat_unw)

    #fit weighted
    res_inj_w, mod_inj_w = fit_weighted_injury_model(X_scaled, y_inj, weights_inj)
    res_fat_w, mod_fat_w = fit_weighted_fatality_model(X_scaled, y_fat, weights_fat)
    print_weighted_model_summary(res_inj_w, "Weighted Injury Model")
    print_weighted_model_summary(res_fat_w, "Weighted Fatality Model")

    # Odds ratios (weighted)
    or_inj_w_summary, _, _, _, _ = calculate_weighted_odds_ratios(res_inj_w)
    or_fat_w_summary, _, _, _, _ = calculate_weighted_odds_ratios(res_fat_w)
    # Remove the intercept row from weighted OR summaries to align length with unweighted (if present)
    if len(or_inj_w_summary) == len(pred_list) + 1:
        or_inj_w_summary_no_const = or_inj_w_summary.iloc[1:].copy()
    else:
        or_inj_w_summary_no_const = or_inj_w_summary.copy()
    if len(or_fat_w_summary) == len(pred_list) + 1:
        or_fat_w_summary_no_const = or_fat_w_summary.iloc[1:].copy()
    else:
        or_fat_w_summary_no_const = or_fat_w_summary.copy()

    # Compare and visualize

    # Output directory: 
    out_dir = Path("../plots/class_imbalance")

    # Coefficient comparisons
    coef_cmp_inj = create_coefficient_comparison(res_inj_unw, res_inj_w, pred_list, "injury_target")
    coef_cmp_fat = create_coefficient_comparison(res_fat_unw, res_fat_w, pred_list, "fatality_target")

    # Odds ratio 
    or_cmp_inj = create_odds_ratio_comparison(or_inj_unw_summary, or_inj_w_summary_no_const, pred_list, "injury_target")
    or_cmp_fat = create_odds_ratio_comparison(or_fat_unw_summary, or_fat_w_summary_no_const, pred_list, "fatality_target")

    # Percentage change tables
    pct_change_inj = create_pct_change_odds_ratio(or_cmp_inj)
    pct_change_fat = create_pct_change_odds_ratio(or_cmp_fat)

    # Significant changes (>=10%)
    sig_inj = identify_significant_changes(or_cmp_inj, threshold=10)
    sig_fat = identify_significant_changes(or_cmp_fat, threshold=10)

    # Plots
    plot_coefficient_comparison(coef_cmp_inj, coef_cmp_fat, out_dir)
    plot_odds_ratio_comparison(or_cmp_inj, or_cmp_fat, out_dir)
    plot_pct_change_odds_ratio(pct_change_inj, pct_change_fat, out_dir)
    plot_class_distribution(target_distribution_inj, target_distribution_fat, out_dir)
    create_boxplot_by_injury_target(df, pred_list, out_dir)
    create_boxplot_for_fatality_target(df, pred_list, out_dir)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
