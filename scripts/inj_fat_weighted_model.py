"""
Weighted Logistic Regression Models
Fits logistic regression models with class weights to handle imbalance
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm


def fit_weighted_injury_model(X_scaled, y_inj, weights):
    
    X_scaled_const = sm.add_constant(X_scaled)
    model_weighted_inj = sm.Logit(y_inj, X_scaled_const)
    
    result_weighted_inj = model_weighted_inj.fit()
    
    return result_weighted_inj, model_weighted_inj


def fit_weighted_fatality_model(X_scaled, y_fat, weights):
    
    X_scaled_const = sm.add_constant(X_scaled)
    model_weighted_fat = sm.Logit(y_fat, X_scaled_const)
    
    # Note: Same as above
    result_weighted_fat = model_weighted_fat.fit()
    
    return result_weighted_fat, model_weighted_fat


def calculate_weighted_odds_ratios(result):
    """
    Calculate odds ratios and confidence intervals 
    
    """
    conf = result.conf_int()
    conf.columns = ['2.5%', '97.5%']
    
    odds_ratios = np.exp(result.params)
    ci_lower = np. exp(conf['2.5%'])
    ci_upper = np.exp(conf['97.5%'])
    
    summary = pd.DataFrame({
        'Odds Ratio': odds_ratios,
        'CI Lower': ci_lower,
        'CI Upper': ci_upper
    })
    
    return summary, conf, odds_ratios, ci_lower, ci_upper


def print_weighted_model_summary(result, model):
    print(f"\n{'='*60}")
    print(f"Model: {model}")
    print(f"{'='*60}\n")
    print(result.summary())

