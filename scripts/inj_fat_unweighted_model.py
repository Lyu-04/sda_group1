import numpy as np
import pandas as pd
import statsmodels.api as sm


def fit_unweighted_injury_model(X_scaled, y_inj):
    """
    Fit unweighted logistic regression for injury 
    
    """
    model_inj = sm.Logit(y_inj, X_scaled)
    result_inj = model_inj. fit()
    
    return result_inj, model_inj


def fit_unweighted_fatality_model(X_scaled, y_fat):
    """
    Fit unweighted logistic regression for fatality 
    
    """
    model_fat = sm.Logit(y_fat, X_scaled)
    result_fat = model_fat.fit()
    
    return result_fat, model_fat


def calculate_odds_ratios(result):
    """
    Calculate odds ratios and confidence intervals from model results
    
    """
    conf = result.conf_int()
    conf. columns = ['2.5%', '97.5%']
    
    odds_ratios = np.exp(result.params)
    ci_lower = np.exp(conf['2.5%'])
    ci_upper = np.exp(conf['97.5%'])
    
    summary = pd.DataFrame({
        'Odds Ratio': odds_ratios,
        'CI Lower': ci_lower,
        'CI Upper': ci_upper
    })
    
    return summary, conf, odds_ratios, ci_lower, ci_upper


def print_model_summary(result, model):
    print(f"\n{'='*60}")
    print(f"Model:  {model}")
    print(f"{'='*60}\n")
    print(result.summary())
