import pandas as pd
import numpy as np


def create_coefficient_comparison(result_unweighted, result_weighted, predictors, target_name):
    """
    Create comparison dataframe of coefficients between models
    
    Inputs:
        result_unweighted: pd.df
            Unweighted model result
        result_weighted: pd.df
            Weighted model result
        
    Outputs:
        Comparison dataframe: pd.df
    """
    coef_unweighted = result_unweighted.params
    coef_weighted = result_weighted.params[1:]  # Skip constant term
    
    # Align indices if needed
    if len(coef_weighted) == len(coef_unweighted) + 1:
        coef_unweighted_adjusted = np.insert(coef_unweighted, 0, np.nan)
        comparison = pd.DataFrame({
            'Predictor': ['(Intercept)'] + predictors,
            'Unweighted_Coef': coef_unweighted_adjusted,
            'Weighted_Coef': coef_weighted. values
        })
    else:
        comparison = pd.DataFrame({
            'Predictor': predictors,
            'Unweighted_Coef': coef_unweighted. values,
            'Weighted_Coef': coef_weighted. values
        })
    
    comparison['Coef_Difference'] = comparison['Weighted_Coef'] - comparison['Unweighted_Coef']
    
    return comparison


def create_odds_ratio_comparison(or_unweighted, or_weighted, predictors, target_name):

    comparison = pd.DataFrame({
        'Predictor': predictors,
        'Unweighted_OR': or_unweighted['Odds Ratio']. values,
        'Weighted_OR': or_weighted['Odds Ratio'].values
    })
    
    comparison['OR_Difference'] = comparison['Weighted_OR'] - comparison['Unweighted_OR']
    comparison['Pct_Change_OR'] = (
        (comparison['Weighted_OR'] - comparison['Unweighted_OR']) / 
        comparison['Unweighted_OR'] * 100
    )
    
    return comparison


def create_pct_change_odds_ratio(comparison_df):
    """
    Simplified dataframe showing percentage change in odds ratios

    """
    pct_change = pd.DataFrame({
        'Predictor': comparison_df['Predictor'],
        'Unweighted_OR': comparison_df['Unweighted_OR'],
        'Weighted_OR': comparison_df['Weighted_OR'],
        'Pct_Change':  comparison_df['Pct_Change_OR']
    })
    
    pct_change = pct_change.sort_values('Pct_Change', ascending=False)
    
    return pct_change


def identify_significant_changes(comparison_df, threshold=10):
    """
    Identify predictors with significant changes in odds ratios

    """
    significant = comparison_df[
        abs(comparison_df['Pct_Change_OR']) >= threshold
    ].copy()
    
    significant = significant.sort_values('Pct_Change_OR', key=abs, ascending=False)
    
    return significant

