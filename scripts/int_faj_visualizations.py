import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path


def create_output_directory(output_dir='../ouput'):

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def plot_coefficient_comparison(coef_comparison_inj, coef_comparison_fat, output_dir):

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Injury coefficients
    x_pos = np.arange(len(coef_comparison_inj))
    width = 0.35
    
    axes[0]. bar(x_pos - width/2, coef_comparison_inj['Unweighted_Coef'], 
                width, label='Unweighted', alpha=0.8)
    axes[0].bar(x_pos + width/2, coef_comparison_inj['Weighted_Coef'], 
                width, label='Weighted', alpha=0.8)
    axes[0].set_xlabel('Predictors')
    axes[0].set_ylabel('Coefficient Value')
    axes[0].set_title('Injury Target:  Coefficient Comparison')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(coef_comparison_inj['Predictor'], rotation=45, ha='right')
    axes[0].legend()
    
    # Fatality coefficients
    x_pos_fat = np.arange(len(coef_comparison_fat))
    
    axes[1].bar(x_pos_fat - width/2, coef_comparison_fat['Unweighted_Coef'], 
                width, label='Unweighted', alpha=0.8)
    axes[1].bar(x_pos_fat + width/2, coef_comparison_fat['Weighted_Coef'], 
                width, label='Weighted', alpha=0.8)
    axes[1].set_xlabel('Predictors')
    axes[1].set_ylabel('Coefficient Value')
    axes[1].set_title('Fatality Target: Coefficient Comparison')
    axes[1].set_xticks(x_pos_fat)
    axes[1].set_xticklabels(coef_comparison_fat['Predictor'], rotation=45, ha='right')
    axes[1].legend()

    plt.savefig(output_dir / 'coefficient_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    


def plot_odds_ratio_comparison(or_comparison_inj, or_comparison_fat, output_dir):

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Injury odds ratios
    x_pos = np.arange(len(or_comparison_inj))
    width = 0.35
    
    axes[0].bar(x_pos - width/2, or_comparison_inj['Unweighted_OR'], 
                width, label='Unweighted', alpha=0.8)
    axes[0].bar(x_pos + width/2, or_comparison_inj['Weighted_OR'], 
                width, label='Weighted', alpha=0.8)
    axes[0].axhline(y=1, color='red', linestyle='--', linewidth=1, label='OR=1')
    axes[0].set_xlabel('Predictors')
    axes[0].set_ylabel('Odds Ratio')
    axes[0].set_title('Injury Target: Odds Ratio Comparison')
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(or_comparison_inj['Predictor'], rotation=45, ha='right')
    axes[0].legend()

    
    # Fatality odds ratios
    x_pos_fat = np.arange(len(or_comparison_fat))
    
    axes[1].bar(x_pos_fat - width/2, or_comparison_fat['Unweighted_OR'], 
                width, label='Unweighted', alpha=0.8)
    axes[1].bar(x_pos_fat + width/2, or_comparison_fat['Weighted_OR'], 
                width, label='Weighted', alpha=0.8)
    axes[1].axhline(y=1, color='red', linestyle='--', linewidth=1, label='OR=1')
    axes[1].set_xlabel('Predictors')
    axes[1].set_ylabel('Odds Ratio')
    axes[1].set_title('Fatality Target: Odds Ratio Comparison')
    axes[1].set_xticks(x_pos_fat)
    axes[1].set_xticklabels(or_comparison_fat['Predictor'], rotation=45, ha='right')
    axes[1].legend()
    
    plt.savefig(output_dir / 'odds_ratio_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    


def plot_pct_change_odds_ratio(pct_change_inj, pct_change_fat, output_dir):
  
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Injury percentage change
    colors_inj = ['green' if x > 0 else 'blue' for x in pct_change_inj['Pct_Change']]
    axes[0].barh(pct_change_inj['Predictor'], pct_change_inj['Pct_Change'], color=colors_inj, alpha=0.8)
    axes[0].set_xlabel('Percentage Change in Odds Ratio')
    axes[0].axvline(x=0, color='black', linestyle='-')
    
    # Fatality percentage change
    colors_fat = ['green' if x > 0 else 'red' for x in pct_change_fat['Pct_Change']]
    axes[1]. barh(pct_change_fat['Predictor'], pct_change_fat['Pct_Change'], color=colors_fat, alpha=0.8)
    axes[1].set_xlabel('Percentage Change in Odds Ratio')
    axes[1].axvline(x=0, color='black', linestyle='-')
    
    plt.savefig(output_dir / 'pct_change_odds_ratio.png', dpi=300, bbox_inches='tight')
    plt.close()



def plot_class_distribution(target_distribution_inj, target_distribution_fat, output_dir):

    fig, axes = plt. subplots(1, 2, figsize=(12, 4))
    
    # Injury distribution
    injury_labels = ['No Injury (0)', 'Injury (1)']
    axes[0].bar(injury_labels, target_distribution_inj.values, 
                color=['#2ecc71', '#e74c3c'], alpha=0.8)
    axes[0].set_title('Injury Target Distribution')
    axes[0].set_ylabel('Proportion')
    for i, v in enumerate(target_distribution_inj.values):
        axes[0].text(i, v + 0.01, f'{v:.4f}', ha='center')
    axes[0].set_ylim([0, 1.0])
    
    # Fatality distribution
    fatality_labels = ['No Fatality (0)', 'Fatality (1)']
    axes[1].bar(fatality_labels, target_distribution_fat.values, 
                color=['#3498db', '#e67e22'], alpha=0.8)
    axes[1].set_title('Fatality Target Distribution')
    axes[1].set_ylabel('Proportion')
    for i, v in enumerate(target_distribution_fat.values):
        axes[1].text(i, v + 0.01, f'{v:.4f}', ha='center')
    axes[1].set_ylim([0, 1.0])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_boxplot_by_injury_target(df, predictors, output_dir):
    """
    Create boxplots of predictors grouped by injury target
    
    """
    # Select first 10 predictors for visualization (to avoid overcrowding)
    predictors_subset = predictors[:10]
    
    fig, axes = plt.subplots(2, 5, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, predictor in enumerate(predictors_subset):
        data_no_injury = df[df['injury_target'] == 0][predictor]
        data_injury = df[df['injury_target'] == 1][predictor]
        
        axes[idx].boxplot([data_no_injury, data_injury], 
                         labels=['No Injury', 'Injury'])
        axes[idx].set_title(f'{predictor}', fontweight='bold')
        axes[idx].set_ylabel('Value')
        axes[idx].grid(axis='y', alpha=0.3)
    
    plt.suptitle('Predictor Distributions by Injury Status', 
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(output_dir / 'boxplot_by_injury_target.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_boxplot_for_fatality_target(df, predictors, output_dir):
    """
    Create boxplots of predictors grouped by injury target
    
    """
    # Select first 10 predictors for visualization (to avoid overcrowding)
    predictors_subset = predictors[:10]
    
    fig, axes = plt.subplots(2, 5, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, predictor in enumerate(predictors_subset):
        data_no_fatality = df[df['fatality_target'] == 0][predictor]
        data_fatality = df[df['fatality_target'] == 1][predictor]
        
        axes[idx].boxplot([data_no_fatality, data_fatality], 
                         labels=['No Fatality', 'Fatality'])
        axes[idx].set_title(f'{predictor}', fontweight='bold')
        axes[idx].set_ylabel('Value')
        axes[idx].grid(axis='y', alpha=0.3)
    
    plt.suptitle('Predictor Distributions by Fatality Status', 
                fontsize=14, fontweight='bold', y=1.00)
    plt.savefig(output_dir / 'boxplot_by_fatality_target.png', dpi=300, bbox_inches='tight')
    plt.close()
    
