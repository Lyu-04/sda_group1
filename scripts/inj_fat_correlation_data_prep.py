import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

data_path = '../Data/final/final_data1.csv'

predictors = [
    'u10', 'v10', 'd2m', 't2m', 'msl', 'tcc', 'tp',
    'len', 'wid',
    'yr', 'mo', 'dy', 
    'mag', 'slat', 'slon', 'elat', 'elon', 
    'latitude', 'longitude', 
    'dist2']

def load_data(data_path):
    """
    Load final cleaned dataset from CSV file
    
    """
    df = pd.read_csv(data_path)
    return df

def create_target_variables(df):
    """
    Create binary target variables for injury and fatality

    """
    df['fatality_target'] = (df['fat'] > 0).astype(int)
    df['injury_target'] = (df['inj'] > 0).astype(int)
    
    return df


def get_target_distribution(df, target_name):
    """
    Normalized class distribution for given target variable

    """
    distribution = df[target_name].value_counts(normalize=True)
    return distribution


def prepare_features(df):
    """
    Prepare and scale features for modeling
    
    """
    X = df[predictors]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    y_inj = df['injury_target']
    y_fat = df['fatality_target']
    
    return X_scaled, scaler, predictors, y_inj, y_fat


def calculate_class_weights(df, target_name):
    """
    Calculate class weights for handling imbalanced data
    
    """
    n_total = len(df)
    n_positive = df[target_name].sum()
    
    weight_positive = n_total / (2 * n_positive)
    weight_negative = n_total / (2 * (n_total - n_positive))
    
    return weight_positive, weight_negative, n_total, n_positive


def apply_class_weights(df, target_name, weight_col_name):
    """
    Apply calculated weights to dataset
    
    """
    weight_positive, weight_negative, _, _ = calculate_class_weights(df, target_name)
    
    df[weight_col_name] = df[target_name]. apply(
        lambda x: weight_positive if x == 1 else weight_negative
    )
    
    return df

