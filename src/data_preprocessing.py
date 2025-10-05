"""
Data Preprocessing for Hanoi Temperature Forecasting
- Load raw CSV
- Handle missing values, numeric/categorical coercion
- Save processed CSV
"""

import pandas as pd
import numpy as np
from pathlib import Path

NUMERIC_CANDIDATES = [
    'tempmax','tempmin','temp','feelslikemax','feelslikemin','feelslike',
    'dew','humidity','precip','precipprob','precipcover','snow','snowdepth',
    'windgust','windspeed','winddir','sealevelpressure','cloudcover','visibility',
    'solarradiation','solarenergy','uvindex','severerisk','moonphase'
]

CATEGORICAL_CANDIDATES = ['preciptype','conditions','description','icon','stations']

def load_data(input_path: str):
    df = pd.read_csv(input_path)
    df.columns = [c.strip() for c in df.columns]
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    for c in ['sunrise','sunset']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')
    return df

def coerce_numeric(df):
    for col in NUMERIC_CANDIDATES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def handle_missing(df):
    df = df.sort_values('datetime')
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # set datetime làm index
    df = df.set_index('datetime')
    
    # interpolate numeric columns
    df[numeric_cols] = df[numeric_cols].interpolate(method='time', limit_direction='both')
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    # reset index về cột bình thường
    df = df.reset_index()
    
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    for c in cat_cols:
        df[c] = df[c].fillna('unknown')
    return df

def preprocess(df):
    """Apply preprocessing on a DataFrame (already loaded)"""
    df = coerce_numeric(df)
    df = handle_missing(df)
    return df

