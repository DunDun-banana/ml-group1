"""
Feature Engineering
- Time-based features (dayofyear, month, weekday, weekend)
- Lag features, rolling mean/std for temp
- One-hot encode categorical features (text)
"""

import pandas as pd
import numpy as np

def create_date_features(df):
    dt = pd.to_datetime(df['datetime'])
    df['year'] = dt.dt.year
    df['month'] = dt.dt.month
    df['day'] = dt.dt.day
    df['dayofyear'] = dt.dt.dayofyear
    df['weekday'] = dt.dt.weekday
    df['is_weekend'] = df['weekday'].isin([5,6]).astype(int)
    # cyclical encoding
    df['month_sin'] = np.sin(2*np.pi*df['month']/12)
    df['month_cos'] = np.cos(2*np.pi*df['month']/12)
    df['dayofyear_sin'] = np.sin(2*np.pi*df['dayofyear']/365.25)
    df['dayofyear_cos'] = np.cos(2*np.pi*df['dayofyear']/365.25)

    # --- convert sunrise/sunset to float hours if present ---
    for col in ['sunrise', 'sunset']:
        if col in df.columns:
            dt_col = pd.to_datetime(df[col], errors='coerce')
            df[col] = dt_col.dt.hour + dt_col.dt.minute/60 + dt_col.dt.second/3600

    return df

def create_lag_rolling(df, target='temp', lags=(1,2,3,7), windows=(3,7)):
    df = df.sort_values('datetime')
    for l in lags:
        df[f'{target}_lag_{l}'] = df[target].shift(l)
    for w in windows:
        df[f'{target}_roll_mean_{w}'] = df[target].rolling(w, min_periods=1).mean()
        df[f'{target}_roll_std_{w}'] = df[target].rolling(w, min_periods=1).std()
    df = df.dropna().reset_index(drop=True)  # drop initial NaNs
    return df

def encode_categorical(df, train_columns=None, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = ['datetime']
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    cat_cols = [c for c in cat_cols if c not in exclude_cols]
    df_encoded = pd.get_dummies(df, columns=cat_cols, dummy_na=True)

    if train_columns is not None:
        df_encoded = df_encoded.reindex(columns=train_columns, fill_value=0)

    return df_encoded

