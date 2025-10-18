"""
Feature Engineering
- Time-based features (dayofyear, month, weekday, weekend)
- Lag features: Nhiều biến trong chuỗi thời gian phụ thuộc vào giá trị trước đó của chính nó (auto-correlation).
- rolling mean/std for temp
- One-hot encode categorical features (text)
"""

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


# áp dụng được cả data
def create_auto_lag(df, forecast_horizon=1):
    """
    Tự động tạo lag = forecast_horizon cho tất cả các cột số học.
    """
    df = df.sort_index()
    numeric_cols = df.select_dtypes(include=['number']).columns

    for col in numeric_cols:
        df[f"{col}_lag_{forecast_horizon}"] = df[col].shift(forecast_horizon)

    return df

def create_date_features(df):
    # df là X thôi
    """
    Tạo các feature từ datetime:
      - month
      - cyclical encoding cho month
      - chuyển sunrise/sunset sang float hours
      - day_length = sunset - sunrise
    """
    dt = df.index
    df['month'] = dt.month
    df['weekday'] = dt.weekday

    # cyclical encoding
    df['month_sin'] = np.sin(2*np.pi*df['month']/12)
    df['month_cos'] = np.cos(2*np.pi*df['month']/12)

    # convert sunrise/sunset sang float hours
    for col in ['sunrise', 'sunset']:
        if col in df.columns:
            dt_col = pd.to_datetime(df[col], errors='coerce')
            df[col] = dt_col.dt.hour + dt_col.dt.minute/60 + dt_col.dt.second/3600

    # day_length = sunset - sunrise
    if 'sunrise' in df.columns and 'sunset' in df.columns:
        df['day_length'] = df['sunset'] - df['sunrise']

    return df


# Ensure time lags are greater than the forecast horizon
def create_lag_rolling(df, columns, lags=(2,3,4,5,6,7), windows=(3, 7,14), forecast_horizon=1, fill_method="bfill"):
    """
    Tạo lag và rolling features cho nhiều cột cùng lúc,
    đảm bảo KHÔNG vượt forecast_horizon (tránh leak).

    fill_method: 'bfill', 'ffill', hoặc 'mean'
    """
    df = df.sort_index()  
    #  Danh sách DataFrame để concat sau cùng
    new_features = []

    valid_lags = [l for l in lags if l >= forecast_horizon]
    if len(valid_lags) == 0:
        print(f"Không có lag nào >= forecast_horizon ({forecast_horizon}). Không tạo lag features.")
    else:
        lag_df = pd.concat(
            {f"{col}_lag_{l}": df[col].shift(l) for col in columns for l in valid_lags},
            axis=1
        )
        new_features.append(lag_df)

    # rolling
    roll_df = pd.concat(
        {
            f"{col}_roll_mean_{w}": df[col].shift(1).rolling(w, min_periods=1).mean()
            for col in columns for w in windows
        } |
        {
            f"{col}_roll_std_{w}": df[col].shift(1).rolling(w, min_periods=1).std()
            for col in columns for w in windows
        },
        axis=1
    )
    new_features.append(roll_df)

    df = pd.concat([df] + new_features, axis=1)


    # fill value
    if fill_method == "bfill":
        df = df.bfill()
    elif fill_method == "ffill":
        df = df.ffill()
    elif fill_method == "mean":
        df = df.fillna(df.mean(numeric_only=True))
    else:
        raise ValueError("fill_method phải là 'bfill', 'ffill' hoặc 'mean'")

    df = df.copy()

    return df



def create_specific_features(df):
    """
    Tạo các feature khí tượng đặc trưng, phù hợp với dữ liệu Hà Nội.
    """
    # Các biến cơ bản từ dữ liệu có sẵn
    df['temp_range'] = df['tempmax'] - df['tempmin']
    df['dew_spread'] = df['tempmin'] - df['dew']
    df['humidity_high'] = (df['humidity'] > 80).astype(int)
    df['rain_binary'] = (df['precip'] > 50).astype(int)
    df['rain_intensity'] = df['precip'] / (df['precipcover'] + 1e-5)

    # Gió - áp suất - nhiệt độ
    df['wind_temp_index'] = df['windspeed'] * df['tempmin']
    df['pressure_temp_index'] = df['sealevelpressure'] * df['tempmin']
    df['humidity_cloud_index'] = (df['humidity'] * df['cloudcover']) / 100
    df['solar_temp_index'] = df['solarradiation'] * df['tempmin']
    df['uv_cloud_index'] = df['moonphase'] * (1 - df['cloudcover'] / 100)  # moonphase gần tương tự UVindex

    # Gió mạnh và biến thiên
    df['wind_variability'] = df['windgust'] - df['windspeed']

    # Phân loại hướng gió
    def categorize_wind_direction(degree):
        if pd.isna(degree):
            return 'Unknown'
        elif 0 <= degree < 45 or 315 <= degree <= 360:
            return 'North'
        elif 45 <= degree < 135:
            return 'East'
        elif 135 <= degree < 225:
            return 'South'
        else:
            return 'West'

    df['wind_category'] = df['winddir'].apply(categorize_wind_direction)

    # Phân loại mùa (theo khí hậu Hà Nội)
    # Đông: 12–2, Xuân: 3–5, Hè: 6–8, Thu: 9–11
    df['season'] = df['month'].apply(
        lambda x: 'winter' if x in [12, 1, 2]
        else 'spring' if x in [3, 4, 5]
        else 'summer' if x in [6, 7, 8]
        else 'autumn'
    )

    # Foggy: tầm nhìn < 2 km
    df['foggy'] = (df['visibility'] < 2).astype(int)

    # One-hot encoding
    df = pd.get_dummies(df, columns=['wind_category', 'season'], drop_first=True)

    return df

def drop_future_features(df, cols_to_drop):
    """
    Loại bỏ các feature lag 0 
    """
    cols_to_drop = list(cols_to_drop)  # đảm bảo là list
    if 'temp' in cols_to_drop:
        cols_to_drop.remove('temp')

    df = df.drop(columns=cols_to_drop, errors='ignore')
    return df

def feature_engineering(df, column, forecast_horizon=1 ):
    """
    Gộp toàn bộ quy trình Feature Engineering.
    column for lag/rolling creating
    """
    df = create_date_features(df)
    df = create_specific_features(df)
    lag_0 = df.columns.copy()

    df = create_auto_lag(df, forecast_horizon=forecast_horizon)
    df = create_lag_rolling(df, columns=column,  forecast_horizon=forecast_horizon)
    df = drop_future_features(df, cols_to_drop=lag_0)
    return df