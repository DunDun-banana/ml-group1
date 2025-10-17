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
def create_date_features(df):
    # df là X thôi
    """
    Tạo các feature từ datetime:
      - year, month, day, dayofyear, weekday, is_weekend
      - cyclical encoding cho month và dayofyear
      - chuyển sunrise/sunset sang float hours
      - day_length = sunset - sunrise
    """
    dt = df.index

    df['year'] = dt.year
    df['month'] = dt.month
    df['day'] = dt.day
    df['dayofyear'] = dt.dayofyear
    df['weekday'] = dt.weekday
    df['is_weekend'] = df['weekday'].isin([5, 6]).astype(int)

    # cyclical encoding
    df['month_sin'] = np.sin(2*np.pi*df['month']/12)
    df['month_cos'] = np.cos(2*np.pi*df['month']/12)
    df['dayofyear_sin'] = np.sin(2*np.pi*df['dayofyear']/365.25)
    df['dayofyear_cos'] = np.cos(2*np.pi*df['dayofyear']/365.25)

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
def create_lag_rolling(df, columns, lags=(1, 2, 3, 7), windows=(3, 7), forecast_horizon=1, fill_method="bfill"):
    """
    Tạo lag và rolling features cho nhiều cột cùng lúc, 
    đảm bảo lag không vượt quá forecast_horizon.
    Thay vì drop các giá trị NaN, sẽ fill hợp lý (bfill/ffill/mean).

    Parameters:
    - df: DataFrame
    - columns: list các cột cần tạo lag/rolling
    - lags: tuple các giá trị lag (VD: (1,2,3,7))
    - windows: tuple các giá trị rolling window
    - forecast_horizon: số ngày dự báo (lag lớn hơn giá trị này sẽ bị bỏ)
    - fill_method: cách điền giá trị NaN ('bfill', 'ffill', 'mean')

    Returns:
    - df: DataFrame có thêm các cột lag/rolling đã được fill
    """
    df = df.sort_index()  # đảm bảo theo thời gian

    valid_lags = [l for l in lags if l <= forecast_horizon]
    if len(valid_lags) == 0:
        print(f"Không có lag nào ≤ forecast_horizon ({forecast_horizon}). Không tạo lag features.")
    else:
        for col in columns:
            for l in valid_lags:
                df[f"{col}_lag_{l}"] = df[col].shift(l)

    for col in columns:
        for w in windows:
            df[f"{col}_roll_mean_{w}"] = df[col].rolling(w, min_periods=1).mean()
            df[f"{col}_roll_std_{w}"] = df[col].rolling(w, min_periods=1).std()

    # 🔹 Fill NaN thay vì drop
    if fill_method == "bfill":
        df = df.bfill()
    elif fill_method == "ffill":
        df = df.ffill()
    elif fill_method == "mean":
        df = df.fillna(df.mean(numeric_only=True))
    else:
        raise ValueError("fill_method phải là 'bfill', 'ffill' hoặc 'mean'")

    return df



def create_specific_features(df):
    """
    Tạo các feature khí tượng đặc trưng, phù hợp với dữ liệu Hà Nội.
    """
    # Các biến cơ bản từ dữ liệu có sẵn
    df['temp_range'] = df['tempmax'] - df['tempmin']
    df['dew_spread'] = df['tempmin'] - df['dew']
    df['humidity_high'] = (df['humidity'] > 80).astype(int)
    df['rain_binary'] = (df['precip'] > 0).astype(int)
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

    # Label encode các biến phân loại
    le = LabelEncoder()
    if 'conditions' in df.columns:
        df['conditions_encoded'] = le.fit_transform(df['conditions'].astype(str))

    # One-hot encoding
    df = pd.get_dummies(df, columns=['wind_category', 'season'], drop_first=True)

    return df


def feature_engineering(df, column):
    """
    Gộp toàn bộ quy trình Feature Engineering.
    column for lag/rolling creating
    """
    df = create_date_features(df)
    df = create_specific_features(df)
    df = create_lag_rolling(df, columns=column)
    return df