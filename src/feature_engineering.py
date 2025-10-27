"""
Feature Engineering
- Time-based features (dayofyear, month, weekday, weekend)
- Lag features: Nhiều biến trong chuỗi thời gian phụ thuộc vào giá trị trước đó của chính nó (auto-correlation).
- rolling mean/std for temp
- One-hot encode categorical features (text)
"""

import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

def create_date_features(df):
    dt = df.index
    df = df.copy()
    df['month'] = dt.month
    df['weekday'] = dt.weekday
    df['day_of_year'] = dt.dayofyear
    df['is_weekend'] = (dt.weekday >= 5).astype(int)
    df['month_sin'] = np.sin(2*np.pi*df['month']/12)
    df['month_cos'] = np.cos(2*np.pi*df['month']/12)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    for col in ['sunrise', 'sunset']:
        if col in df.columns:
            dt_col = pd.to_datetime(df[col], errors='coerce')
            df[col] = dt_col.dt.hour + dt_col.dt.minute/60 + dt_col.dt.second/3600
    if 'sunrise' in df.columns and 'sunset' in df.columns:
        df['day_length'] = df['sunset'] - df['sunrise']
    return df

def create_specific_features(df):
    """
    Tạo các features đặc thù từ current data
    """
    df = df.copy()
    
    # Temperature features
    df['temp_range'] = df['tempmax'] - df['tempmin']
    df['dew_spread'] = df['temp'] - df['dew']
    df['temp_dew_interaction'] = df['temp'] * df['dew']
    
    # Weather condition features
    df['humidity_high'] = (df['humidity'] > 80).astype(int)
    df['rain_binary'] = (df['precip'] > 5.0).astype(int)
    df['rain_intensity'] = df['precip'] / (df['precipcover'] + 1e-5)
    
    # Atmospheric indices
    df['wind_temp_index'] = df['windspeed'] * df['temp']
    df['pressure_temp_index'] = df['sealevelpressure'] * df['temp']
    df['humidity_cloud_index'] = (df['humidity'] * df['cloudcover']) / 100
    df['solar_temp_index'] = df['solarradiation'] * df['temp']
    df['uv_cloud_index'] = df['moonphase'] * (1 - df['cloudcover'] / 100)
    df['wind_variability'] = df['windgust'] - df['windspeed']
    
    # Comfort indices
    df['heat_index'] = 0.5 * (df['temp'] + 61.0 + ((df['temp']-68.0)*1.2) + (df['humidity']*0.094))
    df['wind_chill'] = 13.12 + 0.6215*df['temp'] - 11.37*(df['windspeed']**0.16) + 0.3965*df['temp']*(df['windspeed']**0.16)
    
    # Wind direction categorization
    def categorize_wind_direction(degree):
        if pd.isna(degree): return 'Unknown'
        elif 0 <= degree < 45 or 315 <= degree <= 360: return 'North'
        elif 45 <= degree < 135: return 'East'
        elif 135 <= degree < 225: return 'South'
        else: return 'West'
    
    df['wind_category'] = df['winddir'].apply(categorize_wind_direction)
    
    # Season
    df['season'] = df['month'].apply(
        lambda x: 'winter' if x in [12, 1, 2]
        else 'spring' if x in [3, 4, 5]
        else 'summer' if x in [6, 7, 8]
        else 'autumn'
    )
    
    # Weather phenomena
    df['foggy'] = (df['visibility'] < 2).astype(int)
    df['high_wind'] = (df['windspeed'] > 15).astype(int)
    
    # One-hot encoding
    categorical_cols = []
    if 'wind_category' in df.columns:
        categorical_cols.append('wind_category')
    if 'season' in df.columns:
        categorical_cols.append('season')
    
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, prefix_sep='_')
    
    return df

def auto_create_lag_features(df, lag_periods=[1], feature_groups=None):
    """
    Tự động tạo lag features cho các features đã được tạo - ĐÃ SỬA PERFORMANCE
    """
    df = df.copy()
    
    if feature_groups is None:
        # Các nhóm features mặc định cần tạo lag
        feature_groups = {
            'temperature': ['temp', 'tempmax', 'tempmin', 'feelslike', 'feelslikemax', 'feelslikemin'],
            'humidity_dew': ['humidity', 'dew'],
            'precipitation': ['precip', 'precipprob', 'precipcover'],
            'wind': ['windspeed', 'windgust', 'winddir'],
            'pressure': ['sealevelpressure', 'pressure'],
            'cloud_visibility': ['cloudcover', 'visibility'],
            'solar': ['solarradiation', 'uvindex', 'solarenergy'],
            'custom_features': [
                'temp_range', 'dew_spread', 'temp_dew_interaction', 'rain_intensity',
                'wind_temp_index', 'pressure_temp_index', 'humidity_cloud_index',
                'solar_temp_index', 'wind_variability', 'heat_index', 'wind_chill'
            ]
        }
    
    # Tạo tất cả lag features cùng lúc bằng pd.concat
    lag_dataframes = []
    lagged_features = []
    
    # for features in feature_groups:
    #     # Chỉ lấy các features thực sự tồn tại trong DataFrame
    #     existing_features = [f for f in features if f in df.columns]
    #     print(f'k{existing_features}')
        
    for feature in feature_groups:
        for lag in lag_periods:
            lag_col_name = f"{feature}_lag_{lag}"
            # Tạo Series cho lag feature
            lag_series = df[feature].shift(lag)
            lag_series.name = lag_col_name
            lag_dataframes.append(lag_series)
            lagged_features.append(lag_col_name)
    
    # Ghép tất cả lag features cùng lúc
    if lag_dataframes:
        lag_df = pd.concat(lag_dataframes, axis=1)
        df = pd.concat([df, lag_df], axis=1)

    return df, lagged_features

def create_rolling_features(df, windows=[3, 7, 14], feature_groups=None):
    """
    Tạo rolling features từ các features đã lag - 
    """
    df = df.copy()
    
    if feature_groups is None:
        feature_groups = {
            'temperature': ['temp', 'tempmax', 'tempmin'],
            'humidity_dew': ['humidity', 'dew'],
            'pressure': ['sealevelpressure'],
            'wind': ['windspeed']
        }
    
    # Tạo tất cả rolling features cùng lúc
    roll_dataframes = []
    rolling_features = []
    
    # for features in feature_groups:
    #     existing_features = [f for f in features if f in df.columns]
    #     print(f'f{df.columns}')
        
    for feature in feature_groups:
        for window in windows:
            # Sử dụng lag_1 để tạo rolling features (tránh data leakage)
            base_series = df[feature].shift(1)
            
            mean_col = f"{feature}_roll_mean_{window}"
            std_col = f"{feature}_roll_std_{window}"
            min_col = f"{feature}_roll_min_{window}"
            max_col = f"{feature}_roll_max_{window}"
            
            # Tạo rolling features
            mean_series = base_series.rolling(window, min_periods=1).mean()
            std_series = base_series.rolling(window, min_periods=1).std()
            min_series = base_series.rolling(window, min_periods=1).min()
            max_series = base_series.rolling(window, min_periods=1).max()
            
            mean_series.name = mean_col
            std_series.name = std_col
            min_series.name = min_col
            max_series.name = max_col
            
            roll_dataframes.extend([mean_series, std_series, min_series, max_series])
            rolling_features.extend([mean_col, std_col, min_col, max_col])
    
    # Ghép tất cả rolling features cùng lúc
    if roll_dataframes:
        roll_df = pd.concat(roll_dataframes, axis=1)
        df = pd.concat([df, roll_df], axis=1)

    return df, rolling_features

def drop_current_features(df, keep_features=None):
    """
    Loại bỏ các features từ current day, chỉ giữ lại lag features
    """
    if keep_features is None:
        keep_features = ['month', 'weekday', 'day_of_year', 'is_weekend', 
                        'month_sin', 'month_cos', 'day_sin', 'day_cos']
    
    # Tất cả các features cần giữ (date features + lag/rolling features)
    all_keep_features = keep_features.copy()
    
    # Thêm tất cả các features có chứa '_lag_' hoặc '_roll_'
    lag_roll_features = [col for col in df.columns if '_lag_' in col or '_roll_' in col]
    all_keep_features.extend(lag_roll_features)
    
    # Thêm target columns nếu có
    target_features = [col for col in df.columns if col.startswith('temp_next_')]
    all_keep_features.extend(target_features)
    
    # Lấy danh sách features cần xóa (current day features)
    features_to_drop = [col for col in df.columns if col not in all_keep_features]
    # for n in features_to_drop:
    #     print(n)
     
    return df.drop(features_to_drop, axis = 1)

def feature_engineering(df, forecast_horizon=5):
    """
    Thực hiện toàn bộ quy trình Feature Engineering cho bài toán dự báo đa đầu ra.
    ĐÃ SỬA PERFORMANCE: Sử dụng pd.concat thay vì multiple inserts.
    """
    
    # Tạo bản copy để tránh fragmentation
    df = df.copy()
    
    # 1. Tạo targets
    target_cols = []
    target_dataframes = []
    
    for i in range(1, forecast_horizon + 1):
        target_col = f"temp_next_{i}"
        target_series = df['temp'].shift(-i)
        target_series.name = target_col
        target_dataframes.append(target_series)
        target_cols.append(target_col)
    
    # Ghép target columns cùng lúc
    if target_dataframes:
        target_df = pd.concat(target_dataframes, axis=1)
        df = pd.concat([df, target_df], axis=1)

    
    # 2. Tạo date features (được phép dùng vì là thông tin có sẵn)
    df = create_date_features(df)
    
    # 3. Tạo specific features từ current data
    df = create_specific_features(df)
    feature_group = df.columns
    feature_group = feature_group.drop(['temp_next_1','temp_next_2','temp_next_3','temp_next_4','temp_next_5'])
    
    # 4. Tạo lag features cho tất cả features vừa tạo
    df, lagged_features = auto_create_lag_features(df, lag_periods=[1,2,3,5,7], feature_groups= feature_group)
    
    # 5. Tạo rolling features từ các features đã lag
    df, rolling_features = create_rolling_features(df, windows=[3,5,7,10,20], feature_groups= feature_group)
    
    # 6. Loại bỏ current day features, chỉ giữ lại lag/rolling features
    #df = drop_current_features(df)
    
    # 7. Xử lý missing values và clean data
    df = df.ffill().bfill()
    df = df.dropna(subset=target_cols)
    
    # Tạo defragmented frame
    df = df.copy()

    
    return df, target_cols

