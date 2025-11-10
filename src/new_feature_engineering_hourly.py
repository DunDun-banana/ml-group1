"""
Feature Engineering
- Time-based features (dayofyear, month, weekday, weekend)
- Lag features: Nhiều biến trong chuỗi thời gian phụ thuộc vào giá trị trước đó của chính nó (auto-correlation).
- rolling mean/std for temp
- Sử dụng category type thay vì one-hot encoding cho categorical features
- Enhanced với hourly features và Vietnam seasonal patterns
"""

import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

def create_date_features(df, forecast_horizon=5):
    df = df.copy()
    base_dates = df.index
    
    # Date features cho current day (t)
    df['month'] = base_dates.month
    df['weekday'] = base_dates.weekday
    df['day_of_year'] = base_dates.dayofyear
    df['month_sin'] = np.sin(2*np.pi*df['month']/12)
    df['month_cos'] = np.cos(2*np.pi*df['month']/12)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # Tạo date features cho các horizons t+1 đến t+5
    for horizon in range(1, forecast_horizon + 1):
        future_dates = base_dates + pd.Timedelta(days=horizon)
        
        df[f'day_of_year_h{horizon}'] = future_dates.dayofyear
        df[f'day_sin_h{horizon}'] = np.sin(2 * np.pi * df[f'day_of_year_h{horizon}'] / 365)
        df[f'day_cos_h{horizon}'] = np.cos(2 * np.pi * df[f'day_of_year_h{horizon}'] / 365)

    for col in ['sunrise', 'sunset']:
        if col in df.columns:
            dt_col = pd.to_datetime(df[col], errors='coerce')
            df[col] = dt_col.dt.hour + dt_col.dt.minute/60 + dt_col.dt.second/3600
    if 'sunrise' in df.columns and 'sunset' in df.columns:
        df['day_length'] = df['sunset'] - df['sunrise']

    df['sunrise_sin'] = np.sin(2 * np.pi * df['sunrise'] / 24)
    df['sunrise_cos'] = np.cos(2 * np.pi * df['sunrise'] / 24)
    df['sunset_sin']  = np.sin(2 * np.pi * df['sunset'] / 24)
    df['sunset_cos']  = np.cos(2 * np.pi * df['sunset'] / 24)

    return df

def enhance_hourly_features(df):
    """
    Tạo features mới từ các biến đã tổng hợp từ hourly data
    """
    df = df.copy()
    
    # 2. Tính biến động nhiệt độ trong ngày từ các khung 6h
    temp_max_cols = [f'temp_max_6h_next{i}' for i in range(4) if f'temp_max_6h_next{i}' in df.columns]
    temp_min_cols = [f'temp_min_6h_next{i}' for i in range(4) if f'temp_min_6h_next{i}' in df.columns]
    
    if temp_max_cols:
        df['temp_6h_variability'] = df[temp_max_cols].std(axis=1)
        df['temp_6h_range'] = df[temp_max_cols].max(axis=1) - df[temp_min_cols].min(axis=1)
    
    # 3. Xu hướng nhiệt độ trong ngày
    if 'temp_mean_6h_next0' in df.columns and 'temp_mean_6h_next3' in df.columns:
        df['temp_6h_trend'] = (df['temp_mean_6h_next3'] - df['temp_mean_6h_next0']) / 3
    
    # 4. Tương tự cho feelslike
    feelslike_max_cols = [f'feelslike_max_6h_next{i}' for i in range(4) if f'feelslike_max_6h_next{i}' in df.columns]
    if feelslike_max_cols:
        df['feelslike_6h_variability'] = df[feelslike_max_cols].std(axis=1)
    
    # 5. Tương tác giữa các khung giờ
    if all(col in df.columns for col in ['temp_max_6h_next1', 'temp_max_6h_next2', 'temp_min_6h_next0', 'temp_min_6h_next3']):
        df['day_night_temp_diff'] = (df['temp_max_6h_next1'] + df['temp_max_6h_next2']) / 2 - \
                                   (df['temp_min_6h_next0'] + df['temp_min_6h_next3']) / 2
    
    return df

def create_momentum_features(df):
    """
    Tạo features về momentum và acceleration của nhiệt độ
    """
    df = df.copy()
    
    # Temperature momentum (tốc độ thay đổi)
    if 'temp_lag_1' in df.columns and 'temp_lag_2' in df.columns:
        df['temp_momentum_1d'] = df['temp'] - df['temp_lag_1']
        df['temp_momentum_2d'] = df['temp_lag_1'] - df['temp_lag_2']
        
        # Temperature acceleration (gia tốc)
        df['temp_acceleration'] = df['temp_momentum_1d'] - df['temp_momentum_2d']
        
        # Rolling momentum
        df['temp_momentum_3d_avg'] = df['temp_momentum_1d'].rolling(3, min_periods=3).mean()
        df['temp_acceleration_3d_avg'] = df['temp_acceleration'].rolling(3, min_periods=3).mean()
    
    return df

def create_specific_features(df, is_linear=False):
    """
    Tạo các features đặc thù từ current data
    """
    df = df.copy()
    
    # Temperature features
    df['temp_range'] = df['tempmax'] - df['tempmin']
    df['dew_spread'] = df['temp'] - df['dew']
    
    # Weather condition features
    if 'precipcover' in df.columns:
        df['rain_intensity'] = df['precip'] / (df['precipcover'] + 1e-5)
    
    # Atmospheric indices - Các biến interaction không nên dùng với tree-base model
    if is_linear: # Các biến interaction chỉ dùng khi sử dụng linear model
        df['wind_temp_index'] = df['windspeed'] * df['temp']
        df['pressure_temp_index'] = df['sealevelpressure'] * df['temp']
        df['humidity_cloud_index'] = (df['humidity'] * df['cloudcover']) / 100
        df['solar_temp_index'] = df['solarradiation'] * df['temp']
        df["temp_humidity_interaction"] = df["temp"] * df["humidity"]
        df["wind_temp_interaction"] = df["winddir"] * df["temp"]
        df['temp_dew_interaction'] = df['temp'] * df['dew']
        print('Đã tạo 7 biến interaction')

    df['uv_cloud_index'] = df['moonphase'] * (1 - df['cloudcover'] / 100)
    df['solar_visibility_index'] = df['visibility'] * (1 - df['cloudcover'] / 100)
    df['wind_variability'] = df['windgust'] - df['windspeed']

    # Heat Index (cảm giác nóng, chỉ dùng khi temp ≥ 26°C)
    def calculate_heat_index_celsius(temp_c, humidity):
        T = temp_c * 9/5 + 32
        HI_f = 0.5 * (T + 61.0 + ((T - 68.0) * 1.2) + (humidity * 0.094))
        HI_c = (HI_f - 32) * 5/9
        return HI_c

    # Wind Chill (cảm giác lạnh, chỉ có ý nghĩa khi temp ≤ 10°C)
    def calculate_wind_chill(temp_c, windspeed_kmh):
        return 13.12 + 0.6215*temp_c - 11.37*(windspeed_kmh**0.16) + 0.3965*temp_c*(windspeed_kmh**0.16)

    # Tính riêng từng chỉ số 
    df['heat_index'] = df.apply(
        lambda r: calculate_heat_index_celsius(r['temp'], r['humidity']) if r['temp'] >= 26 else np.nan,
        axis=1
    )

    df['wind_chill'] = df.apply(
        lambda r: calculate_wind_chill(r['temp'], r['windspeed']) if r['temp'] <= 10 else np.nan,
        axis=1
    )

    # Tạo chỉ số tổng hợp: thermal_index
    df['thermal_index'] = np.where(
        df['temp'] >= 26, df['heat_index'],
        np.where(df['temp'] <= 10, df['wind_chill'], df['temp'])
    )
    
    # Wind direction categorization    
    df["winddir_cos"] = np.cos(np.deg2rad(df["winddir"]))
    df["winddir_sin"] = np.sin(np.deg2rad(df["winddir"]))
    df['moonphase_sin'] = np.sin(2 * np.pi * df['moonphase'])
    df['moonphase_cos'] = np.cos(2 * np.pi * df['moonphase'])

    def categorize_wind_direction(degree):
        if pd.isna(degree):
            return 'Unknown'
        elif 0 <= degree < 45:
            return 'Bắc_Đông Bắc_N_NE'
        elif 45 <= degree < 90:
            return 'Đông_Bắc_NE'
        elif 90 <= degree < 135:
            return 'Đông_Nam_SE'
        elif 135 <= degree < 180:
            return 'Nam_S'
        elif 180 <= degree < 225:
            return 'Tây_Nam_SW'
        elif 225 <= degree < 270:
            return 'Tây_W'
        elif 270 <= degree < 315:
            return 'Tây_Bắc_NW'
        elif 315 <= degree <= 360:
            return 'Bắc_N'
        else:
            return 'Unknown'
        
    # Ordinal Encode
    df['wind_category'] = df['winddir'].apply(categorize_wind_direction).astype('category')

    # Weather phenomena
    if 'precipcover' in df.columns:
        df['humid_foggy_day'] = ((df['precip'] < 1) & (df['precipcover'] > 50)).astype(int)
        df['moisture_index'] = df['humidity'] * (df['precipcover']/100) * (1 - df['precip']/50)
    df['vis_humidity_index'] = df['visibility'] * (1 - df['humidity'] / 100)
    
    return df

def enhanced_lag_features(df, forecast_horizon=5):
    """
    Tạo lag features tối ưu cho dự báo 5 ngày
    """
    df = df.copy()
    
    # Lag config mở rộng cho multi-horizon
    enhanced_lag_config = {
        'temp': [1, 2, 3, 7, 14],
        'tempmax': [1, 2, 3, 7],
        'tempmin': [1, 2, 3, 7],
        'thermal_index': [1, 2, 3, 7],
        'dew': [1, 2, 3, 7],
        'humidity': [1, 2, 3, 7],
        'sealevelpressure': [1, 2, 3, 7],
        'precip': [1, 2, 3, 7],
        'windgust': [1, 2, 3, 7],
        'windspeed': [1, 3, 7, 30],
        'winddir_cos': [1, 3, 7],
        'winddir_sin': [1, 3, 7],
        'cloudcover': [1, 2, 3, 7],
        'visibility': [1, 2, 3, 7],
        'moonphase': [7, 14],
        'solarradiation': [1, 2, 3],
        'solarenergy': [1, 2, 3],
        'uvindex': [1, 2, 3],
        'feelslikemax': [1, 2, 3],
        'feelslikemin': [1, 2, 3],
        'feelslike': [1, 2, 3],
        'temp_range': [1, 2, 3],
        'dew_spread': [1, 2, 3],
        'wind_variability': [1, 2, 3],
        # Thêm các features từ hourly data nếu có
        'temp_6h_variability': [1, 2, 3],
        'temp_6h_trend': [1, 2, 3],
    }

    lag_dataframes = []
    
    for feature, lags in enhanced_lag_config.items():
        if feature not in df.columns:
            continue
        for lag in lags:
            lag_col_name = f"{feature}_lag_{lag}"
            lag_series = df[feature].shift(lag)
            lag_series.name = lag_col_name
            lag_dataframes.append(lag_series)
    
    if lag_dataframes:
        lag_df = pd.concat(lag_dataframes, axis=1)
        df = pd.concat([df, lag_df], axis=1)
    
    print(f"Đã tạo tổng cộng {len(lag_dataframes)} lag features.")
    return df

def create_rolling_features(df):
    """
    Tạo cụ thể các rolling features theo mapping chi tiết.
    """
    df = df.copy()
    roll_dataframes = []

    # --- 1. Mapping chi tiết: feature -> loại và window cần tạo ---
    rolling_mapping = {
        'dew': {'mean': [7, 14], 'std': [7, 14]},
        'precip': {'mean': [7, 14], 'std': [7, 14], 'sum': [7,14]},
        'windgust': {'mean': [3, 7], 'std': [3, 7]},
        'windspeed': {'mean': [7], 'std': [7]},
        'winddir_sin': {'mean': [7], 'std': [7]},
        'winddir_cos': {'mean': [7], 'std': [7]},
        'cloudcover': {'mean': [3, 7], 'std': [7, 14]},
        'visibility': {'mean': [3, 7], 'std': [3, 7]},
        'solarradiation': {'mean': [3, 5], 'std': [3, 5], 'sum': [3]},
        'uvindex': {'mean': [3, 4, 5], 'std': [3, 4, 5]},
        'moonphase': {'mean': [7], 'std': [7]},
        'sealevelpressure': {'mean': [3, 5, 7], 'std': [3, 5, 7]},
        'tempmax': {'mean': [3, 5,7], 'std': [3, 5]},
        'tempmin': {'mean': [3, 5,7], 'std': [3, 5]},
        'temp': {'mean': [3, 5], 'std': [3, 5]},
        'feelslike': {'mean': [3, 5]},
        'feelslikemax': {'mean': [3, 5]},
        'feelslikemin': {'mean': [3, 5]},
        'humidity': {'mean': [7, 14], 'std': [7, 14]},
        'thermal_index': {'mean': [3,21]},
        'wind_variability': {'mean': [3, 5], 'std': [3]},
        'dew_spread': {'mean': [3, 5], 'std': [3]},
        'temp_range': {'mean': [3, 5], 'std': [3]},
        'rain_intensity': {'mean': [3, 5]},
        # Thêm rolling cho các features mới
        'temp_6h_variability': {'mean': [3, 5]},
        'temp_6h_trend': {'mean': [3, 5]},
        'temp_momentum_1d': {'mean': [3, 5]},
    }

    # --- 2. Tạo từng feature theo mapping ---
    for feature, roll_types in rolling_mapping.items():
        if feature not in df.columns:
            continue

        base_series = df[feature].shift(1)  # tránh data leakage

        for roll_type, windows in roll_types.items():
            for window in windows:
                col_name = f"{feature}_roll_{roll_type}_{window}"

                # Tính rolling cụ thể theo loại
                if roll_type == 'mean':
                    new_series = base_series.rolling(window, min_periods=window).mean()
                elif roll_type == 'std':
                    new_series = base_series.rolling(window, min_periods=window).std()
                elif roll_type == 'sum':
                    new_series = base_series.rolling(window, min_periods=window).sum()
                else:
                    continue

                new_series.name = col_name
                roll_dataframes.append(new_series)

    # --- 3. Ghép tất cả rolling features ---
    if roll_dataframes:
        roll_df = pd.concat(roll_dataframes, axis=1)
        df = pd.concat([df, roll_df], axis=1)
    
    df = df.drop(['heat_index','wind_chill'], axis=1)
    print(f"Đã tạo {len(roll_dataframes)} rolling features.")

    return df

def drop_base_features(df):
    """
    Loại bỏ các base feature, chỉ giữ lại derive
    """
    base = ['tempmax', 'tempmin', 'temp', 'feelslikemax', 'feelslikemin',
       'feelslike', 'dew', 'humidity', 'precip', 'precipprob', 'precipcover',
       'preciptype', 'snow', 'snowdepth', 'windgust', 'windspeed', 'winddir',
       'sealevelpressure', 'cloudcover', 'visibility', 'solarradiation',
       'solarenergy', 'uvindex', 'severerisk', 'moonphase']
    
    base = [feat for feat in base if feat in df.columns]
    
    return df.drop(base, axis=1)

def feature_engineering(df, forecast_horizon=5, is_drop_nan=False, is_linear=False, is_drop_base=False):
    """
    Thực hiện toàn bộ quy trình Feature Engineering cho bài toán dự báo đa đầu ra.
    Enhanced với hourly features và Vietnam seasonal patterns
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
    df = create_date_features(df, forecast_horizon)
    
    # 5. Specific features từ current data
    df = create_specific_features(df, is_linear)
        
    # 3. THÊM MỚI: Enhanced features từ hourly data
    df = enhance_hourly_features(df)
    
    
    # 6. THÊM MỚI: Momentum features
    df = create_momentum_features(df)
    
    # 7. Enhanced lag features
    df = enhanced_lag_features(df, forecast_horizon)
    
    # 8. Rolling features
    df = create_rolling_features(df)
    
    # 9. CHỌN LỌC features phù hợp cho lag/rolling
    target_features = ['temp_next_1','temp_next_2','temp_next_3','temp_next_4','temp_next_5']

    # 10. Dropping columns and rows
    df = df.dropna(subset=target_cols)

    if is_drop_nan:
        df = df.dropna()
        
    if is_drop_base:
        df = drop_base_features(df)

    # Tạo defragmented frame
    df = df.copy()

    print(f"Feature engineering hoàn thành. Tổng số features: {df.shape[1]}")
    return df, target_features