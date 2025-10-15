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
def create_lag_rolling(df, target='temp', lags=(1,2,3,7), windows=(3,7)):
    df = df.sort_values('datetime')
    for l in lags:
        df[f'{target}_lag_{l}'] = df[target].shift(l)
    for w in windows:
        df[f'{target}_roll_mean_{w}'] = df[target].rolling(w, min_periods=1).mean()
        df[f'{target}_roll_std_{w}'] = df[target].rolling(w, min_periods=1).std()
    df = df.dropna().reset_index(drop=True)  # drop initial NaNs
    return df

# 1. Xem có thêm biến interaction giữa các feature ko?

# 2. xử lí categorical theo hướng nào ? 

# Đang drop description và để tạm encoding đơn giản cho icon và condition, xem thử hướng khác ổn hơn ko
# icon bản chất là bản tổng hợp thông tin của feature khác, bị redundant thông tin
### vdu icon cloudy tương đương với feature cloudcover > 20% 
"""
Icon id	                Weather Conditions
snow	                Amount of snow is greater than zero
rain        	        Amount of rainfall is greater than zero
fog	                    Visibility is low (lower than one kilometer or mile)
wind	                Wind speed is high (greater than 30 kph or mph)
cloudy      	        Cloud cover is greater than 90% cover
partly-cloudy-day	    Cloud cover is greater than 20% cover during day time.
partly-cloudy-night	    Cloud cover is greater than 20% cover during night time.
clear-day	            Cloud cover is less than 20% cover during day time
clear-night	            Cloud cover is less than 20% cover during night time

"""
# xử lí conditions nếu chỉ map lại nhóm tương tự, rồi encoding (label hoặc onehot) thì đưa vào preprocessing.
# có drop description ko? nếu ko drop thì xử lí theo hướng preprocessing hay feature engineering (extract keywords, rồi tạo feature mới?) 