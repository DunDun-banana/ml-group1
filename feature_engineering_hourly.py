"""
Feature Engineering (Hourly) — Safe for 120-step forecasting

Mục tiêu:
- Dùng dữ liệu theo GIỜ để dự báo nhiệt độ 120 giờ tiếp theo (5 ngày).
- Tránh leak tương lai: rolling luôn shift(1), drop toàn bộ "lag 0" sau khi tạo đặc trưng.
- Cho phép chạy 2 chế độ:
    * ar_only=True  : chỉ đặc trưng từ thời gian + lag/rolling của TARGET (an toàn khi không có dự báo exogenous).
    * ar_only=False : cho phép thêm một số đặc trưng khí tượng khác (nếu bạn có dữ báo exogenous cho tương lai).

Hàm chính:
- feature_engineering_hourly(df, target="temp", forecast_horizon=1, ar_only=True, lags=(...), windows=(...))

Utils kèm theo:
- make_multi_horizon_targets(df, target='temp', horizons=120): tạo 120 cột nhãn T+1..T+120.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Iterable, List, Tuple, Optional, Sequence


def _ensure_datetime_index(df: pd.DataFrame):
    "Đảm bảo index lf Datetime index vầ được sort tăng dần"
    if not isinstance(df.index,pd.DatetimeIndex):
        if 'datetime' in df.columns:
            df = df.copy()
            df['datetime'] = pd.to_datetime(df['datetime'],errors = 'coerce')
            df = df.set_index('datetime')
        else:
            raise ValueError('DataIndex dont have datetime columns')
    return df.sort_index()

def _drop_future_features(df:pd.DataFrame,cols_to_drop:Sequence[str],keep:Sequence[str]):
    """
    Loại bỏ các feature 'lag 0' (tức là các cột gốc, có nguy cơ leak),
    nhưng giữ lại một số cột cần thiết trong keep (vd: target hiện tại để tạo lag tiếp theo).
    """
    cols_to_drop = list(cols_to_drop)
    for i in keep:
        if i in cols_to_drop:
            cols_to_drop.remove(i)
    return df.drop(columns=cols_to_drop,errors='ignore')
# ----------------------------- #
# Time features (hourly)
# ----------------------------- #

def create_date_feature_hourly(df:pd.DataFrame):
    """
    Tạo đặc trưng theo GIỜ:
    - hour, hour_sin, hour_cos
    - weekday (0-6), month (1-12) + cyclical month
    (Không động tới sunrise/sunset để tránh rối dữ liệu theo giờ)
    """
    df = _ensure_datetime_index(df).copy()
    dt = df.index

    df["hour"] = dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["weekday"] = dt.weekday
    df["month"] = dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    # Seasonal signal over the year
    df["dayofyear"] = dt.dayofyear
    df["dayofyear_sin"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["dayofyear_cos"] = np.cos(2 * np.pi * df["dayofyear"] / 365)


    return df

# ----------------------------- #
# Lag & Rolling (hourly)
# ----------------------------- #

def create_lag_rolling_hourly(
    df: pd.DataFrame,
    columns: Sequence[str],
    lags: Sequence[int] = (1, 3, 6, 12, 24, 48, 72),
    windows: Sequence[int] = (3, 6, 12, 24, 48),
    forecast_horizon: int = 1,
    fill_method: str = "bfill",
) -> pd.DataFrame:
    """
    Tạo lag và rolling cho nhiều cột theo GIỜ.
    - Luôn shift(1) cho rolling để tránh leak.
    - Chỉ tạo các lag >= forecast_horizon.

    fill_method: 'bfill' | 'ffill' | 'mean'
    """
    df = _ensure_datetime_index(df).copy()
    columns = [c for c in columns if c in df.columns]
    if not columns:
        return df

    valid_lags = [l for l in lags if l >= forecast_horizon]

    new_parts = []

    if valid_lags:
        lag_frames = {
            f"{col}_lag_{l}": df[col].shift(l) for col in columns for l in valid_lags
        }
        new_parts.append(pd.concat(lag_frames, axis=1))

    roll_mean = {
        f"{col}_roll_mean_{w}": df[col].shift(1).rolling(window=w, min_periods=1).mean()
        for col in columns for w in windows
    }
    roll_std = {
        f"{col}_roll_std_{w}": df[col].shift(1).rolling(window=w, min_periods=1).std()
        for col in columns for w in windows
    }
    new_parts.append(pd.concat({**roll_mean, **roll_std}, axis=1))

    out = pd.concat([df] + new_parts, axis=1)

    if fill_method == "bfill":
        out = out.bfill()
    elif fill_method == "ffill":
        out = out.ffill()
    elif fill_method == "mean":
        out = out.fillna(out.mean(numeric_only=True))
    else:
        raise ValueError("fill_method must be one of: 'bfill' | 'ffill' | 'mean'")

    return out

# ----------------------------- #
# Specific features (safe set)
# ----------------------------- #

def create_specific_features_hourly(
        df:pd.DataFrame,
        target: str = 'temp'
):
    """
    Một số đặc trưng đơn giản, an toàn cho 120-step:
    - temp_diff_1h, temp_diff_3h (gradient ngắn hạn)
    - Nếu có tempmax/tempmin theo giờ thì temp_range (không bắt buộc)
    - dew_spread (nếu có 'dew')
    """
    df = df.copy()
    if target in df.columns:
        df[f"{target}_diff_1h"] = df[target].diff(1)
        df[f"{target}_diff_3h"] = df[target].diff(3)

    if{'tempmax','tempmin'}.issubset(df.columns):
        df['temp_range'] = df['tempmax'] - df['tempmin']

    if 'dew' in df.columns:
        base = df['tempmin'] if 'tempmin' in df.columns else df.get(target,None)
        if base is not None:
            df['dew_spread'] = base - df['dew']
        
    return df

# ----------------------------- #
# MAIN API
# ----------------------------- #

def feature_engineering_hourly(
    df: pd.DataFrame,
    target: str = "temp",
    forecast_horizon: int = 1,
    ar_only: bool = True,
    lags: Sequence[int] = (1, 3, 6, 12, 24, 48, 72),
    windows: Sequence[int] = (3, 6, 12, 24, 48),
    fill_method: str = "bfill",
    extra_columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Tổng hợp toàn bộ FE HOURLY.

    Tham số:
    - target            : tên cột mục tiêu (mặc định 'temp')
    - forecast_horizon  : khoảng nhìn trước tối thiểu (>=1)
    - ar_only           : True -> chỉ dùng đặc trưng thời gian + lag/rolling của target
                          False -> cho phép thêm đặc trưng khác (nếu có exogenous forecast)
    - lags, windows     : tập lag/rolling theo giờ
    - fill_method       : 'bfill' | 'ffill' | 'mean'
    - extra_columns     : (tùy chọn) danh sách cột exogenous bạn muốn tạo lag/rolling (khi ar_only=False)

    Trả về:
    - DataFrame đã sinh đặc trưng và loại bỏ toàn bộ cột "lag 0" (trừ các cột cần giữ).
    """
    if forecast_horizon < 1:
        raise ValueError("forecast_horizon must be >= 1")

    df = _ensure_datetime_index(df).copy()

    # 1) Time features (hourly)
    df = create_date_feature_hourly(df)

    # 2) Specific safe features
    df = create_specific_features_hourly(df, target=target)

    # 3) Đánh dấu các cột 'lag 0' để drop sau cùng (tránh leak)
    lag0_cols = df.columns.copy()

    # 4) Lag & Rolling
    if ar_only:
        cols_for_lr = [c for c in [target] if c in df.columns]
    else:
        # cho phép thêm exogenous nếu đã có (vd có dự báo tương lai)
        cols_for_lr = [target]
        if extra_columns:
            cols_for_lr += [c for c in extra_columns if c in df.columns]
        # loại trùng
        cols_for_lr = list(dict.fromkeys(cols_for_lr))

    df = create_lag_rolling_hourly(
        df,
        columns=cols_for_lr,
        lags=lags,
        windows=windows,
        forecast_horizon=forecast_horizon,
        fill_method=fill_method,
    )

    # 5) Drop toàn bộ cột gốc (lag 0) để không dùng thông tin hiện tại của tương lai
    keep_cols = [target]  # vẫn giữ target hiện tại để tiếp tục tạo lag ở bước khác nếu cần
    df = _drop_future_features(df, cols_to_drop=lag0_cols, keep=keep_cols)

    # 6) Bỏ các hàng đầu tiên bị NaN do lag/rolling (nếu còn)
    df = df.dropna(how="any")

    return df

# ----------------------------- #
# Multi-horizon label maker
# ----------------------------- #

def make_multi_horizon_targets(
    df: pd.DataFrame,
    target: str = "temp",
    horizons: int = 120,
) -> pd.DataFrame:
    """
    Tạo 120 nhãn tương lai: target_t+1 ... target_t+H
    Lưu ý: Gọi SAU khi đã tạo xong đặc trưng (FE) để tránh rớt dữ liệu không cần thiết.
    """
    if target not in df.columns:
        raise KeyError(f"Target '{target}' not found in DataFrame.")
    df = df.copy()
    for h in range(1, horizons + 1):
        df[f"{target}_t+{h}"] = df[target].shift(-h)
    # Bỏ các hàng cuối không đủ nhãn
    return df.iloc[:-horizons].dropna(how="any")