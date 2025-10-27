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

def _ensure_dtindex(df: pd.DataFrame, tz: str | None = None) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        if "datetime" in df.columns:
            df = df.copy()
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
            df = df.set_index("datetime")
        else:
            raise ValueError("Cần DatetimeIndex hoặc cột 'datetime'.")
    df = df.sort_index()
    if tz is not None:
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert(tz)
    return df

def build_hourly_to_daily_dataset(
    hourly: pd.DataFrame,
    target_col: str = "temp",           # cột để tạo nhãn daily (vd temp)
    target_daily_func: str = "mean",    # "mean" | "max" | "min" | "median" | "sum"
    horizon_days: int = 1,              # dự báo ngày t+1 (hoặc t+H)
    feature_cols: list[str] | None = None,  # None = dùng tất cả cột numeric làm đặc trưng
    window_hours: tuple[int, ...] = (6, 12, 24, 36, 48, 72),  # lookback kết thúc tại 23:59 của ngày t-1
    agg_funcs: tuple[str, ...] = ("mean", "std", "min", "max", "median", "last"),
    tz: str | None = None,
    add_quality_flags: bool = True,     # cờ chất lượng (thiếu giờ, %missing,…)
    add_calendar_daily: bool = True,    # đặc trưng theo ngày (doy, dow, month …)
) -> pd.DataFrame:
    """
    Tạo một dòng/1 ngày:
      - Đặc trưng: tổng hợp từ các *hourly features* của các khoảng lookback kết thúc ở 23:59 ngày *t-1*.
      - Nhãn: aggregate(target_col) của ngày *t + horizon_days*.
    Không dùng dữ liệu của ngày cần dự báo trở đi => không rò rỉ.
    """

    H = _ensure_dtindex(hourly, tz=tz).copy()

    # Chọn cột đặc trưng
    if feature_cols is None:
        feature_cols = [c for c in H.columns if np.issubdtype(H[c].dtype, np.number)]
    if target_col not in H.columns:
        raise ValueError(f"Không thấy target_col='{target_col}' trong dữ liệu.")

    # ===== 1) Tạo nhãn daily ở ngày t =====
    # daily_target[t] = agg(target_col trong ngày t)
    daily_target = (
        H[[target_col]]
        .resample("D")
        .agg({target_col: target_daily_func})
        .rename(columns={target_col: f"{target_col}_daily_{target_daily_func}"})
    )

    # Dịch nhãn về t+horizon_days để dự báo tương lai
    y_col = f"{target_col}_daily_{target_daily_func}_t+{horizon_days}d"
    daily_target[y_col] = daily_target.iloc[:, 0].shift(-horizon_days)
    daily_target = daily_target[[y_col]]

    # ===== 2) Tạo khung ngày gốc (mỗi ngày 1 dòng) =====
    days = H.resample("D").size().to_frame("hours_present")
    days["hours_missing"] = 24 - days["hours_present"].clip(upper=24)

    # ===== 3) Tổng hợp HOURLY FEATURES thành DAILY FEATURES (kết thúc ở t-1) =====
    # Ta sẽ tạo đặc trưng cho *ngày t* bằng cách nhìn "cửa sổ kết thúc tại 23:59 của t-1".
    # Cách làm: đưa mốc về 00:00 của t, rồi lấy dữ liệu trước mốc đó.
    day_index = days.index  # mốc 00:00 mỗi ngày (t)
    feats_list = []

    # Precompute: tích lũy (expensive) -> dùng rolling on fixed timedelta is messy; ta dùng mask theo khoảng thời gian.
    # Làm gọn: với mỗi ngày t, với mỗi W giờ, lấy H.loc[(t - Wh, t)) và tính agg.
    # Để chạy nhanh hơn trên dữ liệu lớn, có thể vectorize bằng resample->expanding, nhưng đơn giản & an toàn trước.
    for t in day_index:
        # mốc trái/phải cho từng W: (t - W giờ) .. (t - 1 giờ)
        row = {}
        # Cờ chất lượng theo ngày t-1 (riêng cho 24h gần nhất)
        if add_quality_flags:
            last24 = H.loc[t - pd.Timedelta(hours=24): t - pd.Timedelta(seconds=1)]
            row["q_last24_hours_present"] = len(last24)
            row["q_last24_hours_missing"] = 24 - min(len(last24), 24)

        # Lặp từng cửa sổ
        for W in window_hours:
            start = t - pd.Timedelta(hours=W)
            end = t  # exclusive
            block = H.loc[start : t - pd.Timedelta(seconds=1)]
            if block.empty:
                # fill NaNs để sau dropna
                for c in feature_cols:
                    for f in agg_funcs:
                        row[f"{c}_win{W}h_{f}"] = np.nan
                continue

            # Tính số giờ hiện diện của block (cho cờ chất lượng theo W)
            if add_quality_flags:
                row[f"q_win{W}h_hours_present"] = len(block)
                row[f"q_win{W}h_hours_missing"] = W - min(len(block), W)

            # Tính các phép gộp trên từng cột feature
            for c in feature_cols:
                series = block[c].dropna()
                for f in agg_funcs:
                    if f == "mean":
                        val = series.mean() if not series.empty else np.nan
                    elif f == "std":
                        val = series.std(ddof=1) if len(series) > 1 else np.nan
                    elif f == "min":
                        val = series.min() if not series.empty else np.nan
                    elif f == "max":
                        val = series.max() if not series.empty else np.nan
                    elif f == "median":
                        val = series.median() if not series.empty else np.nan
                    elif f == "last":
                        # giá trị *cuối cùng trước mốc t* (t-ε)
                        val = series.iloc[-1] if not series.empty else np.nan
                    elif f == "sum":
                        val = series.sum() if not series.empty else np.nan
                    elif f == "p10":
                        val = series.quantile(0.10) if len(series) > 0 else np.nan
                    elif f == "p90":
                        val = series.quantile(0.90) if len(series) > 0 else np.nan
                    else:
                        raise ValueError(f"agg_funcs không hỗ trợ '{f}'")
                    row[f"{c}_win{W}h_{f}"] = val

            # Ví dụ đặc trưng tuỳ biến hữu ích cho thời tiết (nếu có các cột này)
            if {"precip", "rain", "snow"} & set(block.columns):
                precip_like = [c for c in ("precip", "rain", "snow") if c in block.columns]
                tot = block[precip_like].sum(axis=1)
                row[f"precip_win{W}h_sum"] = tot.sum()
                row[f"precip_win{W}h_hours>0"] = int((tot > 0).sum())

        feats_list.append(pd.Series(row, name=t))

    X = pd.DataFrame(feats_list).sort_index()

    # ===== 4) Thêm calendar features theo NGÀY t (không gây leak) =====
    if add_calendar_daily:
        dt = X.index
        cal = pd.DataFrame(index=dt)
        cal["doy"] = dt.dayofyear
        cal["doy_sin"] = np.sin(2 * np.pi * cal["doy"] / 366.0)
        cal["doy_cos"] = np.cos(2 * np.pi * cal["doy"] / 366.0)
        cal["dow"] = dt.dayofweek
        cal["is_weekend"] = (cal["dow"] >= 5).astype(int)
        cal["month"] = dt.month
        cal["month_sin"] = np.sin(2 * np.pi * cal["month"] / 12.0)
        cal["month_cos"] = np.cos(2 * np.pi * cal["month"] / 12.0)
        X = X.join(cal, how="left")

    # ===== 5) Ghép nhãn (t+horizon_days) & làm sạch =====
    dataset = X.join(daily_target, how="left")
    dataset = dataset.dropna(subset=[y_col]).dropna(axis=1, how="all")  # cột toàn NaN thì bỏ
    return dataset, y_col
