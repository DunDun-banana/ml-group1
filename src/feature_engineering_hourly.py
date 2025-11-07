"""
Quy trình Gộp: Tiền xử lý (Hourly) -> Tạo đặc trưng (Daily)

File này chứa 2 phần:
1. Định nghĩa hàm 'build_hourly_to_daily_dataset'.
2. Một khối `if __name__ == "__main__":` để chạy toàn bộ quy trình,
   bằng cách GỌI (import) các công cụ từ `data_preprocessing.py`.
"""

from __future__ import annotations
import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Sequence

# ================================================================== #
# PHẦN 1: IMPORT TỪ data_preprocessing.py
# (Thay vì định nghĩa lại các lớp)
# ================================================================== #
try:
    from data_preprocessing import (
        load_data, 
        basic_preprocessing, 
        HandleMissing, 
        DropLowVariance, 
        DropCategorical, 
        CategoricalEncoder
    )
    print("Đã import thành công các công cụ từ data_preprocessing.py")
except ImportError:
    print("LỖI: Không tìm thấy file 'data_preprocessing.py'.")
    print("Hãy đảm bảo file này nằm cùng thư mục hoặc trong một module.")
    # Bạn vẫn cần các import này để file chạy
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.preprocessing import LabelEncoder
    from sklearn.feature_selection import VarianceThreshold


# ================================================================== #
# PHẦN 2: HÀM TẠO ĐẶC TRƯNG HOURLY-TO-DAILY
# (Đây là logic cốt lõi của file này)
# ================================================================== #

def _ensure_dtindex(df: pd.DataFrame, tz: str | None = None) -> pd.DataFrame:
    """Đảm bảo index là DatetimeIndex, sort và chuyển timezone nếu cần."""
    if not isinstance(df.index, pd.DatetimeIndex):
        if "datetime" in df.columns:
            df = df.copy()
            df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
            df = df.set_index("datetime")
        else:
            pass 
    df = df.sort_index()
    if tz is not None:
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert(tz)
    return df

def build_hourly_to_daily_dataset(
    hourly: pd.DataFrame,
    target_col: str = "temp",
    target_daily_func: str = "mean",
    horizon_days: int = 1,
    feature_cols: list[str] | None = None,
    window_hours: tuple[int, ...] = (6, 12, 24, 36, 48, 72),
    agg_funcs: tuple[str, ...] = ("mean", "std", "min", "max", "median", "last"),
    tz: str | None = None,
    add_quality_flags: bool = True,
    add_calendar_daily: bool = True,
) -> pd.DataFrame:
    """
    Tạo một dòng/1 ngày từ dữ liệu giờ đã được làm sạch.
   
    """

    H = _ensure_dtindex(hourly, tz=tz).copy()

    # Chọn cột đặc trưng
    if feature_cols is None:
        print("Cảnh báo: 'feature_cols' là None. Sẽ dùng tất cả các cột số.")
        feature_cols = [c for c in H.columns if np.issubdtype(H[c].dtype, np.number)]
    if target_col not in H.columns:
        raise ValueError(f"Không thấy target_col='{target_col}' trong dữ liệu.")
        
    if target_col in feature_cols:
        feature_cols.remove(target_col)

    # ===== 1) Tạo nhãn daily ở ngày t =====
    daily_target = (
        H[[target_col]]
        .resample("D")
        .agg({target_col: target_daily_func})
        .rename(columns={target_col: f"{target_col}_daily_{target_daily_func}"})
    )

    y_col = f"{target_col}_daily_{target_daily_func}_t+{horizon_days}d"
    daily_target[y_col] = daily_target.iloc[:, 0].shift(-horizon_days)
    daily_target = daily_target[[y_col]]

    # ===== 2) Tạo khung ngày gốc (mỗi ngày 1 dòng) =====
    days = H.resample("D").size().to_frame("hours_present")
    days["hours_missing"] = 24 - days["hours_present"].clip(upper=24)

    # ===== 3) Tổng hợp HOURLY FEATURES thành DAILY FEATURES (kết thúc ở t-1) =====
    day_index = days.index
    feats_list = []

    print(f"Bắt đầu tổng hợp {len(day_index)} ngày...")
    for t in day_index:
        row = {}
        if add_quality_flags:
            last24 = H.loc[t - pd.Timedelta(hours=24): t - pd.Timedelta(seconds=1)]
            row["q_last24_hours_present"] = len(last24)
            row["q_last24_hours_missing"] = 24 - min(len(last24), 24)

        for W in window_hours:
            start = t - pd.Timedelta(hours=W)
            block = H.loc[start : t - pd.Timedelta(seconds=1)]
            if block.empty:
                for c in feature_cols:
                    for f in agg_funcs:
                        row[f"{c}_win{W}h_{f}"] = np.nan
                continue

            if add_quality_flags:
                row[f"q_win{W}h_hours_present"] = len(block)
                row[f"q_win{W}h_hours_missing"] = W - min(len(block), W)

            for c in feature_cols:
                series = block[c].dropna()
                for f in agg_funcs:
                    val = np.nan
                    if not series.empty:
                        if f == "mean": val = series.mean()
                        elif f == "std": val = series.std(ddof=1) if len(series) > 1 else 0
                        elif f == "min": val = series.min()
                        elif f == "max": val = series.max()
                        elif f == "median": val = series.median()
                        elif f == "last": val = series.iloc[-1]
                        elif f == "sum": val = series.sum()
                        elif f == "p10": val = series.quantile(0.10)
                        elif f == "p90": val = series.quantile(0.90)
                    row[f"{c}_win{W}h_{f}"] = val

            if {"precip", "rain", "snow"} & set(block.columns):
                precip_like = [c for c in ("precip", "rain", "snow") if c in block.columns]
                tot = block[precip_like].sum(axis=1)
                row[f"precip_win{W}h_sum"] = tot.sum()
                row[f"precip_win{W}h_hours>0"] = int((tot > 0).sum())

        feats_list.append(pd.Series(row, name=t))

    X = pd.DataFrame(feats_list).sort_index()
    print("...Hoàn thành tổng hợp.")

    # ===== 4) Thêm calendar features theo NGÀY t (không gây leak) =====
    if add_calendar_daily:
        print("Thêm đặc trưng lịch (ngày)...")
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
    dataset = dataset.dropna(subset=[y_col]).dropna(axis=1, how="all")
    return dataset, y_col


# ================================================================== #
# PHẦN 3: QUY TRÌNH (FLOW) KẾT HỢP ĐỂ CHẠY
# ================================================================== #

if __name__ == "__main__":

    CSV_PATH = r"data/raw data/hanoi_weather_data_hourly.csv"
    TARGET = "temp"

    print("--- BƯỚC 1: TẢI DỮ LIỆU THÔ ---")
    # Gọi hàm `load_data` từ `data_preprocessing.py`
    raw_df = load_data('data/raw data/hanoi_weather_data_hourly.csv')

    print("\n--- BƯỚC 2: TIỀN XỬ LÝ CƠ BẢN ---")
    # Gọi hàm `basic_preprocessing` từ `data_preprocessing.py`
    hourly_df = basic_preprocessing(raw_df)
    print(f"Hình dạng dữ liệu sau khi xử lý cơ bản: {hourly_df.shape}")
    
    print("\n--- BƯỚC 3: CHẠY TIỀN XỬ LÝ NÂNG CAO (TUẦN TỰ) ---")
    
    print("\n--- BƯỚC 3.1: XỬ LÝ MISSING ---")
    # Gọi lớp `HandleMissing` từ `data_preprocessing.py`
    missing_handler = HandleMissing(drop_threshold=0.15)
    hourly_df = missing_handler.fit_transform(hourly_df)
    print(f"Hình dạng sau khi xử lý missing: {hourly_df.shape}")

    print("\n--- BƯỚC 3.2: LOẠI BỎ BIẾN PHƯƠNG SAI THẤP ---")
    # Gọi lớp `DropLowVariance` từ `data_preprocessing.py`
    variance_dropper = DropLowVariance(threshold=0.0) 
    hourly_df = variance_dropper.fit_transform(hourly_df)
    print(f"Hình dạng sau khi lọc phương sai thấp: {hourly_df.shape}")

    print("\n--- BƯỚC 3.3: LOẠI BỎ BIẾN CATEGORICAL KHÔNG HỮU ÍCH ---")
    # Gọi lớp `DropCategorical` từ `data_preprocessing.py`
    cat_dropper = DropCategorical(unique_ratio_threshold=0.9) 
    hourly_df = cat_dropper.fit_transform(hourly_df)
    print(f"Hình dạng sau khi lọc categorical: {hourly_df.shape}")

    print("\n--- BƯỚC 3.4: MÃ HÓA CATEGORICAL (Conditions) ---")
    # Gọi lớp `CategoricalEncoder` từ `data_preprocessing.py`
    encoder = CategoricalEncoder(columns=['conditions']) 
    clean_hourly_df = encoder.fit_transform(hourly_df)
    print(f"Hình dạng sau khi mã hóa: {clean_hourly_df.shape}")
    
    # clean_hourly_df LÚC NÀY ĐÃ SẠCH SẼ
    
    print("\n--- BƯỚC 4: LỌC CÁC CỘT ĐẦU VÀO ĐỂ TẠO ĐẶC TRƯNG ---")
    # Đây là "danh sách trắng" 15 cột mà chúng ta đã thống nhất
    MY_FEATURE_COLS = [
        'temp', 'feelslike', 'humidity', 'dew', 'precip', 
        'windspeed', 'windgust', 'winddir', 'pressure', 
        'cloudcover', 'visibility',
        'solarradiation', 'solarenergy', 'uvindex', 
        'conditions' 
    ]
    
    final_feature_cols = [col for col in MY_FEATURE_COLS if col in clean_hourly_df.columns]
    print(f"Sẽ tạo đặc trưng từ {len(final_feature_cols)} cột đầu vào (đã lọc).")

    print("\n--- BƯỚC 5: TẠO BỘ DỮ LIỆU DAILY TỔNG HỢP ---")
    AGG_FUNCS = ("mean", "std", "min", "max", "last")
    WINDOWS = (12, 24, 48, 72)
    
    # Gọi hàm `build_hourly_to_daily_dataset` (định nghĩa ở PHẦN 2)
    daily_dataset, y_col = build_hourly_to_daily_dataset(
        hourly=clean_hourly_df,
        target_col=TARGET,
        target_daily_func="mean",
        horizon_days=1,
        feature_cols=final_feature_cols, # Chỉ dùng 15 cột đã chọn
        window_hours=WINDOWS,
        agg_funcs=AGG_FUNCS
    )

    print("\n--- HOÀN THÀNH ---")
    print(f"✅ Dataset daily cuối cùng:", daily_dataset.shape, "| Target:", y_col)
    print(daily_dataset.head(5))