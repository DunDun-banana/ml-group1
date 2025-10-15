"""
Data Preprocessing for Hanoi Temperature Forecasting
- Load raw CSV
- Handle missing values, numeric/categorical coercion
...
- Save processed CSV
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from collections import Counter


# 1. Cho toàn bộ data set
def load_data(input_path: str):
    """
    Load dataset từ đường dẫn CSV.
    """
    try:
        df = pd.read_csv(input_path)
        print(f" Loaded data with shape: {df.shape}")
        return df.drop_duplicates()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {input_path}")


def transform_dtype(df: pd.DataFrame):
    """
    Chuyển đổi các kiểu dữ liệu phù hợp.
    """
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df = df.sort_values("datetime").set_index("datetime")
    for col in ["sunrise", "sunset"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def drop_description(df: pd.DataFrame):
    """
    Cân nhắc bỏ luôn description trên toàn bộ data set dc ko 
    tại dựa vào data understading 
    thì nó chỉ là detailed của icon, và description
    """
    if 'description' in df.columns:
        df = df.drop('description', axis=1)
        print("Dropped column: 'description'")
    else:
        print("Column 'description' not found, skip dropping.")
    return df

def basic_preprocessing(df: pd.DataFrame):
    """
    Chạy các bước preprocessing không phụ thuộc train/test
    """
    df = transform_dtype(df)
    df = drop_description(df)
    return df



# 2. Pipeline transformer chỉ fit trên train
# mục đích loại cột missing preciptype và severerisk
class HandleMissing(BaseEstimator, TransformerMixin):
    """Drop và fill missing values"""
    def __init__(self, drop_threshold=0.05):
        self.drop_threshold = drop_threshold
        self.cols_to_drop_ = []
        self.fill_values_ = {}

    def fit(self, X, y=None):
        # Drop columns trên train
        missing_ratio = X.isnull().mean()
        self.cols_to_drop_ = missing_ratio[missing_ratio > self.drop_threshold].index.tolist()

        # Fill remaining: khả năng là ko cần fill tại drop hết rồi
        X_train_kept = X.drop(columns=self.cols_to_drop_, errors='ignore')
        for col in X_train_kept.columns:
            if X_train_kept[col].isnull().any():
                if pd.api.types.is_numeric_dtype(X_train_kept[col]):
                    self.fill_values_[col] = X_train_kept[col].median()
                else:
                    self.fill_values_[col] = X_train_kept[col].mode()[0]
        return self

    def transform(self, X):
        X = X.drop(columns=self.cols_to_drop_, errors='ignore').copy()
        for col, val in self.fill_values_.items():
            if col in X.columns:
                X[col] = X[col].fillna(val)
        return X


# loại mấy cột như snow, snowdepth toàn 0
class DropLowVariance(BaseEstimator, TransformerMixin):
    """Drop numeric feature variance thấp"""
    def __init__(self, threshold=0.0):
        self.threshold = threshold
        self.kept_cols_ = None

    def fit(self, X, y=None):
        numeric_df = X.select_dtypes(include=['number'])
        if numeric_df.empty:
            self.kept_cols_ = [c for c in X.columns if c != 'stations']
            return self
        
        selector = VarianceThreshold(threshold=self.threshold)
        selector.fit(numeric_df)

        # Chỉ giữ numeric có variance > threshold
        kept_numeric = numeric_df.columns[selector.get_support()].tolist()
        # Chỉ giữ non-numeric trừ 'stations'
        kept_non_numeric = [c for c in X.select_dtypes(exclude=['number']).columns if c != 'stations']

        self.kept_cols_ = kept_numeric + kept_non_numeric
        return self

    def transform(self, X):
        return X[self.kept_cols_]


# DROP CATEGORICAL FEATURES: name, preciptype chỉ có rain và null
class DropCategorical(BaseEstimator, TransformerMixin):
    """Drop categorical feature theo unique ratio và description"""
    def __init__(self, unique_ratio_threshold=0.9):
        self.unique_ratio_threshold = unique_ratio_threshold
        self.to_drop_ = []

    def fit(self, X, y=None):
        cat_cols = X.select_dtypes(include=['object']).columns
        n_rows = len(X)
        self.to_drop_ = []
        for col in cat_cols:
            nunique_ratio = X[col].nunique() / n_rows
            if X[col].nunique() == 1 or nunique_ratio > self.unique_ratio_threshold:
                self.to_drop_.append(col)
        return self

    def transform(self, X):
        return X.drop(columns=self.to_drop_, errors='ignore')
    

class DropHighlyCorrelated(BaseEstimator, TransformerMixin):
    """
    Drop một trong hai feature có tương quan cao hơn thres
    Giữ lại feature có tương quan cao hơn với target.
    """
    def __init__(self, threshold=0.95, target_col='temp'):
        self.threshold = threshold
        self.target_col = target_col
        self.to_drop_ = []

    def fit(self, X, y=None):
        # Chỉ dùng X numeric
        numeric_df = X.select_dtypes(include=['number']).copy()
        if self.target_col in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=[self.target_col])

        corr_matrix = numeric_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Tính correlation feature với target y
        if y is not None:
            feature_target_corr = numeric_df.apply(lambda col: abs(np.corrcoef(col, y)[0, 1]))

        to_drop = set()
        for col in upper.columns:
            correlated_features = upper.index[upper[col] > self.threshold].tolist()
            for corr_col in correlated_features:
                if y is None:  # fallback nếu không có target
                    to_drop.add(corr_col)
                else:
                    # Giữ feature có correlation(target) cao hơn
                    if feature_target_corr[col] < feature_target_corr[corr_col]:
                        to_drop.add(col)
                    else:
                        to_drop.add(corr_col)

        self.to_drop_ = list(to_drop)
        return self

    def transform(self, X):
        return X.drop(columns=self.to_drop_, errors='ignore')
    
# để tạm encoding đơn giản cho conditions và icon
class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Encode categorical features như 'icon' và 'conditions' bằng LabelEncoder.
    - Chỉ fit trên train
    - Chuyển object -> int
    """
    def __init__(self, columns=None):
        self.columns = columns  # list các cột cần encode
        self.encoders_ = {}

    def fit(self, X, y=None):
        X = X.copy()
        for col in self.columns:
            if col in X.columns:
                le = LabelEncoder()
                le.fit(X[col].astype(str))
                self.encoders_[col] = le
        return self

    def transform(self, X):
        X = X.copy()
        for col, le in self.encoders_.items():
            if col in X.columns:
                X[col] = le.transform(X[col].astype(str))
        return X

# phân vân không biết xử lí stations cách nào
class StationPreprocessor(BaseEstimator, TransformerMixin):
    """
    fit trên train thôi
    Preprocessing cho cột 'stations':
    - Tách list các station trong từng sample
    - Chọn 'main_station' dựa trên frequency trên toàn train set
    - Encode main_station thành số nguyên (LabelEncoder)
    """
    def __init__(self):
        self.freqs_ = None
        self.encoder_ = LabelEncoder()

    def fit(self, X, y=None):
        # Tính frequency của tất cả station trong train set
        all_stations = []
        for s in X["stations"]:
            all_stations += str(s).split(",")
        self.freqs_ = Counter(all_stations)

        # Tạo main_station cho train set
        main_stations = [self._get_main_station(s) for s in X["stations"]]
        self.encoder_.fit(main_stations)
        return self

    def transform(self, X):
        X = X.copy()
        # Tạo main_station
        X["main_station"] = [self._get_main_station(s) for s in X["stations"]]
        # Encode
        X["main_station"] = self.encoder_.transform(X["main_station"])

        
        # Không giữ cột text gốc
        return X.drop('stations', axis = 1)

    def _get_main_station(self, station_list):
        stations = str(station_list).split(",")
        # Chọn station xuất hiện nhiều nhất trong train set
        return max(stations, key=lambda s: self.freqs_.get(s, 0))
    
class DropColumns(BaseEstimator, TransformerMixin):
    """Drop explicit columns"""
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns, errors='ignore')



class StationMultiHotEncoder(BaseEstimator, TransformerMixin):
    """
    fit train only
    Transformer xử lý feature 'stations' dạng list:
    - Split chuỗi station thành list
    - Encode mỗi station thành cột binary (multi-hot)
    - Fit trên train, giữ danh sách station duy nhất để áp dụng cho test
    """
    def __init__(self, column='stations'):
        self.column = column
        self.unique_stations_ = []

    def fit(self, X, y=None):
        # X[column] là dạng string, có thể chứa nhiều station cách nhau bởi ','
        all_stations = []
        for stations_str in X[self.column]:
            if pd.isna(stations_str):
                continue
            all_stations.extend([s.strip() for s in stations_str.split(',')])
        # danh sách station duy nhất trên train
        self.unique_stations_ = sorted(list(set(all_stations)))
        return self

    def transform(self, X):
        X = X.copy()
        # khởi tạo cột mới cho từng station
        for station in self.unique_stations_:
            X[f'station_{station}'] = 0

        # gán giá trị 1 nếu station có trong list
        for idx, stations_str in X[self.column].iteritems():
            if pd.isna(stations_str):
                continue
            for s in stations_str.split(','):
                s = s.strip()
                if s in self.unique_stations_:
                    X.at[idx, f'station_{s}'] = 1

        # có thể drop cột gốc nếu muốn
        X = X.drop(columns=[self.column])
        return X






