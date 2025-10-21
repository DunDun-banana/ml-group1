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
from sklearn.neighbors import LocalOutlierFactor
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

def drop_redundant_column(df: pd.DataFrame):
    """
    Drop description (trùng thông tin với conditions), 
    icon (trùng thông tin với các thông số khác), 
    severisk sẽ drop thông qua handle missing khi miss quá 5% (feature này chỉ available từ năm 2023)
    stations cho thấy insignificant khi kiểm tra ở data_understanding
    name chỉ có 1 location Hanoi
    
    Parameters:
    - df: DataFrame cần xử lý (entire data)

    Returns:
    - df: DataFrame sau khi đã drop column
    """
    # Drop 'description' nếu có
    if 'description' in df.columns:
        df = df.drop('description', axis=1)
        print("Dropped column: 'description'")
    else:
        print("Column 'description' not found, skip dropping.")

    # Drop 'icon' nếu có
    if 'icon' in df.columns:
        df = df.drop('icon', axis=1)
        print("Dropped column: 'icon'")
    else:
        print("Column 'icon' not found, skip dropping.")

    # drop station vì nhận xét thấy insignificant
    if 'stations' in df.columns:
        df = df.drop('stations', axis=1)
        print("Dropped column: 'stations'")
    else:
        print("Column 'station' not found, skip dropping.")

    # drop name chỉ có hanoi
    if 'name' in df.columns:
        df = df.drop('name', axis=1)
        print("Dropped column: 'name'")
    else:
        print("Column 'name' not found, skip dropping.")

    return df.drop_duplicates()


def basic_preprocessing(df: pd.DataFrame):
    """
    Chạy các bước preprocessing không phụ thuộc train/test
    """
    df = transform_dtype(df)
    df = drop_redundant_column(df)
    return df


# 2. Pipeline transformer chỉ fit trên train
# ko cần thiết lắm
class HandleOutlier(BaseEstimator, TransformerMixin):
    """
    Xử lý outliers bằng Local Outlier Factor (LOF) riêng sau khi chạy preprocessing pipeline

    Parameters
    ----------
    contamination : float, default=0.05
        Tỷ lệ dự đoán outlier trong dữ liệu.
    n_neighbors : int, default=20
        Số lượng hàng xóm để tính mật độ cục bộ.
    drop : bool, default=True
        Nếu True -> drop các outliers.
        Nếu False -> chỉ thêm cột 'is_outlier' đánh dấu.
    target_col : str, default='temp'
        Tên cột target cần loại trừ khi tính LOF tránh gây nhiễu khi lọc outlier.
    """

    def __init__(self, contamination=0.05, n_neighbors=20, drop=False, target_col='temp'):
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.drop = drop
        self.target_col = target_col
        self.model_ = None
        self.outlier_mask_ = None

    def fit(self, X, y = None):
        # Bỏ cột target 
        X_features = X.drop(columns=[self.target_col], errors='ignore')

        # Lấy các cột numeric
        numeric_df = X_features.select_dtypes(include=[np.number])
        if numeric_df.empty:
            raise ValueError("Không có cột numeric nào để tính LOF.")

        # Fit LOF trên tập train (numeric features)
        self.model_ = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination
        )
        self.model_.fit(numeric_df)
        return self

    def transform(self, X):
        # Bỏ cột target trước khi predict
        X_features = X.drop(columns=[self.target_col], errors='ignore')

        if self.model_ is None:
            raise RuntimeError("Cần gọi fit() trước khi transform().")

        numeric_df = X_features.select_dtypes(include=[np.number])
        y_pred = self.model_.fit_predict(numeric_df)
        self.outlier_mask_ = (y_pred == -1)

        if self.drop:
            print(f"Dropping {self.outlier_mask_.sum()} outliers ({self.outlier_mask_.mean()*100:.2f}%).")
            X_cleaned = X.loc[~self.outlier_mask_]
            return X_cleaned
        else:
            X_marked = X.copy()
            X_marked['is_outlier'] = self.outlier_mask_
            return X_marked
 
# mục đích loại cột missing preciptype
class HandleMissing(BaseEstimator, TransformerMixin):
    """Drop và fill missing values"""
    def __init__(self, drop_threshold=0.05):
        self.drop_threshold = drop_threshold
        self.cols_to_drop_ = []
        self.fill_values_ = {}

    def fit(self, X, y=None):
        # Tính tỷ lệ missing
        missing_ratio = X.isnull().mean()
        self.cols_to_drop_ = missing_ratio[missing_ratio > self.drop_threshold].index.tolist()

        # Xác định giá trị fill
        X_train_kept = X.drop(columns=self.cols_to_drop_, errors='ignore')
        for col in X_train_kept.columns:
            if X_train_kept[col].isnull().any():
                if pd.api.types.is_numeric_dtype(X_train_kept[col]):
                    self.fill_values_[col] = X_train_kept[col].median()
                else:
                    self.fill_values_[col] = X_train_kept[col].mode()[0]
        return self

    def transform(self, X, y = None):
        X = X.drop(columns=self.cols_to_drop_, errors='ignore')

        for col, val in self.fill_values_.items():
            if col in X.columns:
                X = X[col].fillna(val)

        return X


# loại các cột numeric chỉ có 1 value (snow, snowdepth)
class DropLowVariance(BaseEstimator, TransformerMixin):
    """Drop numeric feature variance thấp"""
    def __init__(self, threshold=0.0):
        self.threshold = threshold
        self.kept_cols_ = None

    def fit(self, X, y = None): # input là df chưa tách X,y 
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

    def transform(self, X, y = None):
        return X[self.kept_cols_]


# # loại các cột categorical chỉ có 1 value (preciptype)
class DropCategorical(BaseEstimator, TransformerMixin):
    """Drop categorical feature có nunique == 1 hoặc unique ratio >= 0.9 """
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

    def transform(self, X, y = None):
        return X.drop(columns=self.to_drop_, errors='ignore')
    
    
# để tạm encoding đơn giản cho conditions và icon
class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Encode categorical features 'conditions' bằng LabelEncoder.
    - Chỉ fit trên train
    - Chuyển object -> int
    """
    def __init__(self, columns= ['conditions']):
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

    def transform(self, X, y = None):
        X = X.copy()
        for col, le in self.encoders_.items():
            if col in X.columns:
                X[col] = le.transform(X[col].astype(str))
        return X


