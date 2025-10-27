"""
Reducing the number of input variables after feature engineering
- Unsupervised Methods: Ignore the target variable ( removing 
features with low variance or high correlation)
- Supervised Methods (using feature importance from Random Forest)
"""

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectPercentile, mutual_info_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.feature_selection import RFE, SequentialFeatureSelector
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor


    
class FeatureSelectionGradientBoosting(BaseEstimator, TransformerMixin):
    """Feature selection using GradientBoostingRegressor feature importances."""

    def __init__(self, top_k= 30 , random_state=42):
        """
        Parameters
        ----------
        top_k : int, default= 30 
            Number of top features to select based on feature importance.
        random_state : int, default=42
            Controls randomness for reproducibility.
        """
        self.top_k = top_k
        self.random_state = random_state
        self.selected_features_ = None

    def fit(self, X, y):
        model = GradientBoostingRegressor(random_state=self.random_state)
        model.fit(X, y)

        # Lấy feature importances
        importances = pd.Series(model.feature_importances_, index=X.columns)

        # Chọn top_k feature có importance cao nhất
        self.selected_features_ = importances.nlargest(self.top_k).index.tolist()
        return self

    def transform(self, X, y=None):
        if self.selected_features_ is None:
            raise RuntimeError("You must fit before calling transform().")
        return X[self.selected_features_]

    def get_feature_names_out(self):
        return self.selected_features_

class FeatureSelectionGradientBoosting1(BaseEstimator, TransformerMixin):
    """Feature selection using GradientBoostingRegressor feature importances."""

    def __init__(self, top_k=30, random_state=42):
        """
        Parameters
        ----------
        top_k : int, default= 30 
            Number of top features to select based on feature importance.
        random_state : int, default=42
            Controls randomness for reproducibility.
        """
        self.top_k = top_k
        self.random_state = random_state
        self.selected_features_ = None

    def fit(self, X, y):
        # 1. Tạo mô hình GradientBoostingRegressor cơ sở
        base_model = GradientBoostingRegressor(random_state=self.random_state)

        # 2. Bọc mô hình cơ sở bằng MultiOutputRegressor để xử lý y đa cột
        # n_jobs=-1 để huấn luyện các mô hình con song song, tăng tốc độ
        multi_output_model = MultiOutputRegressor(estimator=base_model, n_jobs=-1)

        # 3. Huấn luyện mô hình đa đầu ra
        multi_output_model.fit(X, y)

        # 4. Lấy feature importances từ mỗi mô hình con (estimator)
        # multi_output_model.estimators_ là một danh sách các mô hình GBR đã được huấn luyện
        all_importances = []
        for estimator in multi_output_model.estimators_:
            all_importances.append(estimator.feature_importances_)

        # 5. Tính trung bình feature importances trên tất cả các đầu ra
        # all_importances sẽ là list của các mảng, ví dụ 5 mảng cho 5 target
        # np.mean(..., axis=0) sẽ tính trung bình theo cột (tức là cho từng feature)
        mean_importances = np.mean(all_importances, axis=0)

        # 6. Chọn top_k features dựa trên importance trung bình
        importances_series = pd.Series(mean_importances, index=X.columns)
        self.selected_features_ = importances_series.nlargest(self.top_k).index.tolist()

        return self

    def transform(self, X, y=None):
        if self.selected_features_ is None:
            raise RuntimeError("You must fit before calling transform().")
        return X[self.selected_features_]

    def get_feature_names_out(self, input_features=None):
        return self.selected_features_

# Unsupervised method
class DropHighlyCorrelated1(BaseEstimator, TransformerMixin):
    """
    Drop feature khi có tương quan cao hơn threshold, dùng phương pháp loại dần.
    """
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.to_drop_ = []

    def fit(self, X, y=None):
        numeric_df = X.select_dtypes(include=['number']).copy()

        # Nếu có target "temp" lẫn trong X, bỏ ra
        if 'temp' in numeric_df.columns:
            numeric_df = numeric_df.drop(columns=['temp'])

        corr_matrix = numeric_df.corr().abs()
        # loại bỏ self-corr (đường chéo)
        np.fill_diagonal(corr_matrix.values, 0)

        # Loại bỏ từng feature một cách an toàn
        to_drop = set()
        cols = corr_matrix.columns

        for col in cols:
            # Bỏ qua nếu col đã bị drop
            if col in to_drop:
                continue

            # Tìm các feature tương quan cao với col
            high_corr = corr_matrix.index[corr_matrix[col] > self.threshold].tolist()
            # Loại tất cả các feature còn lại có corr cao hơn threshold với col
            to_drop.update(high_corr)

        self.to_drop_ = list(to_drop)
        return self

    def transform(self, X, y=None):
        return X.drop(columns=self.to_drop_, errors='ignore')

class FeatureSelectLGB1(BaseEstimator, TransformerMixin):
    """Feature selection using GradientBoostingRegressor feature importances."""

    def __init__(self, top_k= 30 , random_state=42):
        self.top_k = top_k
        self.random_state = random_state
        self.importance_type = 'gain'
        self.selected_features_ = None

    def fit(self, X, y):
        model =  LGBMRegressor(
            random_state=self.random_state,
            importance_type=self.importance_type,
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31
        )

        model.fit(X, y)

        # Lấy feature importances
        importances = pd.Series(model.feature_importances_, index=X.columns)

        # Chọn top_k feature có importance cao nhất
        self.selected_features_ = importances.nlargest(self.top_k).index.tolist()
        return self

    def transform(self, X, y=None):
        if self.selected_features_ is None:
            raise RuntimeError("You must fit before calling transform().")
        return X[self.selected_features_]

    def get_feature_names_out(self):
        return self.selected_features_

class FeatureSelectionLGB(BaseEstimator, TransformerMixin):
    """
    Feature selection using LightGBM feature importances (averaged over multiple targets).
    """

    def __init__(self, top_k=30, random_state=42, importance_type='gain'):
        self.top_k = top_k
        self.random_state = random_state
        self.importance_type = importance_type
        self.selected_features_ = None

    def fit(self, X, y):
        base_model = LGBMRegressor(
            random_state=self.random_state,
            importance_type=self.importance_type,
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31
        )

        # Bọc MultiOutput
        multi_model = MultiOutputRegressor(base_model, n_jobs=-1)
        multi_model.fit(X, y)

        # Lấy importance từ từng estimator
        all_importances = []
        for est in multi_model.estimators_:
            all_importances.append(est.feature_importances_)

        # Trung bình importance qua tất cả đầu ra
        mean_importances = np.mean(all_importances, axis=0)

        # Chọn top_k feature
        importance_series = pd.Series(mean_importances, index=X.columns)
        self.selected_features_ = importance_series.nlargest(self.top_k).index.tolist()

        return self

    def transform(self, X, y=None):
        if self.selected_features_ is None:
            raise RuntimeError("You must fit before calling transform().")
        # Lọc các cột được chọn
        return X[self.selected_features_]

    def get_feature_names_out(self, input_features=None):
        return self.selected_features_


class FeatureSelector1(BaseEstimator, TransformerMixin):
    """
    Flexible Feature Selection supporting Wrapper & Embedded methods.
    Supports multi-output y (multiple target columns).
    
    Supported methods:
        - 'rfe': Recursive Feature Elimination
        - 'forward': Sequential Forward Selection
        - 'backward': Sequential Backward Selection
        - 'lasso': L1 regularization
        - 'ridge': L2 regularization
        - 'elasticnet': Combination of L1 & L2
    """

    def __init__(self, method="rfe", top_k=30, random_state=42, alpha=1.0, l1_ratio=0.5):
        self.method = method
        self.top_k = top_k
        self.random_state = random_state
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.selected_features_ = None
        self.scaler_ = None  # sẽ lưu scaler để dùng ở transform

    def fit(self, X, y):
        X = pd.DataFrame(X)

        # === Chuẩn hóa dữ liệu ===
        self.scaler_ = StandardScaler()
        X_scaled = pd.DataFrame(self.scaler_.fit_transform(X), columns=X.columns)

        # === Wrapper Methods ===
        if self.method == "rfe":
            base_model = LinearRegression()
            selector = RFE(base_model, n_features_to_select=self.top_k)
            selector.fit(X_scaled, y)
            self.selected_features_ = X.columns[selector.support_].tolist()

        elif self.method in ["forward", "backward"]:
            base_model = LinearRegression()
            direction = "forward" if self.method == "forward" else "backward"
            selector = SequentialFeatureSelector(
                base_model,
                n_features_to_select=self.top_k,
                direction=direction,
                n_jobs=-1
            )
            selector.fit(X_scaled, y)
            self.selected_features_ = X.columns[selector.get_support()].tolist()

        # === Embedded Methods (support multi-output) ===
        elif self.method == "lasso":
            model = Lasso(alpha=self.alpha, max_iter=10000, random_state=self.random_state)
            model.fit(X_scaled, y)
            importance = np.abs(model.coef_)
            if importance.ndim > 1:  # multi-output case
                importance = np.mean(importance, axis=0)
            self.selected_features_ = X.columns[np.argsort(importance)[-self.top_k:]].tolist()

        elif self.method == "ridge":
            model = Ridge(alpha=self.alpha, max_iter=10000, random_state=self.random_state)
            model.fit(X_scaled, y)
            importance = np.abs(model.coef_)
            if importance.ndim > 1:
                importance = np.mean(importance, axis=0)
            self.selected_features_ = X.columns[np.argsort(importance)[-self.top_k:]].tolist()

        elif self.method == "elasticnet":
            model = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio, max_iter=10000, random_state=self.random_state)
            model.fit(X_scaled, y)
            importance = np.abs(model.coef_)
            if importance.ndim > 1:
                importance = np.mean(importance, axis=0)
            self.selected_features_ = X.columns[np.argsort(importance)[-self.top_k:]].tolist()

        else:
            raise ValueError(f"Unknown method '{self.method}'")

        return self

    def transform(self, X, y=None):
        if self.selected_features_ is None:
            raise RuntimeError("You must fit before calling transform().")
        if self.scaler_ is None:
            raise RuntimeError("Scaler not fitted — fit() must be called before transform().")

        # Áp dụng scaler đã học trước đó
        X_scaled = pd.DataFrame(self.scaler_.transform(X), columns=X.columns)
        return X_scaled[self.selected_features_]

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
