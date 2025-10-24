"""
Reducing the number of input variables after feature engineering
- Unsupervised Methods: Ignore the target variable ( removing 
features with low variance or high correlation)
- Supervised Methods (using feature importance from Random Forest)
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectPercentile, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.feature_selection import SelectPercentile, mutual_info_regression


# SUPERVISED METHODS
class SelectPercentileMutualInfoRegression(BaseEstimator, TransformerMixin):
    """
    Feature selector using mutual information for regression tasks.
    Selects a given percentile of top features with the highest scores.

    Parameters
    ----------
    percentile : int, default=10
        Percent of top features to select (0 < percentile <= 100).
    random_state : int, optional
        Random state for reproducibility.
    """

    def __init__(self, percentile=90, random_state=None):
        self.percentile = percentile
        self.random_state = random_state
        self.selector_ = None
        self.selected_features_ = None

    def fit(self, X, y):
        self.selector_ = SelectPercentile(
            score_func=lambda X, y: mutual_info_regression(X, y, random_state=self.random_state),
            percentile=self.percentile
        )
        self.selector_.fit(X, y)
        self.selected_features_ = X.columns[self.selector_.get_support()]
        return self

    def transform(self, X):
        if self.selector_ is None:
            raise RuntimeError("You must fit the selector before calling transform().")
        return X[self.selected_features_]

    def get_support(self):
        """Return a boolean mask of selected features."""
        return self.selector_.get_support()

    def get_feature_names_out(self):
        """Return the names of selected features."""
        return self.selected_features_


class FeatureSelectionRandomForest(BaseEstimator, TransformerMixin):
    """Feature selection using RandomForestRegressor feature importances."""
    def __init__(self, top_k=15, random_state=42):
        """
        Parameters
        ----------
        top_k : int, default=15
            Number of top features to select based on importance.
        random_state : int, default=42
            Controls randomness for reproducibility.
        """
        self.top_k = top_k
        self.random_state = random_state
        self.selected_features_ = None

    def fit(self, X, y):
        model = RandomForestRegressor(random_state=self.random_state)
        model.fit(X, y)

        # Tính importance cho từng feature
        importances = pd.Series(model.feature_importances_, index=X.columns)

        # Chọn top_k feature quan trọng nhất
        self.selected_features_ = importances.nlargest(self.top_k).index.tolist()
        return self

    def transform(self, X, y=None):
        if self.selected_features_ is None:
            raise RuntimeError("You must fit before calling transform().")
        return X[self.selected_features_]

    def get_feature_names_out(self):
        return self.selected_features_


class FeatureSelectionExtraTrees(BaseEstimator, TransformerMixin):
    """Feature selection using ExtraTreesRegressor feature importances."""
    def __init__(self, top_k=30, random_state=42):
        """
        Parameters
        ----------
        top_k : int, default=15
            Number of top features to select based on feature importance.
        random_state : int, default=42
            Controls randomness for reproducibility.
        """
        self.top_k = top_k
        self.random_state = random_state
        self.selected_features_ = None

    def fit(self, X, y):
        model = ExtraTreesRegressor(random_state=self.random_state)
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

import lightgbm as lgb # Cần import lightgbm

class FeatureSelectionLightGBM(BaseEstimator, TransformerMixin):
    """Feature selection using LightGBM Regressor feature importances."""
    def __init__(self, top_k=30, random_state=42, n_estimators=100):
        """
        Parameters
        ----------
        top_k : int, default=30
            Number of top features to select based on feature importance.
        random_state : int, default=42
            Controls randomness for reproducibility.
        n_estimators : int, default=100
            Number of boosting iterations (trees) for the LightGBM model.
        """
        self.top_k = top_k
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.selected_features_ = None

    def fit(self, X, y):
        # Khởi tạo và huấn luyện LightGBM Regressor
        model = lgb.LGBMRegressor(
            random_state=self.random_state, 
            n_estimators=self.n_estimators,
            n_jobs=-1 # Sử dụng tất cả các cores
        )
        model.fit(X, y)

        # Lấy feature importances
        # LightGBM trả về một mảng numpy cho feature_importances_
        importances = pd.Series(model.feature_importances_, index=X.columns)

        # Chọn top_k feature có importance cao nhất
        self.selected_features_ = importances.nlargest(self.top_k).index.tolist()
        return self

    def transform(self, X, y=None):
        if self.selected_features_ is None:
            raise RuntimeError("You must fit before calling transform().")
        return X[self.selected_features_]

    def get_feature_names_out(self):
        """Return the names of selected features."""
        return self.selected_features_
# Unsupervised method
class DropHighlyCorrelated(BaseEstimator, TransformerMixin):
    """
    input là X
    Drop feature xuất hiện trước khi cả hai feature có tương quan cao hơn threshold.
    """
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self.to_drop_ = []

    def fit(self, X, y=None):
        # CHỈ sử dụng features, không dùng target
        numeric_df = X.select_dtypes(include=['number'])

        # trường hợp khi chưa tách X,y phải tách temp
        if 'temp' in numeric_df.columns:
            numeric_df = numeric_df.drop('temp', axis = 1)

        corr_matrix = numeric_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = set()
        for col in upper.columns:
            correlated_features = upper.index[upper[col] > self.threshold].tolist()
            for corr_col in correlated_features:
                # Simple rule: giữ feature xuất hiện trước
                to_drop.add(corr_col)
        
        self.to_drop_ = list(to_drop)
        return self

    def transform(self, X, y = None):
        return X.drop(columns=self.to_drop_, errors='ignore')
