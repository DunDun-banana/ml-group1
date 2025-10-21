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
from collections import Counter


"""
Reducing the number of input variables after feature engineering
- Unsupervised Methods: Ignore the target variable ( removing 
features with low variance or high correlation)
- Supervised Methods (using feature importance from Random Forest)
"""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectPercentile, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
import pandas as pd
import numpy as np


# SUPERVISED METHODS
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.feature_selection import SelectPercentile, mutual_info_regression


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

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
