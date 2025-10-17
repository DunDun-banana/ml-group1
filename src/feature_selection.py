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


