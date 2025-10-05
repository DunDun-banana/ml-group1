"""
Model evaluation
- RMSE, R2, MAPE
"""

from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred)/y_true)) * 100
    return {'RMSE': rmse, 'R2': r2, 'MAPE': mape}
