"""
Model Training for time series
- Uses time-based train/test split
- Trains a RandomForestRegressor
- Handles categorical features via one-hot encoding
- Saves model
"""
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from src.model_evaluation import evaluate

def train_random_forest(train_df, val_df, test_df, target='temp', model_path='models/random_forest_model.pkl'):
   """
   Train RandomForestRegressor on time series data.
   Returns model, validation metrics, test metrics.
   """
   # 1. Separate features and target
   X_train = train_df.drop(columns=[target, 'datetime'], errors='ignore')
   y_train = train_df[target]
   X_val   = val_df.drop(columns=[target, 'datetime'], errors='ignore')
   y_val   = val_df[target]
   X_test  = test_df.drop(columns=[target, 'datetime'], errors='ignore')
   y_test  = test_df[target]

   # convert any remaining object columns to numeric or drop
   obj_cols = X_train.select_dtypes(include=['object']).columns
   if len(obj_cols) > 0:
      print("Warning: dropping remaining object columns:", obj_cols.tolist())
      X_train = X_train.drop(columns=obj_cols)
      X_val = X_val.drop(columns=obj_cols)
      X_test = X_test.drop(columns=obj_cols)

 
   # 2. Train RandomForest
   model = RandomForestRegressor(n_estimators=100, random_state=42)
   model.fit(X_train, y_train)

   # 3. Evaluate on validation
   y_val_pred = model.predict(X_val)
   val_metrics = evaluate(y_val, y_val_pred)
   print("Validation metrics:", val_metrics)

   # 4. Evaluate on test
   y_test_pred = model.predict(X_test)
   test_metrics = evaluate(y_test, y_test_pred)
   print("Test metrics:", test_metrics)

   # 5. Save model
   joblib.dump(model, model_path)
   print(f"RandomForest model saved to {model_path}")

   return model, val_metrics, test_metrics