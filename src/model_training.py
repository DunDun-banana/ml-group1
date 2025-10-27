"""
Model Training for time series
- Uses time-based train/test split
- Trains a RandomForestRegressor
- Handles categorical features via one-hot encoding
- Saves model
"""
import joblib
import pandas as pd
import clearml
import random
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import optuna
import numpy as np

from pathlib import Path
from clearml import Logger, Task
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from lightgbm import LGBMRegressor

from src.data_preprocessing import load_data, basic_preprocessing
from src.pipeline import build_preprocessing_pipeline, build_GB_featture_engineering_pipeline
from src.feature_engineering import feature_engineering
from src.model_evaluation import evaluate_multi_output, evaluate


DATA_PATH = r"data\raw data\latest_3_year.csv"
OLD_MODEL_PATH = r"models\Current_model.pkl"
NEW = r"models\Update_model.pkl"
PIPE_1 = r"pipelines\preprocessing_pipeline.pkl"
PIPE_2 = r"pipelines\featureSelection_pipeline.pkl"

def load_new_data(data_path: str):
   """Load Data 3 năm gần nhất để retrain model"""
   df = load_data(data_path)
   return df

def preprocess_data(df):
   """Gọi pipeline preprocessing để xử lý data"""

   # 1. basic preprocessing for all data set
   df = basic_preprocessing(df=df)

   # 2. Chia train, test
   train_size = 0.8
   n = len(df)
   train_df = df.iloc[:int(train_size * n)]
   test_df = df.iloc[int(train_size * n):]

   # 3. Pipeline 1: preprocessing
   pipeline1 = build_preprocessing_pipeline()
   pipeline1.fit(train_df)

   train_processed = pipeline1.transform(train_df)
   test_processed = pipeline1.transform(test_df)
   joblib.dump(pipeline1, "pipelines/preprocessing_pipeline.pkl")

   # 4. Feature engineering
   train_feat, target_col = feature_engineering(train_processed)
   test_feat, _ = feature_engineering(test_processed)

   # 5. Chia X,y
   X_train = train_feat.drop(columns= target_col)
   y_train = train_feat[target_col]

   X_test = test_feat.drop(columns= target_col)
   y_test = test_feat[target_col]

   # 6. Pipeline 2: GB selection
   pipeline2 = build_GB_featture_engineering_pipeline(top_k= 35)
   X_train_sel = pipeline2.fit_transform(X_train, y_train['temp_next_1'])
   X_test_sel = pipeline2.transform(X_test)
   joblib.dump(pipeline2, "pipelines/featureSelection_pipeline.pkl")

   # 7. lưu lại processed data
   save_dir = "data"
   os.makedirs(save_dir, exist_ok=True)

   X_train_sel.to_csv(f"{save_dir}/New_X_train_sel.csv", index=False)
   y_train.to_csv(f"{save_dir}/New_y_train.csv", index=False)
   X_test_sel.to_csv(f"{save_dir}/New_X_test_sel.csv", index=False)
   y_test.to_csv(f"{save_dir}/New_y_test.csv", index=False)
   
   return X_train_sel, y_train, X_test_sel, y_test

def train_model(X_train, y_train, random_state=42, n_trials=25):
   """Training LightGBM MultiOutputRegressor với Optuna tuning."""
   # sẽ hiện khi chạy, không cần quan tâm, thông báo thôi ======> WARNING! Git diff too large to store (530kb), skipping uncommitted changes <======
   # === 1. Khởi tạo ClearML Task cho tuning session ===
   task = Task.init(
        project_name="Hanoi Temperature Forecast",
        task_name=f"Tuning_{datetime.now():%Y%m%d_%H%M}",
        tags=["optuna", "tuning"]
   )
   logger = Logger.current_logger()

   def objective(trial):
      boosting_type = trial.suggest_categorical('boosting_type', ['gbdt', 'dart'])
      params = {
            'boosting_type': boosting_type,
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10.0, log=True),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 0.3),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 1.0, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'subsample_freq': trial.suggest_int('subsample_freq', 1, 7),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_bin': trial.suggest_int('max_bin', 64, 512),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'objective': 'regression',
            'metric': 'rmse',
            'random_state': random_state,
            'n_jobs': -1,
            'verbose': -1
        }

      if boosting_type == 'dart':
            params['drop_rate'] = trial.suggest_float('drop_rate', 0.05, 0.5)
            params['skip_drop'] = trial.suggest_float('skip_drop', 0.3, 0.7)

      cv = TimeSeriesSplit(n_splits=5)
      rmse_scores = []

      for train_idx, val_idx in cv.split(X_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            base_model = LGBMRegressor(**params)
            model = MultiOutputRegressor(estimator=base_model, n_jobs=-1)
            model.fit(X_tr, y_tr)

            y_pred_val = model.predict(X_val)
            metrics = evaluate_multi_output(y_val, y_pred_val)
            rmse_scores.append(metrics["average"]["RMSE"])

      # === Log RMSE của trial lên ClearML ===
      mean_rmse = np.mean(rmse_scores)
      logger.report_scalar(
            title="Optuna Trials",
            series="RMSE",
            value=mean_rmse,
            iteration=trial.number
        )
      return mean_rmse

    # === 2. Chạy tuning Optuna ===
   sampler = optuna.samplers.TPESampler(seed=42)
   study = optuna.create_study(direction='minimize', sampler=sampler)
   study.optimize(objective, n_trials=n_trials, show_progress_bar=True, )
   print('Finished tuning')

   # === 3. Train lại mô hình tốt nhất trên toàn bộ tập train ===
   best_params = study.best_trial.params
   best_model = LGBMRegressor(**best_params)
   model = MultiOutputRegressor(estimator=best_model, n_jobs=-1)
   model.fit(X_train, y_train)

   # === 4. Lưu model
   joblib.dump(model, NEW)
   print('Đã lưu model mới')

   # Trả về model, best_params, và task để log tiếp
   return model, best_params, task


def evaluate_model(model, X_test_sel, y_test):
   """Đánh giá performance của model"""
   y_preds = model.predict(X_test_sel)

   # --- Kiểm tra xem là multi-output hay single-output ---
   if len(y_preds.shape) > 1 and y_preds.shape[1] > 1:
        metrics = evaluate_multi_output(y_test, y_preds)
   else:
        metrics = evaluate(y_test, y_preds)

   return metrics


def save_artifacts(model, best_param, metrics, task, output_dir="logs"):
   """Lưu model & pipeline & log metrics"""
   os.makedirs(output_dir, exist_ok=True)

   # Lưu metrics ra file
   with open(f"{output_dir}/metrics.txt", "w") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

   # Log final metrics
   # artifact
   task.upload_artifact("metrics", artifact_object= r'logs\metrics.txt')

   # scalar
   for k, v in metrics.items():
      if isinstance(v, dict):  # ví dụ: {"RMSE": 1.2, "R2": 0.9}
         for sub_k, sub_v in v.items():
               if isinstance(sub_v, (int, float)):
                  Logger.current_logger().report_scalar(
                     title=f"metrics_{k}", series=sub_k, value=sub_v, iteration=0
                  )
      elif isinstance(v, (int, float)):
         Logger.current_logger().report_scalar(
               title="metrics", series=k, value=v, iteration=0
         )


   # Log artifacts
   task.upload_artifact("model", artifact_object=NEW)
   task.upload_artifact("best_parameters", artifact_object=best_param)

   # Lưu retrain history
   record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "best_params": best_param
   }
   retrain_log_path = os.path.join(output_dir, "retrain_log.pkl")
   if os.path.exists(retrain_log_path):
        old_records = joblib.load(retrain_log_path)
   else:
        old_records = []
   old_records.append(record)
   joblib.dump(old_records, retrain_log_path)

   # Đóng task
   task.close()
   
   return

def retrain_pipeline(data_path):
   """Full pipeline: load data -> prepare data -> train, predict -> eval -> save"""
   # 1. Load data 3 năm gần nhất
   df = load_new_data(data_path)

   # 2. Prepare data
   X_train_sel, y_train, X_test_sel, y_test = preprocess_data(df)

   # 3. Train và Tune model lgb
   model, best_param, task = train_model(X_train_sel, y_train)
   metrics = evaluate_model(model, X_test_sel, y_test)

   # 4. save
   save_artifacts(model, best_param, metrics, task= task)

   print("Retrain complete. Metrics:", metrics)
   return metrics

if __name__ == "__main__":
   retrain_pipeline(data_path= DATA_PATH)
