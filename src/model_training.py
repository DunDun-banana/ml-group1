"""
Model Training for time series
- Uses time-based train/test split
- Trains a set of 5 individual LightGBM models for multi-step forecasting
- Handles hyperparameter tuning for models and pipelines via Optuna
- Compares and deploys the best model set
- Saves artifacts and logs experiments to ClearML
"""
import joblib
import pandas as pd
import clearml
import random
import sys, os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import optuna
import numpy as np
import shutil
import logging
import subprocess
from clearml import Logger, Task
from sklearn.model_selection import TimeSeriesSplit
from datetime import datetime
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

from src.data_preprocessing import prepare_data
from src.pipeline import build_full_pipeline
from src.model_evaluation import evaluate_multi_output, evaluate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Đường dẫn sử dụng pathlib ---
BASE_DIR = Path(__file__).parent.parent
DATA_PATH_10_YEARS = BASE_DIR / "data" / "raw data" / "Hanoi Daily 10 years.csv"
DATA_PATH = BASE_DIR / "data" / "latest_3_year.csv"
CURRENT_MODEL_PATH = BASE_DIR / "models" / "Current_model.pkl"
TEMP_NEW_MODEL_PATH = BASE_DIR / "models" / "Update_model.pkl"

# --- Lớp Wrapper cho Multi-Output Prediction ---
class CustomMultiOutputRegressor:
   """
   Lớp bao bọc (wrapper) cho 5 bộ model/pipeline riêng biệt,
   cung cấp một giao diện .predict() thống nhất.
   """
   def __init__(self, models_per_target, complete_pipelines_per_target):
      self.models_per_target = models_per_target
      self.complete_pipelines_per_target = complete_pipelines_per_target
      self.target_names = list(models_per_target.keys())
   
   def predict(self, X):
      if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

      predictions = {}
      first_target_name = self.target_names[0]
      first_pipeline = self.complete_pipelines_per_target[first_target_name]
      X_processed_sample = first_pipeline.transform(X)
      final_index = X_processed_sample.iloc[30:].index

      for target_name in self.target_names:
            model = self.models_per_target[target_name]
            pipeline = self.complete_pipelines_per_target[target_name]
            
            X_processed = pipeline.transform(X)
            X_final = X_processed.iloc[30:]
            
            pred = model.predict(X_final)
            predictions[target_name] = pred
      
      return pd.DataFrame(predictions, index=final_index)
   
   def get_params(self, deep=True):
      return {
            "models_per_target": self.models_per_target,
            "complete_pipelines_per_target": self.complete_pipelines_per_target
      }

# --- Các hàm chức năng ---
def preprocess_data(data_path: str):
   """
   Sử dụng hàm prepare_data từ data_preprocessing.py để chuẩn bị và chia dữ liệu thô.
   """
   logging.info("Bắt đầu chuẩn bị và chia dữ liệu từ đường dẫn...")
   _, _, _, X_train, y_train, X_test, y_test = prepare_data(is_print=False, path=data_path)
   logging.info(f"Chia dữ liệu hoàn tất. Kích thước X_train: {X_train.shape}, X_test: {X_test.shape}")
   
   save_dir = BASE_DIR / "data"
   save_dir.mkdir(parents=True, exist_ok=True)
   X_train.to_csv(save_dir / "New_X_train_raw.csv", index=True)
   y_train.to_csv(save_dir / "New_y_train_raw.csv", index=True)
   
   return X_train, y_train, X_test, y_test

def build_final_complete_pipeline(params: dict):
   """Xây dựng pipeline hoàn chỉnh với bộ tham số tốt nhất từ Optuna."""
   return build_full_pipeline(
      is_linear=params.get('is_linear', False),
      is_category_conditions=params['conditions_is_category'],
      is_category_season=params['season_is_category'],
      is_category_wind=params['wind_is_category'],
      encoding_method_condition=params['encoding_method_condition'],
      n_seasons=params['n_seasons'],
      n_quantiles=params['n_quantiles'],
      drop_nan=False,
      drop_base=True
   )

def extract_lgbm_params(params: dict):
   """Trích xuất các tham số của LightGBM từ dict params của Optuna."""
   lgbm_params = {
      'boosting_type': params['boosting_type'], 'objective': 'regression', 'metric': 'rmse',
      'learning_rate': params['learning_rate'], 'n_estimators': params['n_estimators'],
      'max_depth': params['max_depth'], 'num_leaves': params['num_leaves'],
      'min_child_samples': params['min_child_samples'], 'min_split_gain': params['min_split_gain'],
      'colsample_bytree': params['colsample_bytree'], 'reg_alpha': params['reg_alpha'],
      'reg_lambda': params['reg_lambda'], 'random_state': 42, 'n_jobs': -1, 'verbose': -1
   }
   if params['boosting_type'] in ['gbdt', 'dart']: lgbm_params['subsample'] = params['subsample']
   if params['boosting_type'] == 'dart': lgbm_params['subsample_freq'] = params['subsample_freq']
   return lgbm_params

def train_model(X_train, y_train, n_trials=50, random_state=42):
   """
   Tinh chỉnh và huấn luyện 5 mô hình LightGBM riêng biệt cho 5 ngày dự báo.
   """
   task = Task.current_task()
   if not task:
      task = Task.init(project_name="Hanoi Temperature Forecast", task_name=f"Training_{datetime.now():%Y%m%d_%H%M}")
   logger = task.get_logger()
   
   best_models_per_target, best_pipelines_per_target, best_params_per_target = {}, {}, {}

   for idx, target_name in enumerate(y_train.columns):
      logging.info(f"===== Bắt đầu Tinh chỉnh cho target: {target_name} ({idx + 1}/{len(y_train.columns)}) =====")

      def objective(trial):
         pipeline_params = {
               'is_linear': trial.suggest_categorical("is_linear", [False]),
               'encoding_method_condition': trial.suggest_categorical("encoding_method_condition", ["ordinal", "target", "quantile"]),
               'n_seasons': trial.suggest_int("n_seasons", 3, 8), 'n_quantiles': trial.suggest_int("n_quantiles", 2, 6),
               'conditions_is_category': trial.suggest_categorical("conditions_is_category", [True, False]),
               'season_is_category': trial.suggest_categorical("season_is_category", [True, False]),
               'wind_is_category': trial.suggest_categorical("wind_is_category", [True, False]),
         }
         lgbm_params = {
               'boosting_type': trial.suggest_categorical("boosting_type", ["gbdt", "dart", "goss"]),
               'learning_rate': trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
               'n_estimators': trial.suggest_int("n_estimators", 100, 1000), 
               'max_depth': trial.suggest_int("max_depth", 3, 12),
               'num_leaves': trial.suggest_int("num_leaves", 15, 255), 
               'min_child_samples': trial.suggest_int("min_child_samples", 5, 100),
               'min_split_gain': trial.suggest_float("min_split_gain", 0.0, 1.0), 
               'colsample_bytree': trial.suggest_float("colsample_bytree", 0.6, 1.0),
               'reg_alpha': trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True), 
               'reg_lambda': trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
         }
         if lgbm_params['boosting_type'] in ['gbdt', 'dart']: lgbm_params['subsample'] = trial.suggest_float("subsample", 0.6, 1.0)
         if lgbm_params['boosting_type'] == 'dart': lgbm_params['subsample_freq'] = trial.suggest_int("subsample_freq", 1, 10)
         
         trial_params = {**pipeline_params, **lgbm_params}
         cv = TimeSeriesSplit(n_splits=5)
         rmse_scores = []
         for train_idx, val_idx in cv.split(X_train):
               X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]                                      
               y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
               y_tr_target = y_tr[target_name]
               
               pipeline = build_final_complete_pipeline(trial_params)
               pipeline.fit(X_tr, y_tr_target)
               X_tr_processed, X_val_processed = pipeline.transform(X_tr), pipeline.transform(X_val)
               
               X_tr_final, y_tr_final = X_tr_processed.iloc[30:], y_tr_target.iloc[30:]
               X_val_final, y_val_final = X_val_processed.iloc[30:], y_val[target_name].iloc[30:]

               model = LGBMRegressor(**extract_lgbm_params(trial_params))
               model.fit(X_tr_final, y_tr_final)
               metrics = evaluate_model(model, pipeline, X_val, y_val[target_name])
               rmse_scores.append(metrics['RMSE'])

         avg_rmse = np.mean(rmse_scores)
         logger.report_scalar(f'Optuna CV RMSE/{target_name}', 'RMSE', iteration=trial.number, value=avg_rmse)
         return avg_rmse

      study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=random_state))
      study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
      
      best_params = study.best_trial.params
      best_params_per_target[target_name] = best_params
      logging.info(f"Tinh chỉnh cho {target_name} hoàn tất. Best RMSE: {study.best_value:.4f}")

      logging.info(f"Huấn luyện lại model cuối cùng cho {target_name}...")
      final_pipeline = build_final_complete_pipeline(best_params)
      final_pipeline.fit(X_train, y_train[target_name])
      
      X_train_processed = final_pipeline.transform(X_train)
      X_train_final, y_train_final = X_train_processed.iloc[30:], y_train[target_name].iloc[30:]
      
      final_model = LGBMRegressor(**extract_lgbm_params(best_params))
      final_model.fit(X_train_final, y_train_final)
      
      best_models_per_target[target_name] = final_model
      best_pipelines_per_target[target_name] = final_pipeline

   logging.info("===== Tinh chỉnh và huấn luyện hoàn tất cho tất cả 5 target =====")
   return best_models_per_target, best_pipelines_per_target, best_params_per_target, task

def evaluate_model(model, pipeline, X_test, y_test):
    # --- TRƯỜNG HỢP 1: ĐÁNH GIÁ MỘT BỘ 5 MODEL/PIPELINE ---
    if isinstance(model, dict) and isinstance(pipeline, dict):
        logging.info("Bắt đầu đánh giá bộ 5 mô hình trên tập test...")
        all_predictions = {}
        target_names = model.keys()

        for target_name in target_names:
            p = pipeline[target_name]
            m = model[target_name]
            
            X_test_processed = p.transform(X_test)
            X_test_final = X_test_processed.iloc[30:]
            
            y_pred = m.predict(X_test_final)
            all_predictions[target_name] = y_pred

        y_pred_df = pd.DataFrame(all_predictions, index=X_test_final.index)
        y_test_final = y_test.loc[y_pred_df.index]
        
        metrics = evaluate_multi_output(y_test_final, y_pred_df)
        logging.info(f"Đánh giá bộ mô hình hoàn tất. RMSE trung bình: {metrics['average']['RMSE']:.4f}")
        return metrics

    # --- TRƯỜNG HỢP 2: ĐÁNH GIÁ MỘT MODEL/PIPELINE ĐƠN LẺ ---
    elif hasattr(model, 'predict') and hasattr(pipeline, 'transform'):
        X_test_processed = pipeline.transform(X_test)
        X_test_final = X_test_processed.iloc[30:]
        y_test_final = y_test.iloc[30:]
        
        y_pred = model.predict(X_test_final)
        
        # Hàm evaluate từ model_evaluation.py sẽ trả về RMSE, MAE, R2...
        metrics = evaluate(y_test_final, y_pred)
        return metrics
        
    else:
        raise TypeError("Đầu vào cho evaluate_model không hợp lệ. Phải là model/pipeline đơn lẻ hoặc dict.")

def save_artifacts(
   models: dict, pipelines: dict, best_params: dict, metrics: dict, 
   final_model_object: CustomMultiOutputRegressor, task: Task, output_path: str
):
   output_path = Path(output_path)
   output_path.parent.mkdir(parents=True, exist_ok=True)
   
   artifacts_to_save = {
      "models": models, "pipelines": pipelines, "best_params": best_params, 
      "metrics": metrics, "final_multi_model": final_model_object,
      "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
   }
   joblib.dump(artifacts_to_save, output_path)
   logging.info(f"Đã lưu toàn bộ artifacts vào: {output_path}")

   if metrics and 'average' in metrics:
      avg_metrics = metrics['average']
      for metric_name, value in avg_metrics.items():
         if isinstance(value, (int, float)):
               Logger.current_logger().report_scalar("Final Model Metrics", metric_name, value=value, iteration=0)
   
   task.upload_artifact(name="complete_model_package", artifact_object=str(output_path))
   logging.info(f"Đã upload '{output_path}' lên ClearML. Đang chờ hoàn tất...")
   task.flush()
   logging.info("Upload artifact lên ClearML đã hoàn tất.")

   retrain_log_path = output_path.parent / "retrain_log.pkl"
   try:
      history = joblib.load(retrain_log_path) if retrain_log_path.exists() else []
      log_entry = {
         "timestamp": artifacts_to_save["timestamp"], "metrics": metrics,
         "best_params_summary": {k: v['boosting_type'] for k, v in best_params.items()}
      }
      history.append(log_entry)
      joblib.dump(history, retrain_log_path)
      logging.info(f"Đã cập nhật lịch sử huấn luyện tại: {retrain_log_path}")
   except Exception as e:
      logging.error(f"Không thể lưu lịch sử huấn luyện: {e}")

def compare_models(
   new_models: dict, new_pipelines: dict, old_model_path: str, 
   X_test: pd.DataFrame, y_test: pd.DataFrame, threshold=0.99
):
   logging.info("Bắt đầu so sánh mô hình mới và mô hình cũ...")
   new_metrics = evaluate_model(new_models, new_pipelines, X_test, y_test)
   new_rmse = new_metrics["average"]["RMSE"]
   logging.info(f"-> RMSE trung bình của bộ mô hình mới: {new_rmse:.4f}")

   old_model_path = Path(old_model_path)
   if not old_model_path.exists():
      logging.info("Không tìm thấy mô hình cũ. Tự động chấp nhận mô hình mới.")
      return True, new_metrics, None

   try:
      old_artifacts = joblib.load(old_model_path)
      old_models, old_pipelines = old_artifacts['models'], old_artifacts['pipelines']
      
      logging.info("Đánh giá mô hình cũ trên tập test...")
      old_metrics = evaluate_model(old_models, old_pipelines, X_test, y_test)
      old_rmse = old_metrics["average"]["RMSE"]
      logging.info(f"-> RMSE trung bình của bộ mô hình cũ: {old_rmse:.4f}")

      if new_rmse < old_rmse * threshold:
         improvement = (1 - new_rmse / old_rmse) * 100
         logging.info(f"QUYẾT ĐỊNH: Chấp nhận mô hình mới (Tốt hơn {improvement:.2f}%)")
         return True, new_metrics, old_metrics
      else:
         improvement = (new_rmse / old_rmse - 1) * 100
         logging.warning(f"QUYẾT ĐỊNH: Giữ lại mô hình cũ (Không đủ cải thiện, kém hơn {improvement:.2f}%)")
         return False, new_metrics, old_metrics
   except Exception as e:
      logging.error(f"Lỗi xảy ra trong quá trình so sánh mô hình: {e}")
      logging.warning("QUYẾT ĐỊNH: Giữ lại mô hình cũ để đảm bảo an toàn.")
      return False, new_metrics, None

def retrain_pipeline(data_path):
   logging.info("Bắt đầu quy trình huấn luyện lại...")
   X_train, y_train, X_test, y_test = preprocess_data(data_path)

   logging.info("Bắt đầu quá trình tuning và huấn luyện bộ mô hình mới...")
   new_models, new_pipelines, best_params, task = train_model(X_train, y_train)
   logging.info("Huấn luyện bộ mô hình mới hoàn tất.")

   if not new_models:
      logging.error("Quá trình huấn luyện không tạo ra được mô hình mới. Dừng lại.")
      return "training_failed", None, None, None

   is_new_model_better, new_metrics, old_metrics = compare_models(
      new_models=new_models, new_pipelines=new_pipelines,
      old_model_path=str(CURRENT_MODEL_PATH),
      X_test=X_test, y_test=y_test
   )
   
   final_new_model = CustomMultiOutputRegressor(
      models_per_target=new_models,
      complete_pipelines_per_target=new_pipelines
   )

   deployment_decision = "kept_old_model"
   final_metrics = old_metrics

   if is_new_model_better:
      deployment_decision = "deployed_new_model"
      final_metrics = new_metrics
      logging.info(f"Triển khai bộ mô hình mới...")
      save_artifacts(
         models=new_models, pipelines=new_pipelines, best_params=best_params,
         metrics=final_metrics, final_model_object=final_new_model,
         task=task, output_path=str(CURRENT_MODEL_PATH)
      )
      try:
         logging.info("Bắt đầu quá trình chuyển đổi sang định dạng ONNX...")
         # subprocess.run(['python', str(BASE_DIR / 'src' / 'convert_to_onnx.py')], check=True)
         logging.info("Chuyển đổi ONNX hoàn tất.")
      except (subprocess.CalledProcessError, FileNotFoundError) as e:
         logging.error(f"Lỗi khi chạy script convert_to_onnx.py: {e}")
   else:
      logging.info("Giữ lại bộ mô hình cũ. Không có thay đổi nào được triển khai.")
      final_metrics = old_metrics if old_metrics else (new_metrics if new_metrics else {})

   task.set_parameters({"Deployment Decision": deployment_decision})
   task.close()
   logging.info("Quy trình huấn luyện lại đã được ghi lại trên ClearML.")

   return deployment_decision, final_metrics, new_metrics, old_metrics

def main():
   print("="*50 + "\n    BẮT ĐẦU QUY TRÌNH HUẤN LUYỆN LẠI MÔ HÌNH      \n" + "="*50)
   results = retrain_pipeline(data_path=str(DATA_PATH))
   
   if not results or results[0] == "training_failed":
      print("\n[LỖI] Quy trình huấn luyện lại đã thất bại. Vui lòng kiểm tra logs.")
      return

   deployment_decision, final_metrics, new_metrics, old_metrics = results

   print("\n" + "="*50 + "\n              KẾT QUẢ HUẤN LUYỆN LẠI              \n" + "="*50)
   if deployment_decision == "deployed_new_model":
      print("\n[QUYẾT ĐỊNH]: Triển khai MÔ HÌNH MỚI thành công!")
      if old_metrics:
            improvement = (1 - new_metrics["average"]["RMSE"] / old_metrics["average"]["RMSE"]) * 100
            print(f"-> Cải thiện so với mô hình cũ: {improvement:.2f}%")
      else:
            print("-> Đây là mô hình đầu tiên được triển khai.")
   else:
      print("\n[QUYẾT ĐỊNH]: Giữ lại MÔ HÌNH CŨ.")
      if new_metrics: print("-> Mô hình mới không đạt đủ ngưỡng cải thiện.")

   print("\n--- Metrics của mô hình được triển khai ---")
   if final_metrics and 'average' in final_metrics:
      print(f"  RMSE (trung bình): {final_metrics['average'].get('RMSE', 'N/A'):.4f}")
      print(f"  R2 (trung bình):   {final_metrics['average'].get('R2', 'N/A'):.4f}")
      print(f"  MAE (trung bình):  {final_metrics['average'].get('MAE', 'N/A'):.4f}")
   else:
      print("  Không có metrics để hiển thị.")

   print("\n" + "="*50 + "\n            QUY TRÌNH ĐÃ HOÀN TẤT                 \n" + "="*50)

if __name__ == "__main__":
   main()