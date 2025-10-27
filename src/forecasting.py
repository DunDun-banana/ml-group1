import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import joblib
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from io import StringIO
from sklearn.base import BaseEstimator, TransformerMixin
import schedule  # dùng để chạy theo lịch trình

from lightgbm import LGBMRegressor
from src.data_preprocessing import load_data, basic_preprocessing
from src.pipeline import build_preprocessing_pipeline, build_GB_featture_engineering_pipeline
from src.feature_engineering import feature_engineering
from src.model_evaluation import evaluate_multi_output, evaluate

# --- PATHs ---
DATA_PATH = r"data\latest_3_year.csv"
API_KEY = r"642BDT8N8D49CTFJCX8ZWU6RT"
MODEL_PATH = r"models\Current_model.pkl"
LOG_PATH = r"logs/daily_rmse.pkl"
PIPE_1 = r"pipelines/preprocessing_pipeline.pkl"
PIPE_2 = r"pipelines/featureSelection_pipeline.pkl"


# --- Lấy dữ liệu mới nhất ---
def fetch_latest_weather_data(location="Hanoi", days=21): # oke
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)

    base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
    url = f"{base_url}/{location}/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"

    params = {
        "unitGroup": "metric",
        "include": "days",
        "key": API_KEY,
        "contentType": "csv"
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        print("Lấy dữ liệu thời tiết thành công.")
        df = pd.read_csv(StringIO(response.text))
        df.to_csv(f"data/Current_Raw_3weeks.csv", index=False)
        return df
    else:
        print(f"Lỗi khi gọi API ({response.status_code}): {response.text}")
        return pd.DataFrame()


# --- Cập nhật dữ liệu 3 năm ---
def update_three_year_data(new_data: pd.DataFrame, data_path=DATA_PATH): #oke 
    """
    Cập nhật file current_3_year.csv:
      - Đọc file cũ nếu có
      - Gộp thêm dữ liệu mới
      - Loại bỏ trùng lặp theo cột datetime
      - Giữ lại đúng 3 năm gần nhất (tính từ hôm nay)
    """
    # Đảm bảo có cột datetime
    if "datetime" not in new_data.columns:
        raise ValueError("new_data không có cột 'datetime'")

    # Ép kiểu datetime (cho cả new_data)
    new_data["datetime"] = pd.to_datetime(new_data["datetime"], errors="coerce")

    # Đọc dữ liệu cũ nếu có
    if os.path.exists(data_path):
        old_data = pd.read_csv(data_path)
        if "datetime" not in old_data.columns:
            raise ValueError("old_data không có cột 'datetime'")
        old_data["datetime"] = pd.to_datetime(old_data["datetime"], errors="coerce")

        # Gộp dữ liệu
        combined = pd.concat([old_data, new_data], ignore_index=True)
    else:
        combined = new_data.copy()

    # Loại bỏ dòng có datetime bị lỗi hoặc trùng
    combined.dropna(subset=["datetime"], inplace=True)
    combined.drop_duplicates(subset=["datetime"], keep="last", inplace=True)

    # Giới hạn dữ liệu trong 3 năm gần nhất
    three_years_ago = datetime.today() - timedelta(days=3 * 365)
    combined = combined[combined["datetime"] >= three_years_ago]

    # sắp xếp và lưu lại
    combined.sort_values("datetime", inplace=True)
    combined.to_csv(data_path, index=False)

    # kiểm tra
    print(f"Dữ liệu đã được cập nhật, còn lại {len(combined)} dòng (~3 năm).")
    print(f"Khoảng thời gian: {combined['datetime'].min().date()} → {combined['datetime'].max().date()}")


# --- Chuẩn bị dữ liệu ---
def prepare_data(df):

   # 1. basic preprocessing for all data set
   df = basic_preprocessing(df=df)
   #print(1, df.columns)

   # 3. Pipeline 1: preprocessing
   pipeline1 = joblib.load(PIPE_1)
   df_processed = pipeline1.transform(df)
   #print(2, df_processed.columns)

   # 4. Feature engineering cho input
   df_feat, target_col = feature_engineering(df_processed)
   #print(3, df_feat.columns)

   # 5. Lây input X
   X_df = df_feat.drop(columns= target_col)
   #print(4, X_df.columns)
   # y_df = train_feat[target_col]

   # 6. Pipeline 2: GB selection
   pipeline2 = joblib.load(PIPE_2)
   proccessed_X = pipeline2.transform(X_df)
   #print(5, proccessed_X.columns)

   proccessed_X.to_csv(f"data/Today_X_input.csv", index=False)
   return proccessed_X

# --- Dự đoán ---
def predict_tomorrow(processed_X):
    model = joblib.load(MODEL_PATH)
    y_pred_5_days = model.predict(processed_X)
    return y_pred_5_days


# --- Ghi log RMSE ---
# sửa lại dùng hàm của mình + vde cụ thể tí t nêu sau
def log_rmse_daily(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    log_entry = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "rmse": rmse
    }

    os.makedirs('logs', exist_ok=True)
    if not os.path.exists(LOG_PATH):
        joblib.dump([log_entry], LOG_PATH)
        print(f"Log file created. First entry: RMSE = {rmse:.4f}")
    else:
        logs = joblib.load(LOG_PATH)
        logs.append(log_entry)
        joblib.dump(logs, LOG_PATH)
        print(f"Logged RMSE = {rmse:.4f}")
    return rmse


# ---  Task tự động hàng ngày ---
def daily_update():
    print(f"Bắt đầu cập nhật dữ liệu lúc {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Lấy dữ liệu mới
    new_data = fetch_latest_weather_data()

    if new_data.empty:
        print(" Không có dữ liệu mới để cập nhật.")
        return

    # 2. Cập nhật file 3 năm
    update_three_year_data(new_data)

    # 3. Chuẩn bị dữ liệu & dự báo
    processed = prepare_data(new_data)
    y_pred = predict_tomorrow(processed)

    # 4. Ghi log kết quả dự đoán hôm nay
    save_prediction_log(y_pred)

    print(f" Dự báo hoàn tất, {len(y_pred)} giá trị được dự đoán.")
    print("Cập nhật & dự báo hoàn tất.\n")


import os
import pandas as pd
from datetime import datetime

def save_prediction_log(y_pred, output_dir="data"):
    """Lưu dự đoán từng dòng, mỗi dòng là 1 bộ giá trị dự đoán."""
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "realtime_predictions.csv")

    today = datetime.now()
    weekday = today.strftime("%A")
    date_str = today.strftime("%Y-%m-%d")

    # Đảm bảo y_pred là list của list
    y_pred = y_pred.tolist() if hasattr(y_pred, "tolist") else y_pred

    # Tạo DataFrame mỗi dòng là 1 list con trong y_pred
    df_pred = pd.DataFrame(y_pred, columns=[f"pred_day_{i+1}" for i in range(len(y_pred[0]))])
    df_pred.insert(0, "date", date_str)
    df_pred.insert(0, "weekday", weekday)

    # Nếu file đã tồn tại và không rỗng -> đọc
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        old_df = pd.read_csv(file_path)
        # Nếu chưa có ngày này thì nối thêm
        if date_str not in old_df["date"].values:
            df_pred = pd.concat([old_df, df_pred], ignore_index=True)
        else:
            print(f"Ngày {date_str} đã tồn tại, không ghi đè.")
            return

    df_pred.to_csv(file_path, index=False)
    print(f"Lưu dự đoán dạng phẳng vào {file_path}")


# --- Lên lịch chạy lúc 00:00 mỗi ngày ---
schedule.every().day.at("12:00").do(daily_update)

if __name__ == "__main__":
    # Chạy thử ngay lập tức khi khởi động, bao giờ ổn thì xoá daily_update đi để nó tự chạy theo lịch thôi
    daily_update()

    print("Service đang chạy... sẽ tự động cập nhật mỗi ngày lúc 00:00.")
    while True:
        schedule.run_pending()
        time.sleep(60)


