import logging
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
from sklearn.metrics import mean_squared_error
import schedule  # dùng để chạy theo lịch trình

from lightgbm import LGBMRegressor
from src.data_preprocessing import load_data, basic_preprocessing
from src.pipeline import build_preprocessing_pipeline
from src.model_evaluation import evaluate_multi_output, evaluate
from src.model_training import CustomMultiOutputRegressor
from src.monitoring import monitor_and_retrain

# --- PATHs ---
DATA_PATH = r"data\latest_3_year.csv"
API_KEY = r"642BDT8N8D49CTFJCX8ZWU6RT"
MODEL_PATH = r"models\Current_model.pkl"
LOG_PATH = r"logs/daily_rmse.txt"
PIPE_1 = r"pipelines/preprocessing_pipeline.pkl"
# PIPE_2 = r"pipelines/featureSelection_pipeline.pkl"


# --- Lấy dữ liệu mới nhất ---
def fetch_latest_weather_data(location="Hanoi", days=35): 
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
def update_three_year_data(new_data: pd.DataFrame, data_path=DATA_PATH):
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
    # print(f"Dữ liệu đã được cập nhật, còn lại {len(combined)} dòng (~3 năm).")
    # print(f"Khoảng thời gian: {combined['datetime'].min().date()} → {combined['datetime'].max().date()}")
    logging.info(f"Dữ liệu đã được cập nhật, còn lại {len(combined)} dòng (~3 năm).")
    logging.info(f"Khoảng thời gian: {combined['datetime'].min().date()} → {combined['datetime'].max().date()}")


# --- Chuẩn bị dữ liệu ---
def prepare_data(df):
    logging.info("Bắt đầu tiền xử lý cơ bản cho dữ liệu dự báo...")
    df_processed = basic_preprocessing(df=df)
    logging.info(f"Tiền xử lý cơ bản hoàn tất. Dữ liệu có shape: {df_processed.shape}")
    
    # Lưu lại để tiện debug
    df_processed.to_csv(f"data/Today_X_input.csv", index=True)
    
    return df_processed

# --- Dự đoán ---
def predict_tomorrow(raw_input_df):
    try:
        logging.info(f"Tải bộ mô hình từ: {MODEL_PATH}")
        # Tải dictionary chứa tất cả các artifacts
        artifacts = joblib.load(MODEL_PATH)
        
        # Lấy ra đối tượng wrapper CustomMultiOutputRegressor
        multi_output_model = artifacts['final_multi_model']
        
        logging.info("Bắt đầu thực hiện dự báo 5 ngày...")
        # Gọi phương thức .predict(), nó sẽ tự động xử lý toàn bộ pipeline
        y_pred_df = multi_output_model.predict(raw_input_df)
        
        # Chúng ta chỉ cần dự báo cho ngày gần nhất, nên lấy dòng cuối cùng
        # Chuyển DataFrame thành numpy array như đầu ra của hàm cũ
        y_pred_latest = y_pred_df.iloc[-1:].to_numpy()
        
        logging.info(f"Dự báo hoàn tất. Kết quả: {y_pred_latest}")
        return y_pred_df

    except FileNotFoundError:
        logging.error(f"Lỗi: Không tìm thấy file mô hình tại '{MODEL_PATH}'.")
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Lỗi không xác định khi dự báo: {e}")
        return pd.DataFrame()


# --- Ghi log RMSE ---
def log_rmse_daily(pred_path, actual_path):
    """
    So sánh dự đoán trong file realtime_predictions với dữ liệu thật trong current3weeks.
    Tính RMSE cho 5 ngày tiếp theo CHỈ KHI đủ dữ liệu cho cả 5 ngày.
    """

    # --- Đọc dữ liệu ---
    pred_df = pd.read_csv(pred_path)
    actual_df = pd.read_csv(actual_path)

    pred_df['date'] = pd.to_datetime(pred_df['date'])
    actual_df['datetime'] = pd.to_datetime(actual_df['datetime'])

    # --- Chuẩn bị thư mục ---
    os.makedirs('logs', exist_ok=True)

    # --- Đọc log cũ an toàn ---
    if os.path.exists(LOG_PATH):
        try:
            all_logs = joblib.load(LOG_PATH)
            if not isinstance(all_logs, list):
                all_logs = []
        except Exception:
            all_logs = []
    else:
        all_logs = []

    # Lấy danh sách các base_date đã log
    logged_dates = {log['base_date'] for log in all_logs}

    # --- Lặp qua từng dòng dự báo ---
    for _, row in pred_df.iterrows():
        base_date = row['date']
        base_date_str = base_date.strftime("%Y-%m-%d")
        
        # Bỏ qua nếu đã log
        if base_date_str in logged_dates:
            continue
            
        forecast_dates = [base_date + timedelta(days=i) for i in range(1, 6)]
        forecast_values = [row[f'pred_day_{i}'] for i in range(1, 6)]

        # Lấy dữ liệu thực tế
        actual_values = []
        for d in forecast_dates:
            val = actual_df.loc[actual_df['datetime'].dt.date == d.date(), 'temp']
            actual_values.append(val.values[0] if not val.empty else np.nan)

        # Chỉ tính RMSE khi có đủ dữ liệu cho CẢ 5 NGÀY
        if np.any(np.isnan(actual_values)):
            # Nếu thiếu dữ liệu, bỏ qua không log (chờ ngày sau)
            logging.info(
                f"Base date: {base_date_str} → End date: {forecast_dates[-1].strftime('%Y-%m-%d')} "
                f"| Chờ dữ liệu thực tế (còn thiếu {np.sum(np.isnan(actual_values))} ngày)"
            )
            continue
        
        # Đã đủ dữ liệu -> tính RMSE
        y_true = np.array([actual_values])
        y_pred = np.array([forecast_values])
        metrics = evaluate_multi_output(y_true, y_pred)
        rmse_value = metrics["average"]["RMSE"]

        log_entry = {
            "base_date": base_date_str,
            "end_date": forecast_dates[-1].strftime("%Y-%m-%d"),
            "rmse": rmse_value,
            "logged_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        all_logs.append(log_entry)
        logged_dates.add(base_date_str)

        logging.info(
            f"Base date: {log_entry['base_date']} → End date: {log_entry['end_date']} "
            f"| RMSE = {rmse_value:.4f} | Logged at: {log_entry['logged_at']}"
        )

    # --- Lưu log an toàn ---
    if all_logs:
        joblib.dump(all_logs, LOG_PATH)
        logging.info(f"Đã cập nhật {len(all_logs)} log entries.")
    else:
        logging.info("Không có log mới để lưu.")


# ---  Task tự động hàng ngày ---
def daily_update():
    logging.info(f"Bắt đầu cập nhật dữ liệu lúc {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. Lấy dữ liệu mới
    new_data = fetch_latest_weather_data()

    if new_data.empty:
        logging.error("Không có dữ liệu mới để cập nhật. Dừng quy trình.")
        return

    # 2. Cập nhật file 3 năm
    update_three_year_data(new_data)

    # 3. Chuẩn bị dữ liệu & dự báo
    raw_input_df = prepare_data(new_data)
    y_pred_df = predict_tomorrow(raw_input_df)

    # 4. Ghi log kết quả dự đoán hôm nay
    save_prediction_log(y_pred_df)
    
    # 5. So sánh với giá trị thực tế (nếu có)
    log_rmse_daily(r'data\realtime_predictions.csv', r'data\Current_Raw_3weeks.csv')

    # 6. KÍCH HOẠT QUY TRÌNH GIÁM SÁT VÀ HUẤN LUYỆN LẠI (NẾU CẦN)
    logging.info("Bắt đầu kiểm tra giám sát mô hình...")
    monitor_and_retrain()

    logging.info("Cập nhật & dự báo hoàn tất.\n")

def save_prediction_log(y_pred_df, output_dir="data"):
    """
    Lưu/cập nhật log dự báo.
    Nếu một ngày đã có trong log, nó sẽ được ghi đè bởi dự báo mới nhất.
    """
    if y_pred_df.empty:
        logging.warning("Không có dữ liệu dự báo mới để lưu.")
        return

    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "realtime_predictions.csv")

    # 1. Chuẩn bị DataFrame dự báo mới
    new_preds = y_pred_df.copy()
    new_preds.index.name = 'date'
    new_preds = new_preds.reset_index()
    
    # Đổi tên cột cho nhất quán với file log
    new_preds.columns = ['date'] + [f'pred_day_{i+1}' for i in range(y_pred_df.shape[1])]
    
    # Thêm cột weekday và định dạng lại ngày
    new_preds['date'] = pd.to_datetime(new_preds['date'])
    new_preds.insert(0, 'weekday', new_preds['date'].dt.strftime('%A'))
    new_preds['date'] = new_preds['date'].dt.strftime('%Y-%m-%d')

    # 2. Đọc và kết hợp với log cũ
    if os.path.exists(file_path):
        try:
            old_df = pd.read_csv(file_path)
            # Đảm bảo cột 'date' của file cũ là string để join
            old_df['date'] = old_df['date'].astype(str)
            
            # Gộp hai DataFrame, giữ lại các dòng từ new_preds nếu có ngày trùng lặp
            combined_df = pd.concat([old_df, new_preds]).drop_duplicates(subset=['date'], keep='last')
        except (pd.errors.EmptyDataError, FileNotFoundError):
            # Nếu file cũ rỗng hoặc lỗi, chỉ dùng dự báo mới
            combined_df = new_preds
    else:
        combined_df = new_preds

    # 3. Sắp xếp và lưu lại
    combined_df = combined_df.sort_values(by="date").reset_index(drop=True)
    combined_df.to_csv(file_path, index=False)
    
    logging.info(f"Đã cập nhật/lưu log dự báo vào {file_path}. Tổng cộng có {len(combined_df)} dòng.")


# # --- Lên lịch chạy lúc 00:00 mỗi ngày ---
# schedule.every().day.at("12:00").do(daily_update)

if __name__ == "__main__":
    # Chạy thử ngay lập tức khi khởi động, bao giờ ổn thì xoá daily_update đi để nó tự chạy theo lịch thôi
    daily_update()

    # # print("Service đang chạy... sẽ tự động cập nhật mỗi ngày lúc 12:00.")
    # while True:
    #     schedule.run_pending()
    #     time.sleep(60)


