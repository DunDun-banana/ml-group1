import os
import joblib
import sys
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from src.model_training import retrain_pipeline

# --- Cấu hình logging ---
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')

# --- Đường dẫn sử dụng pathlib ---
BASE_DIR = Path(__file__).parent.parent
LOG_PATH = BASE_DIR / "logs" / "daily_rmse.txt"
RETRAIN_LOG_PATH = BASE_DIR / "logs" / "retrain_log.pkl"
DATA_PATH = BASE_DIR / "data" / "latest_3_year.csv"

# --- Ngưỡng cho monitoring ---
RMSE_THRESHOLD = 2.5       # Ngưỡng cảnh báo drift về hiệu năng
RETRAIN_INTERVAL_DAYS = 90 # Tự động retrain sau 90 ngày


# --- Hàm lấy múi giờ ---
def get_timezone():
    """Lấy múi giờ từ biến môi trường TZ, mặc định là Asia/Ho_Chi_Minh."""
    tz_string = os.getenv("TZ", "Asia/Ho_Chi_Minh")
    try:
        return ZoneInfo(tz_string)
    except Exception:
        return ZoneInfo("Asia/Ho_Chi_Minh")


def check_rmse_drift(log_path=LOG_PATH, threshold=RMSE_THRESHOLD):
    """Kiểm tra xem RMSE trung bình 5 ngày gần nhất có vượt ngưỡng không."""
    log_path = Path(log_path)
    if not log_path.exists():
        logging.warning(f"Không tìm thấy file log metrics tại '{log_path}', bỏ qua kiểm tra drift.")
        return False

    try:
        # 1. Tải dữ liệu và kiểm tra định dạng
        log_data = joblib.load(log_path)
        if not isinstance(log_data, list) or not log_data:
            logging.warning(f"File log '{log_path}' không phải là danh sách hoặc bị rỗng.")
            return False
        
        # 2. Chuyển đổi sang DataFrame
        df = pd.DataFrame(log_data)
        
    except Exception as e:
        logging.error(f"Không thể đọc hoặc xử lý file log tại '{log_path}': {e}")
        return False

    # 3. Kiểm tra sự tồn tại của cột 'rmse'
    if "rmse" not in df.columns:
        logging.warning(f"Log tại '{log_path}' không có cột 'rmse', bỏ qua kiểm tra drift.")
        return False

    # 4. Tính RMSE trung bình 5 ngày gần nhất
    recent_rmse_series = df["rmse"].dropna().tail(5)
    
    if len(recent_rmse_series) < 5:
        logging.info(f"Không có đủ 5 điểm dữ liệu RMSE hợp lệ gần nhất (chỉ có {len(recent_rmse_series)}). Bỏ qua kiểm tra drift.")
        return False
        
    recent_rmse = recent_rmse_series.mean()
    logging.info(f"RMSE trung bình 5 ngày gần nhất: {recent_rmse:.4f}")
    
    # 5. So sánh với ngưỡng
    if recent_rmse > threshold:
        logging.warning(f"CẢNH BÁO: RMSE ({recent_rmse:.4f}) đã vượt ngưỡng {threshold}! Cần huấn luyện lại.")
        return True
    
    return False


def check_retrain_interval(log_path=RETRAIN_LOG_PATH, interval_days=RETRAIN_INTERVAL_DAYS):
    """Kiểm tra xem đã quá 90 ngày kể từ lần retrain gần nhất chưa"""
    tz = get_timezone()
    log_path = Path(log_path)
    
    if not log_path.exists():
        logging.info("Chưa có lịch sử huấn luyện lại. Bỏ qua kiểm tra khoảng thời gian.")
        return False

    try:
        records = joblib.load(log_path)
        if not records:
            logging.info("Lịch sử huấn luyện lại rỗng. Cần huấn luyện lại lần đầu.")
            return True
    except Exception as e:
        logging.error(f"Không thể đọc file lịch sử huấn luyện lại tại {log_path}: {e}")
        return False

    last_date_str = records[-1]["timestamp"]
    last_date = datetime.strptime(last_date_str, "%Y-%m-%d %H:%M:%S")
    
    # Chuyển đổi last_date sang múi giờ hiện tại nếu cần
    if last_date.tzinfo is None:
        last_date = last_date.replace(tzinfo=tz)
    
    current_time = datetime.now(tz)
    days_since = (current_time - last_date).days

    logging.info(f"Lần huấn luyện lại gần nhất: {last_date_str} ({days_since} ngày trước).")
    if days_since > interval_days:
        logging.info(f"Đã quá {interval_days} ngày. Cần huấn luyện lại theo lịch.")
        return True
    return False


def monitor_and_retrain():
    """Giám sát và tự động retrain khi cần"""
    logging.info("="*20 + " BẮT ĐẦU QUY TRÌNH GIÁM SÁT " + "="*20)
    need_retrain = False

    # 1. Kiểm tra drift
    if check_rmse_drift():
        need_retrain = True

    # 2. Kiểm tra thời gian từ lần retrain gần nhất
    if check_retrain_interval():
        need_retrain = True

    # 3. Nếu cần retrain thì gọi pipeline
    if need_retrain:
        logging.info(">>> Bắt đầu quy trình huấn luyện lại mô hình...")
        retrain_pipeline(str(DATA_PATH)) # gọi hàm retrain trong model_training
        logging.info(">>> Quy trình huấn luyện lại đã hoàn tất.")
    else:
        logging.info("Hệ thống ổn định, không cần huấn luyện lại.")
    
    logging.info("="*20 + " KẾT THÚC QUY TRÌNH GIÁM SÁT " + "="*20)


if __name__ == "__main__":
    monitor_and_retrain()

# hàm check interval chạy ổn
# hàm check rmse, chạy thử với file daily_rmse có ổn ko
# xem có cần tự động 5 ngày monitorin, ... / hoặc là sau khi gọi bên forcasting => gọi hàm  monitor_and_retrain trong monitoring