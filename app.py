import pandas as pd
import joblib
import os
from pathlib import Path
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import altair as alt
import base64
import requests
import time

from dotenv import load_dotenv
load_dotenv()

try:
    from src.forecasting import daily_update
except ImportError:
    st.error("Lỗi import: Không tìm thấy các hàm từ thư mục 'src'. Vui lòng kiểm tra lại cấu trúc thư mục của bạn.")
    def daily_update():
        st.warning("Hàm daily_update() không được tìm thấy. Chức năng cập nhật sẽ không hoạt động.")
        return None

# Thêm import mới với try-except
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
except ImportError:
    st.warning("Thư viện 'statsmodels' chưa được cài đặt. Chức năng phân rã chuỗi thời gian sẽ không hoạt động. Vui lòng chạy: pip install statsmodels")
    seasonal_decompose = None

# --- CÁC ĐƯỜNG DẪN TỚI FILE ---
BASE_DIR = Path(__file__).parent
PATH_PREDICTIONS = BASE_DIR / 'data' / 'realtime_predictions.csv'
PATH_RAW_3WEEKS = BASE_DIR / 'data' / 'Current_Raw_3weeks.csv'
PATH_3_YEAR_DATA = BASE_DIR / 'data' / 'latest_3_year.csv'
PATH_RMSE_LOG = BASE_DIR / 'logs' / 'daily_rmse.txt'
PATH_RETRAIN_LOG = BASE_DIR / 'logs' / 'retrain_log.pkl'
PATH_WEATHER_ICON = BASE_DIR / 'assets' / 'sun.png'


# --- HÀM HỖ TRỢ VỚI CACHING ---
@st.cache_data(ttl=3600)
def load_csv(path):
    path = Path(path)
    if path.exists():
        return pd.read_csv(path)
    return None

@st.cache_data(ttl=3600)
def load_joblib(path):
    path = Path(path)
    if path.exists():
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None

def get_img_as_base64(file):
    file = Path(file)
    with open(file, "rb") as f: 
        data = f.read()
    return base64.b64encode(data).decode()

def load_keys_from_env():
    """Tải danh sách Visual Crossing API keys từ file .env."""
    load_dotenv()
    keys_string = os.getenv("VISUAL_CROSSING_API_KEYS")
    if keys_string:
        return [key.strip() for key in keys_string.split(',')]
    else:
        # Hiển thị lỗi một lần duy nhất khi ứng dụng khởi động nếu không tìm thấy key
        st.error("Lỗi cấu hình: Biến 'VISUAL_CROSSING_API_KEYS' không được tìm thấy trong file .env.")
        return ["642BDT8N8D49CTFJCX8ZWU6RT", "PEKQEGZNARR9BQCCZ7V6XERA4"]  # Thêm một key mặc định để tránh lỗi

def get_timezone():
    """Lấy múi giờ từ biến môi trường TZ, mặc định là Asia/Ho_Chi_Minh."""
    tz_string = os.getenv("TZ", "Asia/Ho_Chi_Minh")
    try:
        return ZoneInfo(tz_string)
    except Exception:
        return ZoneInfo("Asia/Ho_Chi_Minh")

@st.cache_data(ttl=900) # Cache kết quả trong 15 phút
def fetch_realtime_weather(location="Hanoi", api_keys=None):
    """
    Gọi API Visual Crossing để lấy dữ liệu thời tiết hiện tại.
    Tự động xoay vòng qua danh sách API keys nếu gặp lỗi.
    """
    if not api_keys:
        return None # Không có key nào để thử

    base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
    url = f"{base_url}/{location}/today"

    for api_key in api_keys:
        params = {
            "unitGroup": "metric",
            "include": "current",
            "key": api_key,
            "contentType": "json"
        }
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            current_data = data.get("currentConditions")
            # print(current_data)  # Debug log
            
            if not current_data:
                continue # Dữ liệu không hợp lệ, thử key tiếp theo

            # Trích xuất dữ liệu và trả về khi thành công
            return {
                "temperature": current_data.get("temp"),
                "feels_like": current_data.get("feelslike"),
                "chance_of_rain": current_data.get("precipprob"),
                "dew": current_data.get("dew"),
                "wind_speed": current_data.get("windspeed"),
                "uv_index": current_data.get("uvindex"),
                "humidity": current_data.get("humidity"),
                "conditions": current_data.get("conditions"),
                "visibility": current_data.get("visibility"),
                "sunrise": current_data.get("sunrise"),
                "sunset": current_data.get("sunset")
            }

        except requests.exceptions.HTTPError as http_err:
            if http_err.response.status_code in [401, 429]:
                # Lỗi sai key hoặc hết hạn ngạch -> thử key tiếp theo
                continue
            else:
                st.error(f"Lỗi HTTP nghiêm trọng: {http_err}")
                return None # Dừng lại nếu là lỗi server
        except Exception:
            # Lỗi mạng hoặc lỗi không xác định khác -> thử key tiếp theo
            continue

    st.error("Tất cả các API key đều thất bại. Vui lòng kiểm tra lại.")
    return None

# --- AUTO-UPDATE KHI QUA NGÀY MỚI ---
# Thêm NGAY SAU các PATH definitions và TRƯỚC st.set_page_config()

if 'last_update_date' not in st.session_state:
    st.session_state.last_update_date = None

def should_run_daily_update():
    """Kiểm tra xem có cần chạy cập nhật hàng ngày không"""
    tz = get_timezone()
    today = datetime.now(tz).date()
    
    # SỬA LỖI: Chuyển đổi last_update_date sang date nếu cần
    last_update = st.session_state.last_update_date
    if last_update is not None:
        # Chuyển Timestamp hoặc datetime thành date
        if isinstance(last_update, pd.Timestamp):
            last_update = last_update.date()
        elif isinstance(last_update, datetime):
            last_update = last_update.date()
    
    # Kiểm tra nếu chưa từng update hoặc đã qua ngày mới
    if last_update is None or last_update < today:
        return True
    
    # Kiểm tra thêm: Nếu file predictions không tồn tại hoặc rỗng
    predictions_df = load_csv(PATH_PREDICTIONS)
    if predictions_df is None or predictions_df.empty:
        return True
    
    # Kiểm tra xem dự báo mới nhất có phải của hôm nay không
    try:
        latest_forecast_date = pd.to_datetime(predictions_df['date'].iloc[-1]).date()
        if latest_forecast_date < today:
            return True
    except:
        return True
    
    return False

# Chạy auto-update nếu cần
if should_run_daily_update():
    try:
        with st.spinner("🔄 Đang cập nhật dự báo cho ngày mới..."):
            daily_update()
            tz = get_timezone()
            st.session_state.last_update_date = datetime.now(tz).date()
            st.cache_data.clear()
            
            # Hiển thị thông báo thành công
            st.success("✅ Dữ liệu đã được cập nhật cho ngày mới!")
            time.sleep(1.5)  # Hiển thị thông báo 1.5 giây
            st.rerun()
    except Exception as e:
        # st.error(f"⚠️ Lỗi khi cập nhật tự động: {e}")
        # Vẫn đánh dấu là đã cập nhật để tránh retry liên tục
        tz = get_timezone()
        st.session_state.last_update_date = datetime.now(tz).date()

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Hanoi Temperature Forecast",
    page_icon="☀️",
    layout="wide"
)

# --- CSS TÙY CHỈNH ---
st.markdown("""
<link rel='stylesheet' href='https://cdn-uicons.flaticon.com/2.6.0/uicons-thin-straight/css/uicons-thin-straight.css'>
<link rel='stylesheet' href='https://cdn-uicons.flaticon.com/2.6.0/uicons-regular-rounded/css/uicons-regular-rounded.css'>
<style>
    /* Main Container */
    [data-testid="stMainBlockContainer"] {
        padding-top: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: transparent;
        border-bottom: 2px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 0;
    }

    .stTabs [data-baseweb="tab"] {
        height: 3.5rem;
        background-color: transparent;
        border: none;
        color: rgba(255, 255, 255, 0.6);
        padding: 0 0.9rem;
        font-weight: 500;
    }

    /* Selector mới, cụ thể hơn để nhắm vào text */
    .stTabs [data-baseweb="tab"] div {
        font-size: 1.1rem !important;
    }

    .stTabs [data-baseweb="tab"]:hover {
        color: rgba(255, 255, 255, 0.9);
    }

    .stTabs [aria-selected="true"] {
        color: #FFFFFF !important;
        font-weight: 600;
        border-bottom: 3px solid #007BFF;
    }

    .stTabs [data-baseweb="tab-panel"] {
        padding-top: 2rem;
    }
          
    /* Realtime weather block */
    .main-info-block {
        background: #1F242D;
        padding-left: 1.8rem;
        padding-top: 1.5rem;
        padding-right: 1.2rem;
        padding-bottom: 0.5rem;
        border-radius: 24px;
        margin-bottom: 1rem;
    }
    
    .city-name {
        font-size: 0.9rem !important; 
        color: rgba(255, 255, 255, 0.7) !important;
        margin: 0 0 0.5rem 0 !important;
        padding: 0 !important;
    }

    .date-time {
        font-size: 0.9rem !important;
        color: rgba(255, 255, 255, 0.7) !important;
        margin: 0 0 0.5rem 0 !important;
        padding: 0 !important;
    }
            
    .big-temp {
        font-size: 4rem !important;
        font-weight: 360 !important;
        color: #FFFFFF !important;
        margin: 0 !important;
        padding: 0 !important;
        line-height: 1 !important;
    }

    /* CSS cho weather icon và condition */
    .weather-icon-wrapper {
        text-align: left;
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        justify-content: center;
    }

    .weather-icon-wrapper img {
        width: 90px;
        height: 90px;
        max-width: 100%;
    }

    /* CSS cho weather details block - Xanh nước biển vừa */
    .weather-details-block {
        background: linear-gradient(135deg, #1F242D 20%, #0D3B4F 80%);
        padding: 1.2rem;
        border-radius: 24px;
        margin-bottom: 1.5rem;
    }
            
    .detail-title {
        font-size: 1.3rem !important;
        font-weight: 400 !important;
        color: rgba(255, 255, 255, 0.8) !important;
        margin: 0 0 1rem 0 !important;
        padding: 0 !important;
    }

    .detail-grid {
        display: grid;
        grid-template-columns: 1fr 1fr 1fr;
        gap: 0.8rem;
    }

    .detail-item {
        background: rgba(255, 255, 255, 0.05);
        padding: 0.8rem 0.6rem;
        border-radius: 12px;
        text-align: center;
        display: flex;
        align-items: flex-start;
    }

    .detail-label {
        font-size: 0.8rem !important;
        color: rgba(255, 255, 255, 0.6) !important;
        margin: 0 0 0.3rem 0 !important;
        padding: 0 !important;
    }

    .detail-value {
        font-size: 2.3rem !important;
        font-weight: 600 !important;
        color: #FFFFFF !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* CSS cho weather condition text */
    .weather-condition {
        font-size: 1.36rem !important;
        color: rgba(255, 255, 255, 0.8) !important;
        margin-top: 0.4rem !important;
        font-weight: 500 !important;
    }

    /* Responsive adjustments */
    @media (max-width: 1200px) {
        .day-of-week {
            font-size: 1.5rem !important;
        }
        .big-temp {
            font-size: 3rem !important;
        }
        .weather-icon-wrapper img {
            width: 80px;
            height: 80px;
        }
    }

    /* CSS cho forecast block - Xanh nước biển nhạt */
    .forecast-block {
        background: linear-gradient(170deg, #1F242D 20%, #103845 80%);
        padding: 1rem;
        border-radius: 24px;
        margin-bottom: 1rem;
    }

    .forecast-title {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        color: #FFFFFF !important;
        margin: 0 0 1rem 0 !important;
        padding: 0 !important;
    }

    .forecast-cards {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 0.8rem;
        margin-bottom: 1.5rem;
    }

    .forecast-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem 0.5rem;
        border-radius: 16px;
        text-align: center;
        transition: transform 0.2s ease, background 0.2s ease;
    }

    .forecast-card:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.08);
    }

    .forecast-day {
        font-size: 0.85rem !important;
        color: rgba(255, 255, 255, 0.7) !important;
        margin: 0 0 0.3rem 0 !important;
        font-weight: 500 !important;
    }

    .forecast-date {
        font-size: 0.75rem !important;
        color: rgba(255, 255, 255, 0.5) !important;
        margin: 0 0 0.8rem 0 !important;
    }

    .forecast-temp {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #FFFFFF !important;
        margin: 0 !important;
    }

    .forecast-chart-container {
        background: rgba(255, 255, 255, 0.03);
        padding: 1rem;
        border-radius: 16px;
    }

    @media (max-width: 1200px) {
        .forecast-cards {
            grid-template-columns: repeat(3, 1fr);
        }
    }

    @media (max-width: 768px) {
        .forecast-cards {
            grid-template-columns: repeat(2, 1fr);
        }
    }
</style>
""", unsafe_allow_html=True)

# --- TẠO TABS THAY VÌ SIDEBAR ---
tab1, tab2, tab3 = st.tabs(["☀️ Forecasting", "📊 Historical Data Analysis", "⚙️ Model Performance"])

# =============================================================================
# --- TAB 1: DỰ BÁO TRỰC TIẾP ---
# =============================================================================
with tab1:
    # st.title("☀️ Dự báo Nhiệt độ Hà Nội")
    # st.markdown("Trang này hiển thị kết quả dự báo mới nhất và cho phép bạn chạy lại quy trình.")

    # --- PHẦN MỚI: HIỂN THỊ THỜI TIẾT HIỆN TẠI TỪ API ---
    # st.subheader("Thời tiết hiện tại ở Hà Nội")
    
    realtime_data = fetch_realtime_weather("Hanoi", api_keys=load_keys_from_env())

    if realtime_data:
        # Tạo HTML trực tiếp thay vì dùng st.markdown riêng lẻ
        col1, col2 = st.columns([0.8, 2])
        
        with col1:
            # Lấy thời gian hiện tại theo múi giờ
            tz = get_timezone()
            now = datetime.now(tz)
            
            # Chọn icon phù hợp
            if realtime_data.get("chance_of_rain", 0) > 50:
                icon_path = BASE_DIR / 'assets' / 'heavy-rain.png'
            elif realtime_data.get("wind_speed", 0) > 20:
                icon_path = BASE_DIR / 'assets' / 'wind.png'
            elif now.hour >= 18 or now.hour < 6:
                icon_path = BASE_DIR / 'assets' / 'moon.png'
            elif realtime_data.get("temperature", 0) < 30:
                icon_path = BASE_DIR / 'assets' / 'cloudy.png'
            else:
                icon_path = PATH_WEATHER_ICON
            
            # Lấy thông tin ngày tháng
            day_of_week = now.strftime("%A")  # Thứ trong tuần
            date_time = now.strftime("%d %B, %Y")  # Ngày tháng năm 
            
            # Lấy mô tả thời tiết
            weather_condition = realtime_data.get("conditions", "Unknown")
            
            # Tạo HTML block với bố cục mới: icon → temperature → condition → location → datetime
            real_time_main_html = f"""
            <div class="main-info-block">
                <div style="text-align: left;">
                    <div class="weather-icon-wrapper" style="margin-bottom: 1rem;">
                        <img src="data:image/png;base64,{get_img_as_base64(icon_path)}" alt="Weather icon">
                    </div>
                    <p class="big-temp" style="margin-bottom: 0.5rem;">{int(realtime_data.get("temperature", 0))}°C</p>
                    <p class="weather-condition">{weather_condition}</p>
                    <hr style="border: none; border-top: 1px solid rgba(255, 255, 255, 0.2); margin: 0.7rem 0;">
                    <p class="city-name" style="margin-bottom: 0.5rem;">⚲ Ha Noi</p>
                    <p class="date-time">🗒 {day_of_week}, {date_time}</p>
                </div>
            </div>
            """
            st.markdown(real_time_main_html, unsafe_allow_html=True)
        
        with col2:
            # Format sunrise và sunset để chỉ lấy giờ:phút (24h format)
            sunrise_time = realtime_data.get("sunrise", "N/A")
            sunset_time = realtime_data.get("sunset", "N/A")
            
            # Chỉ lấy HH:MM từ format "HH:MM:SS"
            if sunrise_time != "N/A" and len(sunrise_time) > 5:
                sunrise_time = sunrise_time[:5]
            if sunset_time != "N/A" and len(sunset_time) > 5:
                sunset_time = sunset_time[:5]
            
            # Tạo block thông tin chi tiết
            weather_details_html = f"""
            <div class="weather-details-block">
                <p class="detail-title">Today's Highlights</p>
                <div class="detail-grid">
                    <div class="detail-item" style="display: flex; justify-content: space-between; align-items: center; text-align: left; padding: 1rem;">
                        <div>
                            <p class="detail-label" style="text-align: left; margin-bottom: 0.5rem;">Humidity</p>
                            <p class="detail-value">{realtime_data.get("humidity", 0):.0f}<span style="font-size: 1.1rem; font-weight: 400; color: rgba(255, 255, 255, 0.6);">%</span></p>
                        </div>
                        <div style="text-align: right; font-size: 0.75rem; color: rgba(255, 255, 255, 0.6); max-width: 80px; line-height: 1.3;">
                            <p style="margin: 0 0 0.2rem 0;"><i class="fi fi-ts-raindrops"></i></p>
                            <p style="margin: 0;">The dew point is {realtime_data.get("dew", 0):.0f}°C right now</p>
                        </div>
                    </div>
                    <div class="detail-item" style="display: flex; justify-content: space-between; align-items: center; text-align: left; padding: 1rem;">
                        <div>
                            <p class="detail-label" style="text-align: left; margin-bottom: 0.5rem;">UV Index</p>
                            <p class="detail-value">{realtime_data.get("uv_index", 0)}</p>
                        </div>
                        <div style="text-align: right; font-size: 0.75rem; color: rgba(255, 255, 255, 0.6); max-width: 80px; line-height: 1.3;">
                            <p style="margin: 0 0 0.2rem 0;"><i class="fi fi-rr-brightness"></i></p>
                            <p style="margin: 0;">Moderate exposure level</p>
                        </div>
                    </div>
                    <div class="detail-item" style="display: flex; justify-content: space-between; align-items: center; text-align: left; padding: 1rem;">
                        <div>
                            <p class="detail-label" style="text-align: left; margin-bottom: 0.5rem;">Wind Speed</p>
                            <p class="detail-value">{realtime_data.get("wind_speed", 0):.1f}<span style="font-size: 1.1rem; font-weight: 400; color: rgba(255, 255, 255, 0.6);"> km/h</span></p>
                        </div>
                        <div style="text-align: right; font-size: 0.75rem; color: rgba(255, 255, 255, 0.6); max-width: 80px; line-height: 1.3;">
                            <p style="margin: 0 0 0.2rem 0;"><i class="fi fi-rr-wind"></i></p>
                            <p style="margin: 0;">Light breeze conditions</p>
                        </div>
                    </div>
                    <div class="detail-item" style="display: flex; justify-content: space-between; align-items: center; text-align: left; padding: 1rem;">
                        <div>
                            <p class="detail-label" style="text-align: left; margin-bottom: 0.5rem;">Visibility</p>
                            <p class="detail-value">{realtime_data.get("visibility", 0):.1f}<span style="font-size: 1.1rem; font-weight: 400; color: rgba(255, 255, 255, 0.6);"> km</span></p>
                        </div>
                        <div style="text-align: right; font-size: 0.75rem; color: rgba(255, 255, 255, 0.6); max-width: 80px; line-height: 1.3;">
                            <p style="margin: 0 0 0.2rem 0;"><i class="fi fi-rr-eye"></i></p>
                            <p style="margin: 0;">Clear visibility today</p>
                        </div>
                    </div>
                    <div class="detail-item" style="display: flex; justify-content: space-between; align-items: center; text-align: left; padding: 1rem;">
                        <div>
                            <p class="detail-label" style="text-align: left; margin-bottom: 0.5rem;">Feels Like</p>
                            <p class="detail-value">{realtime_data.get("feels_like", 0):.1f}<span style="font-size: 1.1rem; font-weight: 400; color: rgba(255, 255, 255, 0.6);">°C</span></p>
                        </div>
                        <div style="text-align: right; font-size: 0.75rem; color: rgba(255, 255, 255, 0.6); max-width: 80px; line-height: 1.3;">
                            <p style="margin: 0 0 0.2rem 0;"><i class="fi fi-ts-face-thinking"></i></p>
                            <p style="margin: 0;">Similar to actual temp</p>
                        </div>
                    </div>
                    <div class="detail-item" style="display: flex; flex-direction: column; justify-content: flex-start; align-items: flex-start; text-align: left; padding: 1rem;">
                        <p class="detail-label" style="text-align: left; margin-bottom: 0.8rem;">Sunrise & Sunset</p>
                        <div style="display: flex; justify-content: space-between; width: 100%; align-items: center;">
                            <div>
                                <p class="detail-value" style="font-size: 1.5rem !important;">{sunrise_time}</p>
                            </div>
                            <div style="text-align: right;">
                                <p class="detail-value" style="font-size: 1.5rem !important;">{sunset_time}</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            """
            st.markdown(weather_details_html, unsafe_allow_html=True)
        
    else:
        st.warning("Không thể tải dữ liệu thời tiết hiện tại. Vui lòng kiểm tra lại cấu hình.")

    # --- HIỂN THỊ DỰ BÁO CỦA MÔ HÌNH ---
    predictions_df = load_csv(PATH_PREDICTIONS)
    
    if predictions_df is not None and not predictions_df.empty:
        latest_forecast = predictions_df.iloc[-1]
        forecast_date = pd.to_datetime(latest_forecast['date'])
        forecast_values = latest_forecast[[f"pred_day_{i}" for i in range(1, 6)]].values
        forecast_dates = [forecast_date + timedelta(days=i) for i in range(1, 6)]
        
        # Chuyển đổi forecast_values thành float và xử lý NaN
        try:
            forecast_values = forecast_values.astype(float)
            if pd.isna(forecast_values).any():
                st.warning("Một số giá trị dự báo không hợp lệ. Đang thay thế bằng giá trị trung bình.")
                forecast_values = pd.Series(forecast_values).fillna(pd.Series(forecast_values).mean()).values
        except Exception as e:
            st.error(f"Lỗi chuyển đổi dữ liệu: {e}")
            forecast_values = [25.0, 26.0, 27.0, 26.5, 25.5]
        
        # TẠO HTML CHO CÁC CARD DỰ BÁO
        forecast_cards_html = ""
        for date, temp in zip(forecast_dates, forecast_values):
            day_name = date.strftime("%a")
            date_str = date.strftime("%d/%m")
            forecast_cards_html += f'<div class="forecast-card"><p class="forecast-day">{day_name}, {date_str}</p><p class="forecast-temp">{temp:.1f}°C</p></div>'
        
        # TẠO KHỐI HTML CHO TITLE VÀ CARDS
        forecast_html_block = f"""
        <div class="forecast-block">
            <p class="forecast-title">🔮 5-Day Temperature Forecast (Model)</p>
            <div class="forecast-cards">
                {forecast_cards_html}
            </div>
        """
        st.markdown(forecast_html_block, unsafe_allow_html=True)
        
        st.markdown('<p class="forecast-title">📈 Temperature Forecast Trend</p>', unsafe_allow_html=True)
        
        # Tính toán range cho trục Y
        y_min = forecast_values.min() - 2
        y_max = forecast_values.max() + 2

        # Tạo DataFrame cho line chart
        chart_data = pd.DataFrame({
            'Date': [d.strftime('%a %d/%m') for d in forecast_dates],
            'Temperature (°C)': forecast_values
        })
        
        # Tạo biểu đồ Altair để tránh rung lắc
        chart = alt.Chart(chart_data).mark_line(
            point=alt.OverlayMarkDef(color="#007BFF", size=60, filled=True, strokeWidth=3),
            strokeWidth=3,
            color="#007BFF"
        ).encode(
            x=alt.X('Date', sort=None, title=None, axis=alt.Axis(labelColor='white', grid=False, labelAngle=0)),
            y=alt.Y('Temperature (°C)', title='°C', 
                    scale=alt.Scale(domain=[y_min, y_max]),
                    axis=alt.Axis(labelColor='white', titleColor='white', gridColor='rgba(255, 255, 255, 0.1)', labelAngle=0)),
            tooltip=[
                alt.Tooltip('Date', title='Day'),
                alt.Tooltip('Temperature (°C)', title='Temp', format='.1f')
            ]
        ).properties(
            background='transparent',
            height=360
        ).configure_view(
            stroke=None
        )

        st.altair_chart(chart, width='stretch')
        
        # THÊM THỜI GIAN CẬP NHẬT CUỐI
        last_update_time = st.session_state.get('last_update_date', None)
        if last_update_time:
            # SỬA LỖI: Chuyển đổi sang date cho tất cả các trường hợp
            if isinstance(last_update_time, pd.Timestamp):
                last_update_time = last_update_time.date()
            elif isinstance(last_update_time, datetime):
                last_update_time = last_update_time.date()
            # Nếu đã là date thì giữ nguyên
            
            last_update_str = last_update_time.strftime("%d %B, %Y")
            
            # SỬA LỖI: Đảm bảo date.today() trả về datetime.date với múi giờ đúng
            tz = get_timezone()
            today = datetime.now(tz).date()
            time_diff = (today - last_update_time).days
            
            if time_diff == 0:
                time_ago = "today"
            elif time_diff == 1:
                time_ago = "yesterday"
            else:
                time_ago = f"{time_diff} days ago"
            
            st.markdown(f"""
            <p style="color: rgba(255, 255, 255, 0.5); font-size: 1rem; margin: 1.5rem 0 0.5rem 0; text-align: center;">
                🕒 Last updated: {last_update_str} ({time_ago})
            </p>
            """, unsafe_allow_html=True)
        else:
            # Nếu chưa có session state, lấy từ file predictions
            forecast_date_str = forecast_date.strftime("%d %B, %Y")
            
            # SỬA LỖI: Chuyển forecast_date (Timestamp) thành date
            forecast_date_only = forecast_date.date()
            
            # SỬA LỖI: Đảm bảo date.today() trả về datetime.date với múi giờ đúng
            tz = get_timezone()
            today = datetime.now(tz).date()
            time_diff = (today - forecast_date_only).days
            
            if time_diff == 0:
                time_ago = "today"
            elif time_diff == 1:
                time_ago = "yesterday"
            else:
                time_ago = f"{time_diff} days ago"
            
            st.markdown(f"""
            <p style="color: rgba(255, 255, 255, 0.5); font-size: 0.85rem; margin: 1.5rem 0 0.5rem 0; text-align: center;">
                🕒 Last updated: {forecast_date_str} ({time_ago})
            </p>
            """, unsafe_allow_html=True)

        # ĐÓNG FORECAST BLOCK
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)

    #     # NÚT CẬP NHẬT
    #     col1, col2, col3 = st.columns([1, 2, 1])
    #     with col2:
    #         if st.button("🔄 Force Update Now", width='stretch'):
    #             with st.spinner("Processing..."):
    #                 try:
    #                     daily_update()
    #                     st.session_state.last_update_date = date.today()
    #                     st.success("✅ Forecast updated successfully!")
    #                     st.cache_data.clear()
    #                     time.sleep(1)
    #                     st.rerun()
    #                 except Exception as e:
    #                     st.error(f"❌ Error during forecast: {e}")
        
    # else:
    #     st.warning(f"⚠️ Không tìm thấy dữ liệu dự báo của mô hình tại '{PATH_PREDICTIONS}'.")
    #     col1, col2, col3 = st.columns([1, 2, 1])
    #     with col2:
    #         if st.button("🚀 Chạy Dự báo của Mô hình lần đầu", width="stretch"):
    #             with st.spinner("Running first-time forecast..."):
    #                 try:
    #                     daily_update()
    #                     st.success("✅ Initial forecast completed!")
    #                     st.cache_data.clear()
    #                     st.rerun()
    #                 except Exception as e:
    #                     st.error(f"❌ Error: {e}")


# =============================================================================
# --- TAB 2: PHÂN TÍCH DỮ LIỆU LỊCH SỬ ---
# =============================================================================
with tab2:
    st.markdown('<p style="margin-bottom: 0rem; font-size: 1.2rem;"> ℹ️ Retraining Strategy</p>', unsafe_allow_html=True)
    st.markdown("""
        <p style="color: rgba(255, 255, 255, 0.8); font-size: 1rem; line-height: 1.6; margin-bottom: 2rem; padding-left: 0.4rem;">
            To ensure the model remains accurate, we retrain it using the most recent three years of historical data. After retraining, the new model's performance is compared against the current one. An update is deployed only if the new model demonstrates a significant improvement in accuracy.
        </p>
    """, unsafe_allow_html=True)

    # --- PHẦN MỚI: LIÊN KẾT TỚI NOTEBOOK REPORT (ĐÃ DI CHUYỂN VÀ THIẾT KẾ LẠI) ---
    st.markdown("""
        <div style="background-color: rgba(0, 123, 255, 0.2); padding: 1rem 1.5rem; border-radius: 8px; margin-bottom: 2.5rem;">
            <p style="color: rgba(255, 255, 255, 0.9); font-size: 0.95rem; margin: 0;">
                For a deeper understanding of the data and our processing methods, please view our detailed report 
                <a href="https://github.com/DunDun-banana/ml-group1/blob/main/Main_Report.ipynb" target="_blank" style="color: #80bfff; font-weight: 600;">here</a>.
            </p>
        </div>
    """, unsafe_allow_html=True)

    df_3y = load_csv(PATH_3_YEAR_DATA)

    if df_3y is not None:
        df_3y['datetime'] = pd.to_datetime(df_3y['datetime'])

        # --- Bố cục mới không dùng forecast-block ---
        st.markdown('<p class="forecast-title">📈 Historical Temperature Trend</p>', unsafe_allow_html=True)
        st.markdown('<p style="color: rgba(255, 255, 255, 0.8); font-size: 0.95rem; margin-bottom: 1.5rem;">This chart displays the daily temperature fluctuations over the selected period. You can zoom and pan to explore specific timeframes.</p>', unsafe_allow_html=True)

        min_date = df_3y['datetime'].min().date()
        max_date = df_3y['datetime'].max().date()

        with st.expander("📅 Filter by Date Range"):
            # --- SỬA LỖI: Quản lý state của radio button ---
            if 'range_option' not in st.session_state:
                st.session_state.range_option = "Last 1 Year"

            def update_range():
                st.session_state.range_option = st.session_state.radio_range
            
            st.radio(
                "Choose a period:",
                ("Last 1 Year", "Last 2 Years", "All Time", "Custom"),
                key="radio_range",
                on_change=update_range,
                horizontal=True,
                label_visibility="collapsed"
            )

            if st.session_state.range_option == "Custom":
                c1, c2 = st.columns(2)
                with c1:
                    start_date = st.date_input("Start date", min_date, min_value=min_date, max_value=max_date)
                with c2:
                    end_date = st.date_input("End date", max_date, min_value=start_date, max_value=max_date)
            else:
                end_date = max_date
                if st.session_state.range_option == "Last 1 Year":
                    start_date = end_date - timedelta(days=365)
                elif st.session_state.range_option == "Last 2 Years":
                    start_date = end_date - timedelta(days=365*2)
                else: # All Time
                    start_date = min_date
        
        # Lọc dữ liệu dựa trên lựa chọn
        mask = (df_3y['datetime'].dt.date >= start_date) & (df_3y['datetime'].dt.date <= end_date)
        filtered_df = df_3y.loc[mask]

        if not filtered_df.empty:
            # Tạo biểu đồ Altair
            chart = alt.Chart(filtered_df).mark_line(
                strokeWidth=2,
                color="#3399FF" # Màu sáng hơn để nổi bật trên nền
            ).encode(
                x=alt.X('datetime:T', title='Date', axis=alt.Axis(labelColor='white', titleColor='white', grid=False, format="%Y-%m-%d")),
                y=alt.Y('temp:Q', title='Temperature (°C)', axis=alt.Axis(labelColor='white', titleColor='white', gridColor='rgba(255, 255, 255, 0.1)')),
                tooltip=[
                    alt.Tooltip('datetime:T', title='Date', format='%A, %B %d, %Y'),
                    alt.Tooltip('temp:Q', title='Temperature', format='.1f')
                ]
            ).properties(
                background='transparent',
                height=450 # Tăng chiều cao cho biểu đồ chính
            ).configure_view(
                stroke=None
            )

            st.altair_chart(chart, width='stretch')
        else:
            st.warning("No data available for the selected date range.")

        st.markdown("<br>", unsafe_allow_html=True)

        # --- PHẦN MỚI: PHÂN TÍCH THEO THÁNG VÀ NĂM ---
        st.markdown('<p class="forecast-title">📅 Monthly & Yearly Average Temperature</p>', unsafe_allow_html=True)
        st.markdown('<p style="color: rgba(255, 255, 255, 0.8); font-size: 0.95rem; margin-bottom: 1.5rem;">These charts break down the average temperature by month and year, revealing seasonal patterns and long-term trends.</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)

        # Biểu đồ nhiệt độ trung bình theo tháng
        with col1:
            st.markdown('<p style="font-size: 1rem; color: rgba(255,255,255,0.8); text-align: center; margin-bottom: 1rem;">Average by Month</p>', unsafe_allow_html=True)
            
            df_3y['month'] = df_3y['datetime'].dt.month_name()
            monthly_avg = df_3y.groupby('month')['temp'].mean().reset_index()
            
            # Sắp xếp các tháng theo đúng thứ tự
            month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
            monthly_avg['month'] = pd.Categorical(monthly_avg['month'], categories=month_order, ordered=True)
            monthly_avg = monthly_avg.sort_values('month')

            monthly_chart = alt.Chart(monthly_avg).mark_bar(
                color="#3399FF",
                cornerRadiusTopLeft=3,
                cornerRadiusTopRight=3
            ).encode(
                x=alt.X('month:N', sort=None, title=None, axis=alt.Axis(labelAngle=-45, labelColor='white')),
                y=alt.Y('temp:Q', title='Avg Temp (°C)', axis=alt.Axis(labelColor='white', titleColor='white')),
                tooltip=[
                    alt.Tooltip('month', title='Month'),
                    alt.Tooltip('temp', title='Avg Temp', format='.1f')
                ]
            ).properties(
                background='transparent',
                height=300
            ).configure_view(
                stroke=None
            )
            st.altair_chart(monthly_chart, width='stretch')

        # Biểu đồ nhiệt độ trung bình theo năm
        with col2:
            st.markdown('<p style="font-size: 1rem; color: rgba(255,255,255,0.8); text-align: center; margin-bottom: 1rem;">Average by Year</p>', unsafe_allow_html=True)
            
            df_3y['year'] = df_3y['datetime'].dt.year
            yearly_avg = df_3y.groupby('year')['temp'].mean().reset_index()

            yearly_chart = alt.Chart(yearly_avg).mark_bar(
                color="#28a745",
                cornerRadiusTopLeft=3,
                cornerRadiusTopRight=3
            ).encode(
                x=alt.X('year:O', title=None, axis=alt.Axis(labelAngle=0, labelColor='white')),
                y=alt.Y('temp:Q', title='Avg Temp (°C)', axis=alt.Axis(labelColor='white', titleColor='white')),
                tooltip=[
                    alt.Tooltip('year:O', title='Year'),
                    alt.Tooltip('temp', title='Avg Temp', format='.1f')
                ]
            ).properties(
                background='transparent',
                height=300
            ).configure_view(
                stroke=None
            )
            st.altair_chart(yearly_chart, width='stretch')

        st.markdown("<br>", unsafe_allow_html=True)

        # --- PHẦN MỚI: PHÂN PHỐI NHIỆT ĐỘ THEO MÙA ---
        st.markdown('<p class="forecast-title">🍃 Temperature Distribution by Season</p>', unsafe_allow_html=True)
        st.markdown('<p style="color: rgba(255, 255, 255, 0.8); font-size: 0.95rem; margin-bottom: 1.5rem;">The box plot illustrates the temperature range for each season. It shows the median, quartiles, and potential outliers, providing a clear comparison of seasonal variability.</p>', unsafe_allow_html=True)

        # Hàm để xác định mùa
        def get_season(month):
            if month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            elif month in [9, 10, 11]:
                return 'Autumn'
            else:
                return 'Winter'

        df_3y['season'] = df_3y['datetime'].dt.month.apply(get_season)
        
        # Định nghĩa thứ tự và màu sắc cho các mùa
        season_order = ['Spring', 'Summer', 'Autumn', 'Winter']
        color_scheme = ['#28a745', '#ffc107', '#fd7e14', '#3399FF'] # Green, Yellow, Orange, Blue

        seasonal_chart = alt.Chart(df_3y).mark_boxplot(
            extent='min-max', # Hiển thị râu từ min đến max
            size=50
        ).encode(
            x=alt.X('season:N', sort=season_order, title=None, axis=alt.Axis(labelAngle=0, labelColor='white')),
            y=alt.Y('temp:Q', title='Temperature (°C)', axis=alt.Axis(labelColor='white', titleColor='white')),
            color=alt.Color('season:N', 
                scale=alt.Scale(domain=season_order, range=color_scheme),
                legend=None # Ẩn legend
            ),
            tooltip=[
                alt.Tooltip('season:N', title='Season'),
                alt.Tooltip('max(temp):Q', title='Max Temp', format='.1f'),
                alt.Tooltip('min(temp):Q', title='Min Temp', format='.1f'),
                alt.Tooltip('median(temp):Q', title='Median Temp', format='.1f'),
            ]
        ).properties(
            background='transparent',
            height=400
        ).configure_view(
            stroke=None
        )
        st.altair_chart(seasonal_chart, width='stretch')

        st.markdown("<br>", unsafe_allow_html=True)

        # --- PHẦN MỚI: PHÂN RÃ CHUỖI THỜI GIAN ---
        if seasonal_decompose:
            st.markdown('<p class="forecast-title">🔬 Time Series Decomposition</p>', unsafe_allow_html=True)
            st.markdown('<p style="color: rgba(255, 255, 255, 0.8); font-size: 0.95rem; margin-bottom: 1.5rem;">This analysis decomposes the time series into three components: <b>Trend</b> (the long-term progression), <b>Seasonality</b> (the yearly cyclical pattern), and <b>Residuals</b> (the random noise). This helps in understanding the underlying structure of the data.</p>', unsafe_allow_html=True)
            
            # Thực hiện phân rã trên dữ liệu đã được lọc
            # Cần ít nhất 2 chu kỳ (2*365=730 ngày) để phân rã tốt
            if len(filtered_df) > 730:
                decomposition = seasonal_decompose(filtered_df.set_index('datetime')['temp'], model='additive', period=365)
                
                # Tạo DataFrame từ kết quả
                decomp_df = pd.DataFrame({
                    'Trend': decomposition.trend,
                    'Seasonality': decomposition.seasonal,
                    'Residuals': decomposition.resid
                }).reset_index()

                # Biến đổi dữ liệu để vẽ 3 biểu đồ cùng lúc
                decomp_melted = decomp_df.melt('datetime', var_name='Component', value_name='Value')

                decomp_chart = alt.Chart(decomp_melted).mark_line().encode(
                    x=alt.X('datetime:T', title='Date', axis=alt.Axis(labelColor='white', titleColor='white', grid=False)),
                    y=alt.Y('Value:Q', title=None, axis=alt.Axis(labelColor='white', titleColor='white')),
                    color=alt.Color('Component:N', legend=alt.Legend(titleColor="white", labelColor="white")),
                    row=alt.Row('Component:N', title=None, header=alt.Header(labelColor="white", labelFontSize=14)),
                    tooltip=['datetime:T', 'Value:Q']
                ).properties(
                    background='transparent',
                    height=150
                ).configure_view(
                    stroke=None
                ).resolve_scale(
                    y='independent' # Cho phép mỗi biểu đồ có trục Y riêng
                )
                
                st.altair_chart(decomp_chart, width='stretch')
            else:
                st.info("ℹ️ Please select a date range of at least 2 years to view the time series decomposition.")
        
        st.markdown("<br>", unsafe_allow_html=True)

    else:
        st.error(f"❌ Data file not found at '{PATH_3_YEAR_DATA}'.")


# =============================================================================
# --- TAB 3: GIÁM SÁT HIỆU SUẤT MÔ HÌNH ---
# =============================================================================
with tab3:
    st.markdown('<p class="forecast-title" style="margin-bottom: 0.5rem;">⚙️ Model Performance Monitoring</p>', unsafe_allow_html=True)
    st.markdown('<p style="color: rgba(255, 255, 255, 0.8); font-size: 0.95rem; margin-bottom: 2rem;">Track and evaluate model accuracy over time. This section provides insights into the model\'s error rate, compares its predictions against actual values, and logs retraining sessions.</p>', unsafe_allow_html=True)
    
    # RMSE History Section
    st.markdown('<p class="forecast-title">📉 RMSE History Over Time</p>', unsafe_allow_html=True)
    st.markdown('<p style="color: rgba(255, 255, 255, 0.8); font-size: 0.95rem; margin-bottom: 1.5rem;">This chart tracks the Root Mean Square Error (RMSE) for each forecasting cycle. An increasing error trend may indicate that the model\'s performance is degrading and it needs to be retrained.</p>', unsafe_allow_html=True)
    
    rmse_logs = load_joblib(PATH_RMSE_LOG)
    if rmse_logs is not None:
        df_rmse = pd.DataFrame(rmse_logs)
        df_rmse['base_date'] = pd.to_datetime(df_rmse['base_date'])
        
        rmse_chart = alt.Chart(df_rmse.dropna()).mark_line(
            strokeWidth=2,
            color="#FF4B4B"
        ).encode(
            x=alt.X('base_date:T', title='Date', axis=alt.Axis(labelColor='white', titleColor='white', grid=False, format="%Y-%m-%d")),
            y=alt.Y('rmse:Q', title='RMSE Value', axis=alt.Axis(labelColor='white', titleColor='white', gridColor='rgba(255, 255, 255, 0.1)')),
            tooltip=[
                alt.Tooltip('base_date:T', title='Date', format='%Y-%m-%d'),
                alt.Tooltip('rmse:Q', title='RMSE', format='.4f')
            ]
        ).properties(
            background='transparent',
            height=350
        ).configure_view(
            stroke=None
        )
        st.altair_chart(rmse_chart, width='stretch')
    else:
        st.warning(f"⚠️ RMSE log file not found at '{PATH_RMSE_LOG}'.")
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Forecast vs Actual Comparison
    st.markdown('<p class="forecast-title">🎯 Forecast vs Actual Comparison</p>', unsafe_allow_html=True)
    st.markdown('<p style="color: rgba(255, 255, 255, 0.8); font-size: 0.95rem; margin-bottom: 1.5rem;">Select a past forecast date to compare the model\'s predictions against the actual recorded temperatures. This helps visualize the model\'s accuracy for specific periods.</p>', unsafe_allow_html=True)
    
    pred_df_comp = load_csv(PATH_PREDICTIONS)
    actual_df_comp = load_csv(PATH_RAW_3WEEKS)

    if pred_df_comp is not None and actual_df_comp is not None:
        pred_df_comp['date'] = pd.to_datetime(pred_df_comp['date'])
        actual_df_comp['datetime'] = pd.to_datetime(actual_df_comp['datetime'])

        available_dates = pred_df_comp['date'].sort_values(ascending=False)
        selected_date = st.selectbox(
            "Select a past forecast date to compare:",
            options=available_dates,
            format_func=lambda date: date.strftime('%Y-%m-%d'),
            index=0 # Mặc định chọn ngày gần nhất
        )

        selected_row = pred_df_comp[pred_df_comp['date'] == selected_date]

        if not selected_row.empty:
            forecast_dates = [selected_date + timedelta(days=i) for i in range(1, 6)]
            forecast_values = selected_row.iloc[0][[f'pred_day_{i}' for i in range(1, 6)]].values

            actual_values = []
            for d in forecast_dates:
                val = actual_df_comp.loc[actual_df_comp['datetime'].dt.date == d.date(), 'temp']
                actual_values.append(val.values[0] if not val.empty else None)

            comparison_df = pd.DataFrame({
                'Date': forecast_dates,
                'Forecast': forecast_values,
                'Actual': actual_values
            })

            # Melt dataframe for Altair
            comparison_melted = comparison_df.melt('Date', var_name='Type', value_name='Temperature')

            comp_chart = alt.Chart(comparison_melted).mark_line(
                strokeWidth=2.5
            ).encode(
                x=alt.X('Date:T', title='Date', axis=alt.Axis(labelColor='white', titleColor='white', grid=False, format="%Y-%m-%d")),
                y=alt.Y('Temperature:Q', title='Temperature (°C)', axis=alt.Axis(labelColor='white', titleColor='white', gridColor='rgba(255, 255, 255, 0.1)')),
                color=alt.Color('Type:N', 
                    scale=alt.Scale(domain=['Forecast', 'Actual'], range=['#007BFF', '#28a745']),
                    legend=alt.Legend(titleColor="white", labelColor="white")
                ),
                tooltip=[
                    alt.Tooltip('Date:T', title='Date', format='%A, %d %b'),
                    alt.Tooltip('Temperature:Q', title='Temp', format='.1f'),
                    alt.Tooltip('Type:N', title='Type')
                ]
            ).properties(
                background='transparent',
                height=350
            ).configure_view(
                stroke=None
            )
            st.altair_chart(comp_chart, width='stretch')
            
            # Display table with styling
            st.markdown('<p style="color: rgba(255, 255, 255, 0.8); font-size: 0.95rem; margin: 1.5rem 0 0.5rem 0;">Detailed Comparison Table</p>', unsafe_allow_html=True)
            
            # Format dataframe for display
            comparison_df_display = comparison_df.set_index('Date').copy()
            comparison_df_display = comparison_df_display.fillna('N/A')
            
            def format_temp(val):
                if val == 'N/A': return val
                try: return f"{float(val):.1f}"
                except: return val
            
            st.dataframe(comparison_df_display.applymap(format_temp), width='stretch')
        else:
            st.warning("⚠️ No forecast log available for the selected date.")
    else:
        st.warning(f"⚠️ Cannot find '{PATH_PREDICTIONS}' or '{PATH_RAW_3WEEKS}' for comparison.")
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Retraining History Section
    st.markdown('<p class="forecast-title">🔄 Model Retraining History</p>', unsafe_allow_html=True)
    st.markdown('<p style="color: rgba(255, 255, 255, 0.8); font-size: 0.95rem; margin-bottom: 1.5rem;">This section logs each time the model is retrained. It includes the performance metrics of the new model and the hyperparameters that yielded the best results.</p>', unsafe_allow_html=True)
    
    retrain_logs = load_joblib(PATH_RETRAIN_LOG)
    if retrain_logs:
        # Hiển thị log mới nhất lên trước
        for record in reversed(retrain_logs):
            with st.expander(f"📅 Retraining session: {record['timestamp']}"):
                col1, col2 = st.columns(2)
                with col1:
                    rmse_val = record.get('metrics', {}).get('average', {}).get('RMSE', 0)
                    st.metric("Average RMSE", f"{rmse_val:.4f}" if rmse_val else "N/A")
                with col2:
                    st.metric("Status", "Completed")
                
                st.markdown("**Best Hyperparameters:**")
                best_params = record.get('best_params', {})
                if best_params:
                    st.json(best_params, expanded=False)
                else:
                    st.write("No parameters recorded.")
    else:
        st.info("ℹ️ No retraining history has been recorded yet.")
    
    st.markdown("<br>", unsafe_allow_html=True)