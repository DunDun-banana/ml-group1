import pandas as pd
import joblib
import os
from pathlib import Path
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
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
        return ["642BDT8N8D49CTFJCX8ZWU6RT"]  # Thêm một key mặc định để tránh lỗi

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
        st.error(f"⚠️ Lỗi khi cập nhật tự động: {e}")
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
        font-size: 1rem;
        font-weight: 500;
        padding: 0 1.5rem;
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
            <p style="color: rgba(255, 255, 255, 0.5); font-size: 0.85rem; margin: 1rem 0 1.5rem 0; text-align: center;">
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
            <p style="color: rgba(255, 255, 255, 0.5); font-size: 0.85rem; margin: 1rem 0 1.5rem 0; text-align: center;">
                🕒 Last updated: {forecast_date_str} ({time_ago})
            </p>
            """, unsafe_allow_html=True)
        
        st.markdown('<p class="forecast-title">📈 Temperature Forecast Trend</p>', unsafe_allow_html=True)
        try:
            fig, ax = plt.subplots(figsize=(12, 3.5))
            
            # Set background color
            fig.patch.set_facecolor('none')
            ax.set_facecolor('none')
            
            # Plot line with gradient fill
            date_labels = [d.strftime('%a\n%d/%m') for d in forecast_dates]
            x_pos = list(range(len(forecast_values)))
            
            # Ensure forecast_values are numeric
            forecast_values_clean = [float(v) for v in forecast_values]
            
            # Draw line
            line = ax.plot(x_pos, forecast_values_clean, color='#4FC3F7', linewidth=2, marker='o', 
                           markersize=8, markerfacecolor='#81D4FA', markeredgewidth=2, 
                           markeredgecolor='#FFFFFF', zorder=3)
            
            # Fill area under curve with gradient effect
            ax.fill_between(x_pos, forecast_values_clean, alpha=0.2, color='#0D3B4F')
            
            # Set labels
            ax.set_xticks(x_pos)
            ax.set_xticklabels(date_labels, fontsize=10, color='#FFFFFF')
            
            # Remove spines
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # Ẩn trục y
            ax.yaxis.set_visible(False)
            
            # Customize ticks
            ax.tick_params(axis='x', colors='#FFFFFF', labelsize=10, length=0)
            
            # Add value labels on points
            for i, (x, y) in enumerate(zip(x_pos, forecast_values_clean)):
                ax.text(x, y + 0.8, f'{y:.1f}°', ha='center', va='bottom', 
                       fontsize=9, color='#81D4FA')
            
            # Adjust layout
            plt.tight_layout()
            
            # Display chart
            st.pyplot(fig)
            plt.close()
            
        except Exception as e:
            st.error(f"Lỗi khi vẽ biểu đồ: {e}")
        
        # ĐÓNG FORECAST BLOCK
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)

    #     # NÚT CẬP NHẬT
    #     col1, col2, col3 = st.columns([1, 2, 1])
    #     with col2:
    #         if st.button("🔄 Force Update Now", use_container_width=True):
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
    st.markdown('<p class="forecast-title" style="margin-bottom: 0.5rem;">📊 Historical Data Analysis</p>', unsafe_allow_html=True)
    st.markdown('<p style="color: rgba(255, 255, 255, 0.6); font-size: 0.95rem; margin-bottom: 2rem;">Explore the data used to train the prediction model</p>', unsafe_allow_html=True)
    
    df_3y = load_csv(PATH_3_YEAR_DATA)

    if df_3y is not None:
        df_3y['datetime'] = pd.to_datetime(df_3y['datetime'])

        # Temperature Trend Section
        st.markdown("""
        <div class="forecast-block">
            <p class="forecast-title">📈 3-Year Temperature Trend</p>
        """, unsafe_allow_html=True)
        
        st.line_chart(df_3y.set_index('datetime')['temp'], height=400)
        
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Correlation Matrix Section
        st.markdown("""
        <div class="forecast-block">
            <p class="forecast-title">🔗 Feature Correlation Matrix</p>
            <p style="color: rgba(255, 255, 255, 0.6); font-size: 0.9rem; margin-bottom: 1rem;">
                This heatmap shows linear relationships between weather features. 
                Colors closer to +1 (red) or -1 (blue) indicate stronger correlations.
            </p>
        """, unsafe_allow_html=True)

        numeric_cols = df_3y.select_dtypes(include=['number']).columns
        corr = df_3y[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        
        # Sửa màu linecolor thành tuple RGBA thay vì string
        sns.heatmap(corr, ax=ax, cmap='coolwarm', annot=False, 
                   cbar_kws={'label': 'Correlation Coefficient'},
                   linewidths=0.5, linecolor=(1, 1, 1, 0.1))  # Sử dụng tuple RGBA
        
        ax.tick_params(colors='white', labelsize=9)
        
        # Thay đổi màu của cbar label
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.label.set_color('white')
        cbar.ax.tick_params(colors='white')
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        st.pyplot(fig)
        plt.close()
        
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # # Raw Data Section
        # if st.checkbox("📋 Show Raw Data"):
        #     st.markdown("""
        #     <div class="forecast-block">
        #         <p class="forecast-title">Raw Dataset</p>
        #     """, unsafe_allow_html=True)
            
        #     st.dataframe(df_3y, height=400)
            
        #     st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.error(f"❌ Data file not found at '{PATH_3_YEAR_DATA}'.")


# =============================================================================
# --- TAB 3: GIÁM SÁT HIỆU SUẤT MÔ HÌNH ---
# =============================================================================
with tab3:
    st.markdown('<p class="forecast-title" style="margin-bottom: 0.5rem;">⚙️ Model Performance Monitoring</p>', unsafe_allow_html=True)
    st.markdown('<p style="color: rgba(255, 255, 255, 0.6); font-size: 0.95rem; margin-bottom: 2rem;">Track and evaluate model accuracy over time</p>', unsafe_allow_html=True)
    
    # RMSE History Section
    st.markdown("""
    <div class="forecast-block">
        <p class="forecast-title">📉 RMSE History Over Time</p>
    """, unsafe_allow_html=True)
    
    rmse_logs = load_joblib(PATH_RMSE_LOG)
    if rmse_logs is not None:
        df_rmse = pd.DataFrame(rmse_logs)
        df_rmse['base_date'] = pd.to_datetime(df_rmse['base_date'])
        st.line_chart(df_rmse.set_index('base_date')['rmse'].dropna(), height=300)
        st.caption("⚠️ An increasing error trend may indicate the model needs retraining.")
    else:
        st.warning(f"⚠️ RMSE log file not found at '{PATH_RMSE_LOG}'.")
    
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Forecast vs Actual Comparison
    st.markdown("""
    <div class="forecast-block">
        <p class="forecast-title">🎯 Forecast vs Actual Comparison</p>
    """, unsafe_allow_html=True)
    
    pred_df_comp = load_csv(PATH_PREDICTIONS)
    actual_df_comp = load_csv(PATH_RAW_3WEEKS)

    if pred_df_comp is not None and actual_df_comp is not None:
        pred_df_comp['date'] = pd.to_datetime(pred_df_comp['date'])
        actual_df_comp['datetime'] = pd.to_datetime(actual_df_comp['datetime'])

        available_dates = pred_df_comp['date']
        selected_date = st.selectbox(
            "Select a past forecast date to compare:",
            options=available_dates,
            format_func=lambda date: date.strftime('%Y-%m-%d'),
            index=len(available_dates) - 1 if not available_dates.empty else 0
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
            }).set_index('Date')

            st.line_chart(comparison_df, height=300)
            
            # Display table with styling
            st.markdown('<p style="color: rgba(255, 255, 255, 0.7); font-size: 0.9rem; margin: 1rem 0 0.5rem 0;">Detailed Comparison Table</p>', unsafe_allow_html=True)
            
            # Format dataframe với xử lý None values
            comparison_df_display = comparison_df.copy()
            comparison_df_display = comparison_df_display.fillna('N/A')
            
            # Chỉ format những giá trị không phải N/A
            def format_temp(val):
                if val == 'N/A':
                    return val
                try:
                    return f"{float(val):.1f}"
                except:
                    return val
            
            st.dataframe(comparison_df_display.map(format_temp), height=200)
        else:
            st.warning("⚠️ No forecast log available for the selected date.")
    else:
        st.warning(f"⚠️ Cannot find '{PATH_PREDICTIONS}' or '{PATH_RAW_3WEEKS}' for comparison.")
    
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Retraining History Section
    st.markdown("""
    <div class="forecast-block">
        <p class="forecast-title">🔄 Model Retraining History</p>
    """, unsafe_allow_html=True)
    
    retrain_logs = load_joblib(PATH_RETRAIN_LOG)
    if retrain_logs:
        for record in reversed(retrain_logs):
            with st.expander(f"📅 Retraining session: {record['timestamp']}"):
                col1, col2 = st.columns(2)
                with col1:
                    # Xử lý trường hợp metrics có thể là None
                    rmse_val = record.get('metrics', {}).get('average', {}).get('RMSE', 0)
                    st.metric("Average RMSE", f"{rmse_val:.4f}" if rmse_val else "N/A")
                with col2:
                    st.metric("Sessions Completed", "1")
                
                st.markdown("**Best Hyperparameters:**")
                best_params = record.get('best_params', {})
                if best_params:
                    st.json(best_params, expanded=False)
                else:
                    st.write("No parameters recorded")
    else:
        st.info("ℹ️ No retraining history recorded yet.")
    
    st.markdown("</div>", unsafe_allow_html=True)