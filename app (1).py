import os
import requests
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

# Cấu hình trang cơ bản
st.set_page_config(
    page_title="Hệ Thống Thanh Toán Khay Cơm AI",
    page_icon="🍲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Bảng giá và Tên món ăn
PRICE_MAP = {
    "Cơm trắng": 10000,
    "Trứng chiên": 25000,
    "Khay inox (Trống)": 0,
    "Đậu hũ sốt cà": 25000,
    "Cá hú kho": 30000,
    "Thịt kho trứng": 30000,
    "Thịt kho": 25000,
    "Canh chua": 25000,
    "Sườn nướng": 30000,
    "Canh rau": 7000,
    "Rau xào": 10000
}

CLASS_NAMES = [
    "Cơm trắng",
    "Trứng chiên",
    "Khay inox (Trống)",
    "Đậu hũ sốt cà",
    "Cá hú kho",
    "Thịt kho trứng",
    "Thịt kho",
    "Canh chua",
    "Sườn nướng",
    "Canh rau",
    "Rau xào"
]

# Hàm khởi tạo và tải Model (Sử dụng Cache để tải 1 lần)
@st.cache_resource
def init_model():
    model_path = "canteen_model_STAGE1.keras"
    model_url = "https://github.com/TienManh15072007/AI_FINALPROJECT_FOODTRAYREGCONITION/releases/download/model.py/canteen_model_STAGE2_latest.keras"
    
    if not os.path.exists(model_path):
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    
    return tf.keras.models.load_model(model_path)

# Hàm tự động phân tích và căn lề khay cơm
def auto_align_tray(img):
    h, w, _ = img.shape
    if w > h:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_resized = cv2.resize(gray, (400, 300))
        top_half = gray_resized[0:150, :]
        bottom_half = gray_resized[150:300, :]
        sobel_x_top = cv2.Sobel(top_half, cv2.CV_64F, 1, 0, ksize=3)
        sobel_x_bottom = cv2.Sobel(bottom_half, cv2.CV_64F, 1, 0, ksize=3)
        score_top = np.sum(np.abs(sobel_x_top))
        score_bottom = np.sum(np.abs(sobel_x_bottom))
        
        if score_top < score_bottom:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_resized = cv2.resize(gray, (300, 400))
    left_half = gray_resized[:, 0:150]
    right_half = gray_resized[:, 150:300]
    sobel_y_left = cv2.Sobel(left_half, cv2.CV_64F, 0, 1, ksize=3)
    sobel_y_right = cv2.Sobel(right_half, cv2.CV_64F, 0, 1, ksize=3)
    score_left = np.sum(np.abs(sobel_y_left))
    score_right = np.sum(np.abs(sobel_y_right))

    if score_left > score_right:
        img = cv2.rotate(img, cv2.ROTATE_180)

    return img

# Phân loại Badge món ăn dựa trên tên
def get_food_badge(food_name):
    if "Canh" in food_name:
        return "CANH", "#F5ECE1", "#9C6644"
    elif "Rau" in food_name or "Xào" in food_name:
        return "MÓN RAU XÀO", "#E8F5E9", "#2E7D32"
    elif "Khay" in food_name:
        return "TRỐNG", "#ECEFF1", "#455A64"
    else:
        return "MÓN MẶN", "#FBE9E7", "#D84315"

# ==========================================
# GIAO DIỆN CHIA TRANG BẰNG SIDEBAR
# ==========================================
st.sidebar.title("🧭 MENU CHÍNH")
page = st.sidebar.radio("Điều hướng:", ["Trang Chủ (Giới thiệu)", "Hệ Thống Nhận Diện", "Góc Ẩm Thực AI"])

# ------------------------------------------
# TRANG 1: MARKETING & GIỚI THIỆU
# ------------------------------------------
if page == "Trang Chủ (Giới thiệu)":
    marketing_bg = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Montserrat', sans-serif;
    }
    .stApp {
        background-image: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.7)), url("https://images.unsplash.com/photo-1544025162-d76694265947?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    h1, h2, h3, p, span, div {
        color: white !important;
    }
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        margin-top: 15vh;
        color: #F39C12 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
    }
    .sub-title {
        font-size: 1.5rem;
        text-align: center;
        margin-bottom: 50px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
    }
    </style>
    """
    st.markdown(marketing_bg, unsafe_allow_html=True)
    st.markdown("<h1 class='main-title'>CANTEEN AI SYSTEM</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-title'>Giải pháp nhận diện khay cơm và thanh toán tự động bằng công nghệ Computer Vision</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("🎯 **Chính xác cao**\n\nNhận diện nhanh chóng từng món ăn trong khay với độ chính xác vượt trội.")
    with col2:
        st.success("⚡ **Tốc độ chớp nhoáng**\n\nXóa bỏ cảnh xếp hàng dài chờ tính tiền. Mọi thứ hoàn tất trong vài giây.")
    with col3:
        st.warning("📊 **Quản lý dễ dàng**\n\nHệ thống tự động xuất hóa đơn và thống kê doanh thu minh bạch.")

# ------------------------------------------
# TRANG 2: HỆ THỐNG XỬ LÝ CHÍNH
# ------------------------------------------
elif page == "Hệ Thống Nhận Diện":
    app_bg = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background-color: #F9F9F9;
    }
    .step-banner {
        background: #586F56;
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 1.1rem;
        margin-bottom: 15px;
        margin-top: 25px;
    }
    [data-testid="stFileUploadDropzone"] {
        background-color: #ffffff !important;
        border: 2px dashed #586F56 !important;
        border-radius: 12px !important;
    }
    
    /* TOÀN BỘ KHU VỰC HÓA ĐƠN THEO THIẾT KẾ MỚI */
    .canteen-invoice-card {
        background-color: #FFFFFF;
        border: 1px solid #EAEAEA;
        border-radius: 24px;
        padding: 30px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.02);
    }
    .invoice-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #EEEEEE;
        padding-bottom: 15px;
        margin-bottom: 20px;
    }
    .invoice-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #264653;
        margin: 0;
    }
    .invoice-subtitle {
        font-size: 0.95rem;
        color: #9A9A9A;
        margin-top: 5px;
    }
    .model-badge {
        background-color: #E2E8F0;
        color: #4A5568;
        font-family: monospace;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: bold;
    }
    .receipt-btn {
        background-color: #586F56;
        color: white;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    /* Item món ăn trong hóa đơn */
    .food-item-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 15px 0;
        border-bottom: 1px dashed #F0F0F0;
    }
    .food-item-left {
        display: flex;
        align-items: center;
        gap: 15px;
    }
    .food-item-img-container {
        position: relative;
        width: 70px;
        height: 70px;
    }
    .food-item-img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        border-radius: 16px;
    }
    .food-number-tag {
        position: absolute;
        top: -5px;
        left: -5px;
        background: #4A4A4A;
        color: white;
        font-size: 0.7rem;
        font-weight: bold;
        padding: 2px 6px;
        border-radius: 8px;
    }
    .food-details {
        display: flex;
        flex-direction: column;
        gap: 4px;
    }
    .food-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #333333;
    }
    .badge-container {
        display: flex;
        gap: 8px;
        align-items: center;
    }
    .category-badge {
        font-size: 0.75rem;
        font-weight: 700;
        padding: 3px 10px;
        border-radius: 4px;
    }
    .accuracy-badge {
        background-color: #EDF2F7;
        color: #4A5568;
        font-size: 0.75rem;
        font-weight: 500;
        padding: 3px 8px;
        border-radius: 4px;
    }
    .food-price {
        font-size: 1.15rem;
        font-weight: 700;
        color: #333333;
    }
    
    /* Khu vực tổng cộng */
    .total-box {
        background-color: #FDFBF7;
        border-radius: 16px;
        padding: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 25px;
        margin-bottom: 25px;
    }
    .total-label {
        font-size: 1.1rem;
        font-weight: bold;
        color: #8C8275;
        letter-spacing: 0.5px;
    }
    .total-price-value {
        font-size: 2.3rem;
        font-weight: 800;
        color: #435241;
    }
    
    /* Khung hình thức thanh toán */
    .payment-title {
        font-size: 0.85rem;
        font-weight: 700;
        color: #A39684;
        text-align: center;
        margin-bottom: 15px;
        letter-spacing: 0.5px;
    }
    .payment-options {
        display: flex;
        gap: 15px;
        margin-bottom: 25px;
    }
    .payment-box {
        flex: 1;
        border-radius: 12px;
        padding
