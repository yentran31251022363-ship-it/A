import os
import requests
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import base64
from io import BytesIO

# ==========================================
# 1. CẤU HÌNH TRANG CƠ BẢN VÀ MENU NGANG
# ==========================================
st.set_page_config(
    page_title="Hệ Thống Thanh Toán Khay Cơm AI",
    page_icon="🍲",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# THAY ĐỔI GIAO DIỆN: Đổi màu nền toàn trang, chỉnh chữ màu đen, nổi bật ô thanh toán
custom_ui_style = """
<style>
/* Đổi màu nền nhẹ dịu mắt cho toàn bộ trang web */
.stApp {
    background-color: #FAF7F0 !important;
}

/* Ẩn hoàn toàn thanh bên sidebar */
[data-testid="stSidebar"] {
    display: none !important;
}
[data-testid="collapsedControl"] {
    display: none !important;
}

/* Khối Tab ngang phía trên rộng rãi */
div[data-testid="stTabs"] {
    margin-top: -30px;
    margin-bottom: 20px;
}
div[data-testid="stTabs"] button {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    padding: 12px 24px !important;
}

/* CẤU HÌNH CÁC Ô LỰA CHỌN THANH TOÁN (BIẾN ST.RADIO THÀNH Ô VUÔNG NỔI BẬT) */
div[data-testid="stRadio"] > div {
    display: flex !important;
    flex-direction: row !important;
    gap: 15px !important;
    padding-top: 10px;
}
div[data-testid="stRadio"] label {
    background-color: #FFFFFF !important;
    border: 2px solid #D6C7A1 !important; /* Viền nổi bật hơn */
    border-radius: 14px !important;
    padding: 15px 25px !important;
    width: 100% !important;
    text-align: center !important;
    cursor: pointer !important;
    transition: all 0.2s ease-in-out !important;
    box-shadow: 0 4px 10px rgba(0,0,0,0.04) !important;
}
/* Hiệu ứng khi hover vào ô chọn */
div[data-testid="stRadio"] label:hover {
    border-color: #C2410C !important; /* Đổi màu viền cam đậm khi di chuột */
    background-color: #FFFDF9 !important;
    transform: translateY(-2px);
}
/* XÓA BUTTON TRÒN TRƯỚC CÁC CHOICE VÀ PHỐI MÀU LÀM NỔI BẬT KHI ĐƯỢC CHỌN */
div[data-testid="stRadio"] [data-checked="true"] ~ label {
    border-color: #EA580C !important; /* Viền cam đậm nổi bật */
    background-color: #FFEDD5 !important; /* Nền cam nhẹ sang xịn mịn */
    box-shadow: 0 6px 15px rgba(234, 88, 12, 0.2) !important;
}
div[data-testid="stRadio"] [data-checked="true"] ~ label p {
    font-weight: 700 !important;
    color: #C2410C !important; /* Chữ màu cam đậm khi active */
}
/* Ẩn hoàn toàn dấu chấm tròn radio mặc định */
div[data-testid="stRadio"] input[type="radio"] {
    display: none !important;
}
div[data-testid="stRadio"] div[data-testid="stMarkdownContainer"] p {
    margin: 0 !important;
    font-size: 1rem !important;
    color: #000000 !important; /* Chữ đen */
}

/* ÉP TẤT CẢ CÁC ĐOẠN TEXT CHÍNH TRONG HÓA ĐƠN THÀNH MÀU ĐEN */
.food-title, .food-price, .invoice-title, .invoice-subtitle {
    color: #000000 !important;
}
</style>
"""
st.markdown(custom_ui_style, unsafe_allow_html=True)

# Khởi tạo Menu Ngang dạng Tabs phía trên cùng thay cho Sidebar cũ
tabs = st.tabs(["🏠 Trang Chủ (Giới thiệu)", "📷 Hệ Thống Nhận Diện", "📊 Góc Ẩm Thực AI"])

# Bảng giá và Danh mục món ăn
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
    "Cơm trắng", "Trứng chiên", "Khay inox (Trống)", "Đậu hũ sốt cà",
    "Cá hú kho", "Thịt kho trứng", "Thịt kho", "Canh chua",
    "Sườn nướng", "Canh rau", "Rau xào"
]

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

def auto_align_tray(img):
    h, w, _ = img.shape
    if w > h:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_resized = cv2.resize(gray, (400, 300))
        top_half = gray_resized[0:150, :]
        bottom_half = gray_resized[150:300, :]
        sobel_x_top = cv2.Sobel(top_half, cv2.CV_64F, 1, 0, ksize=3)
        sobel_x_bottom = cv2.Sobel(bottom_half, cv2.CV_64F, 1, 0, ksize=3)
        if np.sum(np.abs(sobel_x_top)) < np.sum(np.abs(sobel_x_bottom)):
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_resized = cv2.resize(gray, (300, 400))
    left_half = gray_resized[:, 0:150]
    right_half = gray_resized[:, 150:300]
    if np.sum(np.abs(cv2.Sobel(left_half, cv2.CV_64F, 0, 1, ksize=3))) > np.sum(np.abs(cv2.Sobel(right_half, cv2.CV_64F, 0, 1, ksize=3))):
        img = cv2.rotate(img, cv2.ROTATE_180)
    return img

def get_food_badge(food_name):
    if "Canh" in food_name: return "CANH", "#F5ECE1", "#000000"
    elif "Rau" in food_name or "Xào" in food_name: return "MÓN RAU XÀO", "#E8F5E9", "#000000"
    elif "Khay" in food_name: return "TRỐNG", "#ECEFF1", "#000000"
    else: return "MÓN MẶN", "#FBE9E7", "#000000"

# ==========================================
# TRANG 1: MARKETING & GIỚ
