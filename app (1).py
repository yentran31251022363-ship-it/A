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
    .sub
