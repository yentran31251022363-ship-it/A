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

/* ÉP TẤT CẢ CHỮ MẶC ĐỊNH TRÊN TRANG THÀNH MÀU ĐEN TUYỆT ĐỐI */
.stApp, .stApp p, .stApp span, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp label {
    color: #000000 !important;
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
/* CHỮ TRÊN CÁC TAB (GIỚI THIỆU / NHẬN DIỆN / GÓC ẨM THỰC) MÀU ĐEN */
div[data-testid="stTabs"] button p {
    color: #000000 !important;
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
.food-title, .food-price, .invoice-title, .invoice-subtitle, .kpi-value, .card-title, .food-info-name {
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
    model_url = "
