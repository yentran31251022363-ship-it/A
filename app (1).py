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

# THAY ĐỔI GIAO DIỆN: Ép chữ đen toàn diện và cấu hình phông chữ chống lỗi tiếng Việt
custom_ui_style = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Montserrat:wght@400;600;700;800&display=swap');

/* Áp dụng phông chữ an toàn cho toàn bộ ứng dụng */
body, .stApp, p, span, h1, h2, h3, h4, h5, h6, label, div {
    font-family: 'Inter', 'Arial', sans-serif !important;
}

/* Đổi màu nền nhẹ dịu mắt cho toàn bộ trang web */
.stApp {
    background-color: #FAF7F0 !important;
}

/* ÉP TẤT CẢ CHỮ MẶC ĐỊNH TRÊN TRANG THÀNH MÀU ĐEN */
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
/* CHỮ TRÊN CÁC TAB MÀU ĐEN */
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
    border-color: #C2410C !important;
    background-color: #FFFDF9 !important;
    transform: translateY(-2px);
}

/* XÓ
