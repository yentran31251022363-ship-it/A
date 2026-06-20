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
    html, body, [class*="css"] { font-family: 'Montserrat', sans-serif; }
    .stApp {
        background-image: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.7)), url("https://images.unsplash.com/photo-1544025162-d76694265947?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80");
        background-size: cover; background-position: center; background-attachment: fixed;
    }
    h1, h2, h3, p, span, div { color: white !important; }
    .main-title { font-size: 3.5rem; font-weight: 800; text-align: center; margin-top: 1vh; color: #F39C12 !important; text-shadow: 2px 2px 4px rgba(0,0,0,0.8); }
    .sub-title { font-size: 1.5rem; text-align: center; margin-bottom: 50px; text-shadow: 1px 1px 2px rgba(0,0,0,0.8); }
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
    html, body, [class*="css"], .stApp { 
        font-family: 'Inter', sans-serif; 
        background-color: #FDF6E2 !important; 
        color: #000000 !important;
    }
    .step-banner { 
        background: #586F56; 
        color: white !important; 
        padding: 12px 20px; 
        border-radius: 8px; 
        font-weight: 600; 
        font-size: 1.1rem; 
        margin-bottom: 15px; 
        margin-top: 25px; 
    }
    .step-banner * { color: white !important; }
    [data-testid="stFileUploadDropzone"] { 
        background-color: #ffffff !important; 
        border: 2px dashed #586F56 !important; 
        border-radius: 12px !important; 
    }
    
    .canteen-invoice-card { 
        background-color: #FFFFFF; 
        border: 1px solid #E0D4B7; 
        border-radius: 24px; 
        padding: 30px; 
        box-shadow: 0 10px 25px rgba(0,0,0,0.04); 
    }
    .invoice-header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #EAE0C5; padding-bottom: 15px; margin-bottom: 20px; }
    .invoice-title { font-size: 1.8rem; font-weight: 700; color: #264653 !important; margin: 0; }
    .invoice-subtitle { font-size: 0.95rem; color: #666666 !important; margin-top: 5px; }
    .model-badge { background-color: #E2E8F0; color: #4A5568 !important; font-family: monospace; padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: bold; }
    .receipt-btn { background-color: #586F56; color: white !important; padding: 6px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: 600; }
    
    .food-item-row { display: flex; align-items: center; justify-content: space-between; padding: 15px 0; border-bottom: 1px dashed #EAE0C5; }
    .food-item-left { display: flex; align-items: center; gap: 15px; }
    .food-item-img-container { position: relative; width: 70px; height: 70px; }
    .food-item-img { width: 100%; height: 100%; object-fit: cover; border-radius: 16px; }
    .food-number-tag { position: absolute; top: -5px; left: -5px; background: #4A4A4A; color: white !important; font-size: 0.7rem; font-weight: bold; padding: 2px 6px; border-radius: 8px; }
    .food-details { display: flex; flex-direction: column; gap: 4px; }
    .food-title { font-size: 1.05rem; font-weight: 600; color: #111111 !important; }
    .badge-container { display: flex; gap: 8px; align-items: center; }
    .category-badge { font-size: 0.75rem; font-weight: 700; padding: 3px 10px; border-radius: 4px; }
    .accuracy-badge { background-color: #EDF2F7; color: #4A5568 !important; font-size: 0.75rem; font-weight: 500; padding: 3px 8px; border-radius: 4px; }
    .food-price { font-size: 1.15rem; font-weight: 700; color: #111111 !important; }
    
    .total-box { background-color: #FFFDF6; border: 1px solid #EAE0C5; border-radius: 16px; padding: 20px; display: flex; justify-content: space-between; align-items: center; margin-top: 25px; margin-bottom: 25px; }
    .total-label { font-size: 1.1rem; font-weight: bold; color: #5D5446 !important; letter-spacing: 0.5px; }
    .total-price-value { font-size: 2.3rem; font-weight: 800; color: #435241 !important; }
    
    .payment-title { font-size: 0.85rem; font-weight: 700; color: #7D705C !important; text-align: center; margin-bottom: 15px; letter-spacing: 0.5px; }
    
    /* Giao diện nút bấm lựa chọn phương thức thanh toán kiểu mới */
    .payment-options-grid { display: flex; gap: 12px; margin-bottom: 20px; }
    .payment-choice-box { flex: 1; text-align: center; padding: 12px 6px; border-radius: 12px; font-weight: 600; font-size: 0.95rem; border: 1px solid #D0C4A7; background-color: #FFFFFF; color: #6E7B8B !important; }
    .pay-selected { background-color: #E6ECE6; border: 2px solid #586F56; color: #2F3E2E !important; }
    
    .terminal-console { background-color: #222222; border-radius: 12px; padding: 15px; font-family: monospace; color: #AAAAAA !important; font-size: 0.85rem; max-height: 120px; overflow-y: auto; }
    label, p, span, h1, h2, h3, h4, div { color: #000000 !important; }
    </style>
    """
    st.markdown(app_bg, unsafe_allow_html=True)

    with st.spinner("⏳ Khởi động động cơ AI, vui lòng đợi trong giây lát..."):
        model = init_model()

    st.markdown("<h2 style='text-align: center; color: #264653 !important;'>HỆ THỐNG KIỂM TRA & THANH TOÁN KHAY CƠM TỰ ĐỘNG</h2>", unsafe_allow_html=True)
    st.write("---")

    col_left, col_right = st.columns([1.1, 1.3], gap="large")

    with col_left:
        st.markdown("<div class='step-banner'>📸 BƯỚC 1 & 2: THU THẬP & CĂN LỀ KHAY ĂN</div>", unsafe_allow_html=True)
        camera_file = st.camera_input("Chụp ảnh khay ăn trực tiếp")
        uploaded_file = st.file_uploader("Hoặc tải ảnh lên từ thiết bị", type=["jpg", "jpeg", "png"])
        
        rotation_mode = st.radio(
            "Góc xoay hiệu chỉnh tối ưu từ AI Co-pilot:",
            ("Tự động chỉnh hướng", "Giữ nguyên (0°)", "Xoay 90° CW", "Xoay 90° CCW", "Xoay 180°"),
            horizontal=True
        )

    active_file = camera_file if camera_file is not None else uploaded_file

    if active_file is not None:
        image = Image.open(active_file).convert('RGB')
        img_array = np.array(image)

        if rotation_mode == "Tự động chỉnh hướng":
            img_aligned = auto_align_tray(img_array)
        elif rotation_mode == "Xoay 90° CW":
            img_aligned = cv2.rotate(img_array, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_mode == "Xoay 90° CCW":
            img_aligned = cv2.rotate(img_array, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotation_mode == "Xoay 180°":
            img_aligned = cv2.rotate(img_array, cv2.ROTATE_180)
        else:
            img_aligned = img_array

        with col_left:
            st.image(img_aligned, use_container_width=True, caption="Khay cơm chuẩn hóa đưa vào AI core")

        h, w, _ = img_aligned.shape
        regions = {
            "Món chính": img_aligned[int(h*0.66):int(h*0.98), int(w*0.56):int(w*0.98)],
            "Món rau xào": img_aligned[int(h*0.02):int(h*0.32), int(w*0.56):int(w*0.98)],
            "Canh": img_aligned[int(h*0.46):int(h*0.98), int(w*0.02):int(w*0.54)],
            "Cơm trắng": img_aligned[int(h*0.02):int(h*0.44), int(w*0.02):int(w*0.54)],
            "Nước chấm": img_aligned[int(h*0.34):int(h*0.64), int(w*0.56):int(w*0.98)]
        }

        total_bill = 0
