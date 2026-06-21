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

/* Sửa lỗi mờ chữ trong các thẻ KPI Dashboard và text nội dung */
.kpi-value, .card-title, .food-info-name, .food-title, .food-price, .invoice-title, .invoice-subtitle {
    color: #000000 !important;
}

/* ĐỒNG BỘ PHÔNG CHỮ TỔNG CỘNG GIỐNG VỚI KẾT QUẢ - CHỐNG LỖI ĐỊNH DẠNG */
.total-label {
    font-family: 'Inter', 'Arial', sans-serif !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: #000000 !important;
    margin: 0 !important;
    line-height: 1.2 !important;
}

/* Ép luôn màu đen và độ đậm cho kết quả số tiền (thẻ h2 bên trong Streamlit) */
div[data-testid="stMarkdownContainer"] h2 {
    color: #000000 !important;
    font-weight: 700 !important;
    margin-top: 0px !important;
    font-family: 'Inter', 'Arial', sans-serif !important;
}

/* Khung quét mã QR tinh tế */
.qr-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background-color: #F8FAF5;
    border: 2px dashed #A3B899;
    border-radius: 16px;
    padding: 20px;
    margin-top: 15px;
    margin-bottom: 15px;
    text-align: center;
}
.qr-text {
    font-size: 0.9rem;
    font-weight: 600;
    color: #3D4A3A !important;
    margin-top: 10px;
}
</style>
"""
st.markdown(custom_ui_style, unsafe_allow_html=True)

# Hàm bổ trợ mã hóa ảnh local sang base64 để render trực tiếp vào mã HTML
def get_base64_encoded_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return None

# Khởi tạo Menu Ngang dạng Tabs phía trên cùng thay cho Sidebar cũ
tabs = st.tabs(["🏠 Trang Chủ (Giới thiệu)", "📷 Hệ Thống Nhận Diện", "📊 Thống kê"])

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
# TRANG 1: MARKETING & GIỚI THIỆU
# ==========================================
with tabs[0]:
    marketing_bg = """
    <style>
    .marketing-wrapper {
        font-family: 'Montserrat', 'Inter', sans-serif;
        background-image: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.85)), url("https://images.unsplash.com/photo-1544025162-d76694265947?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80");
        background-size: cover; background-position: center; padding: 60px; border-radius: 24px; margin-top: 10px;
    }
    .marketing-wrapper h1, .marketing-wrapper p { color: white !important; text-align: center; }
    .main-title { font-size: 3.5rem; font-weight: 800; color: #F39C12 !important; text-shadow: 2px 2px 4px rgba(0,0,0,0.8); }
    .sub-title { font-size: 1.4rem; margin-bottom: 40px; text-shadow: 1px 1px 2px rgba(0,0,0,0.8); }
    </style>
    <div class="marketing-wrapper">
        <h1 class='main-title'>CANTEEN AI SYSTEM</h1>
        <p class='sub-title'>Giải pháp nhận diện khay cơm và thanh toán tự động bằng công nghệ Computer Vision</p>
    </div>
    """
    st.markdown(marketing_bg, unsafe_allow_html=True)
    st.write("")
    col1, col2, col3 = st.columns(3)
    with col1: st.info("🎯 **Chính xác cao**\n\nNhận diện nhanh chóng từng món ăn trong khay với độ chính xác vượt trội.")
    with col2: st.success("⚡ **Tốc độ chớp nhoáng**\n\nXóa bỏ cảnh xếp hàng dài chờ tính tiền. Mọi thứ hoàn tất trong vài giây.")
    with col3: st.warning("📊 **Quản lý dễ dàng**\n\nHệ thống tự động xuất hóa đơn và thống kê doanh thu minh bạch.")


# ==========================================
# TRANG 2: HỆ THỐNG XỬ LÝ CHÍNH
# ==========================================
with tabs[1]:
    app_bg = """
    <style>
    .main-body-style { font-family: 'Inter', 'Arial', sans-serif; }
    .step-banner { background: #586F56; color: white !important; padding: 12px 20px; border-radius: 8px; font-weight: 600; font-size: 1.1rem; margin-bottom: 15px; margin-top: 10px; }
    .canteen-invoice-card, .st-key-invoice_card { background-color: #FFFFFF; border: 1px solid #E0D4B7; border-radius: 24px; padding: 30px; box-shadow: 0 10px 25px rgba(0,0,0,0.04); }
    .invoice-header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #EAE0C5; padding-bottom: 15px; margin-bottom: 20px; }
    .invoice-title { font-size: 1.6rem; font-weight: 700; color: #000000 !important; margin: 0; }
    .invoice-subtitle { font-size: 0.95rem; color: #333333 !important; }
    .model-badge { background-color: #E2E8F0; color: #000000 !important; font-family: monospace; padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: bold; }
    .food-item-row { display: flex; align-items: center; justify-content: space-between; padding: 15px 0; border-bottom: 1px dashed #EAE0C5; }
    .food-item-left { display: flex; align-items: center; gap: 15px; }
    .food-item-img-container { position: relative; width: 70px; height: 70px; }
    .food-item-img { width: 100%; height: 100%; object-fit: cover; border-radius: 16px; }
    .food-number-tag { position: absolute; top: -5px; left: -5px; background: #000000; color: white !important; font-size: 0.7rem; font-weight: bold; padding: 2px 6px; border-radius: 8px; }
    .food-title { font-size: 1.05rem; font-weight: 600; color: #000000 !important; }
    .badge-container { display: flex; gap: 8px; align-items: center; margin-top: 4px; }
    .category-badge { font-size: 0.75rem; font-weight: 700; padding: 3px 10px; border-radius: 4px; }
    .accuracy-badge { background-color: #EDF2F7; color: #000000 !important; font-size: 0.75rem; font-weight: 500; padding: 3px 8px; border-radius: 4px; }
    .food-price { font-size: 1.15rem; font-weight: 700; color: #000000 !important; }
    </style>
    """
    st.markdown(app_bg, unsafe_allow_html=True)

    with st.spinner("⏳ Khởi động động cơ AI..."):
        model = init_model()

    st.markdown("<h2 style='text-align: center; color: #000000 !important; font-family: \"Inter\", \"Arial\", sans-serif;'>HỆ THỐNG KIỂM TRA & THANH TOÁN KHAY CƠM TỰ ĐỘNG</h2>", unsafe_allow_html=True)
    st.write("---")

    col_left, col_right = st.columns([1.1, 1.3], gap="large")

    with col_left:
        st.markdown("<div class='step-banner'>📸 BƯỚC 1 & 2: THU THẬP & CĂN LỀ KHAY ĂN</div>", unsafe_allow_html=True)
        camera_file = st.camera_input("Chụp ảnh khay ăn trực tiếp")
        uploaded_file = st.file_uploader("Hoặc tải ảnh lên từ thiết bị", type=["jpg", "jpeg", "png"])

        st.markdown("<p style='color: #000000; font-weight: 600; margin-bottom: 2px; font-family: \"Inter\", \"Arial\", sans-serif;'>Góc xoay hiệu chỉnh tối ưu </p>", unsafe_allow_html=True)
        rotation_mode = st.radio(
            "Góc xoay hiệu chỉnh tối ưu ",
            ("Tự động chỉnh hướng", "Giữ nguyên (0°)", "Xoay 90° CW", "Xoay 90° CCW", "Xoay 180°"),
            key="tray_rotation",
            label_visibility="collapsed"
        )

    active_file = camera_file if camera_file is not None else uploaded_file

    if active_file is not None:
        image = Image.open(active_file).convert('RGB')
        img_array = np.array(image)

        if rotation_mode == "Tự động chỉnh hướng": img_aligned = auto_align_tray(img_array)
        elif rotation_mode == "Xoay 90° CW": img_aligned = cv2.rotate(img_array, cv2.ROTATE_90_CLOCKWISE)
        elif rotation_mode == "Xoay 90° CCW": img_aligned = cv2.rotate(img_array, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotation_mode == "Xoay 180°": img_aligned = cv2.rotate(img_array, cv2.ROTATE_180)
        else: img_aligned = img_array

        with col_left:
            st.image(img_aligned, use_container_width=True, caption="Khay cơm chuẩn hóa ")

        h, w, _ = img_aligned.shape
        regions = {
            "Cơm trắng": img_aligned[int(h*0.02):int(h*0.44), int(w*0.02):int(w*0.54)],
            "Canh": img_aligned[int(h*0.46):int(h*0.98), int(w*0.02):int(w*0.54)],
            "Món rau xào": img_aligned[int(h*0.02):int(h*0.32), int(w*0.56):int(w*0.98)],
            "Món chính": img_aligned[int(h*0.66):int(h*0.98), int(w*0.56):int(w*0.98)],
            "Nước chấm": img_aligned[int(h*0.34):int(h*0.64), int(w*0.56):int(w*0.98)]
        }

        with col_right, st.container(key="invoice_card"):
            st.markdown("<div class='invoice-header'><div><h3 class='invoice-title'>🧾 KẾT QUẢ TÍNH TIỀN</h3><div class='invoice-subtitle'></div></div></div>", unsafe_allow_html=True)

            total_bill = 0
            idx = 1
            for region_name, region_img in regions.items():
                if region_img.shape[0] == 0 or region_img.shape[1] == 0: continue

                img_resized = cv2.resize(region_img, (224, 224))
                img_batch = np.expand_dims(img_resized, axis=0).astype('float32')
                img_batch = preprocess_input(img_batch)

                predictions = model.predict(img_batch, verbose=0)
                predicted_class_idx = np.argmax(predictions[0])
                confidence = np.max(predictions[0]) * 100
                food_name = CLASS_NAMES[predicted_class_idx]

                if food_name == "Khay inox (Trống)": continue

                price = PRICE_MAP.get(food_name, 0)
                total_bill += price
                badge_text, bg_color, text_color = get_food_badge(food_name)

                _, buffer = cv2.imencode('.jpg', cv2.cvtColor(region_img, cv2.COLOR_RGB2BGR))
                img_base64 = base64.b64encode(buffer).decode()

                st.markdown(f"""
                <div class='food-item-row'>
                    <div class='food-item-left'>
                        <div class='food-item-img-container'>
                            <img class='food-item-img' src='data:image/jpeg;base64,{img_base64}'>
                            <div class='food-number-tag'>#{idx}</div>
                        </div>
                        <div class='food-details'>
                            <div class='food-title'>{food_name}</div>
                            <div class='badge-container'>
                                <span class='category-badge' style='background-color: {bg_color}; color: {text_color} !important;'>{badge_text}</span>
                                <span class='accuracy-badge'>Độ tin cậy: {confidence:.0f}%</span>
                            </div>
                        </div>
                    </div>
                    <div class='food-price'>{price:,}đ</div>
                </div>
                """, unsafe_allow_html=True)
                idx += 1

            # =================================================================
            # KHỐI TỔNG CỘNG
            # =================================================================
            st.markdown(
                f"<div style='background-color: #FFFDF6; border: 1px solid #EAE0C5; border-radius: 16px; padding: 20px; display: flex; justify-content: space-between; align-items: center; margin-top: 25px; margin-bottom: 25px;'>"
                f"<span style='font-size: 1.3rem; font-weight: 800; color: #000000 !important; font-family: \"Inter\", \"Arial\", sans-serif;'>TỔNG CỘNG:</span>"
                f"<span style='font-size: 2.3rem; font-weight: 900; color: #000000 !important; font-family: \"Inter\", \"Arial\", sans-serif;'>{total_bill:,}đ</span>"
                f"</div>", 
                unsafe_allow_html=True
            )

            # =================================================================
            # PHƯƠNG THỨC THANH TOÁN
            # =================================================================
            st.markdown("<p style='font-weight: 700; font-size: 1.05rem; margin-bottom: 6px; color: #000000 !important; font-family: \"Inter\", \"Arial\", sans-serif;'>💳 PHƯƠNG THỨC THANH TOÁN</p>", unsafe_allow_html=True)
            
            pay_option = st.radio(
                "Chọn hình thức thanh toán:",
                options=["💵 Tiền mặt", "📱 Quét mã QR", "🪪 Thẻ SV RFID"],
                key="payment_checkout",
                label_visibility="collapsed"
            )

            # Xử lý hiển thị mã QR từ file local 'my_qr.png'
            if pay_option == "📱 Quét mã QR" and total_bill > 0:
                qr_base64 = get_base64_encoded_image("my_qr.png")
                if qr_base64:
                    st.markdown(f"""
                    <div class="qr-container">
                        <img src="data:image/png;base64,{qr_base64}" width="160" style="border-radius: 8px; box-shadow: 0 4px 10px rgba(0,0,0,0.08);"/>
                        <div class="qr-text">Vui lòng quét mã để thanh toán {total_bill:,}đ</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("⚠️ Vui lòng thêm file ảnh 'my_qr.png' vào thư mục mã nguồn để hiển thị mã QR thanh toán.")

            st.write("")
            if st.button("XÁC NHẬN HOÀN TẤT HÓA ĐƠN", key="btn_confirm_invoice"):
                st.toast(f"🎉 Đã thanh toán {total_bill:,}đ qua hình thức **{pay_option}**!", icon="✅")
            st.markdown("</div>", unsafe_allow_html=True)


# ==========================================
# TRANG 3: GÓC ẨM THỰC AI (DASHBOARD)
# ==========================================
with tabs[2]:
    # KHAI BÁO BIẾN TRÁNH LỖI NAMERROR 
    dashboard_css = """
    <style>
    .kpi-card { background: #FFFFFF; border-radius: 20px; padding: 20px; box-shadow: 0 4px 15px rgba(165,145,120,0.05); border: 1px solid #F3EFE6; display: flex; justify-content: space-between; align-items: center; font-family: 'Inter', 'Arial', sans-serif; }
    .kpi-title { font-size: 0.75rem; font-weight: 700; color: #555555 !important; text-transform: uppercase; }
    .kpi-value { font-size: 1.8rem; font-weight: 700; color: #000000 !important; margin-top: 5px; }
    .kpi-sub { font-size: 0.8rem; color: #10B981 !important; font-weight: 600; margin-top: 5px; }
    .kpi-icon { width: 45px; height: 45px; border-radius: 12px; display: flex; align-items: center; justify-content: center; background: #EFF6F0; }
    .dashboard-card { background: #FFFFFF; border-radius: 24px; padding: 25px; box-shadow: 0 4px 15px rgba(165,145,120,0.05); border: 1px solid #F3EFE6; height: 100%; font-family: 'Inter', 'Arial', sans-serif; }
    .card-title { font-size: 1.25rem; font-weight: 700; color: #000000 !important; }
    .food-rank-row { display: flex; align-items: center; justify-content: space-between; padding: 12px; background: #FDFCF7; border-radius: 16px; margin-bottom: 10px; border: 1px solid #F7F4EC; }
    .rank-number { background: #5D6B54; color: white !important; font-weight: 700; width: 30px; height: 30px; border-radius: 8px; display: flex; align-items: center; justify-content: center; margin-right: 12px; }
    .select-count-tag { background: #E6ECE6; color: #000000 !important; font-weight: 600; font-size: 0.85rem; padding: 6px 12px; border-radius: 8px; }
    .quick-sell-price-box { border: 1px solid #F3EFE6; background: #FAFAFA; border-radius: 16px; padding: 20px; display: flex; justify-content: space-between; align-items: center; margin-top: 15px; margin-bottom: 15px; }
    .qr-quick-container { display: flex; align-items: center; gap: 15px; background-color: #FAFAFA; border: 1px dashed #CCCCCC; border-radius: 12px; padding: 12px; margin-top: 10px; }
    </style>
    """
    st.markdown(dashboard_css, unsafe_allow_html=True)

    # 4 THÈ KPI HÀNG ĐẦU
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown('<div class="kpi-card"><div><div class="kpi-title">Tổng doanh thu quầy</div><div class="kpi-value">374.500đ</div><div class="kpi-sub">↗ +12.4% so với hôm qua</div></div><div class="kpi-icon">🪙</div></div>', unsafe_allow_html=True)
    with col2: st.markdown('<div class="kpi-card"><div><div class="kpi-title">Tổng số khay bán ra</div><div class="kpi-value">13 khay</div><div style="font-size:0.8rem; color:#555555 !important; margin-top:5px;">Kiểm soát tự động</div></div><div class="kpi-icon" style="background:#FDF5E6;">🥞</div></div>', unsafe_allow_html=True)
    with col3: st.markdown('<div class="kpi-card"><div><div class="kpi-title">Giá khay trung bình</div><div class="kpi-value">28.808đ</div><div style="font-size:0.8rem; color:#555555 !important; margin-top:5px;">Thực đơn quầy</div></div><div class="kpi-icon" style="background:#F0F7FF;">📈</div></div>', unsafe_allow_html=True)
    with col4: st.markdown('<div class="kpi-card"><div><div class="kpi-title">Cổng camera giám sát</div><div class="kpi-value" style="font-size:1.3rem; margin-top:8px; color:#2E7D32 !important;">CAM–01 ACTIVE</div><div class="kpi-sub">🟢 Kết nối ổn định</div></div><div class="kpi-icon" style="background:#E8F5E9;">📷</div></div>', unsafe_allow_html=True)

    st.write("")
    col_chart_left, col_chart_right = st.columns([1.7, 1.1])
    with col_chart_left:
        st.markdown('<div class="dashboard-card"><div class="card-title">Xu Hướng Doanh Thu 7 Ngày Qua</div><div style="font-size:0.85rem; color:#555555 !important; margin-bottom:15px;">Dữ liệu cập nhật thời gian thực</div>', unsafe_allow_html=True)
        chart_data = {"Ngày": ["03/06", "04/06", "05/06", "06/06", "07/06", "08/06", "09/06"], "Doanh thu ngày (đ)": [50000, 62000, 72000, 48000, 55000, 82000, 35000]}
        st.area_chart(data=chart_data, x="Ngày", y="Doanh thu ngày (đ)", color="#5D6B54")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_chart_right:
        st.markdown('<div class="dashboard-card"><div class="card-title">Phương thức thanh toán</div><div style="font-size:0.85rem; color:#555555 !important; margin-bottom:15px;">Tỷ lệ sử dụng dòng tiền</div>', unsafe_allow_html=True)
        pay_methods = [{"name": "🟢 Thẻ Sinh Viên", "val": "179.000đ"}, {"name": "🔴 Ví MoMo", "val": "138.500đ"}, {"name": "🔵 QR Ngân Hàng", "val": "27.000đ"}, {"name": "🟡 Tiền Mặt", "val": "30.000đ"}]
        for m in pay_methods:
            st.markdown(f'<div style="display: flex; justify-content: space-between; font-size: 0.95rem; margin-bottom: 12px; border-bottom: 1px dashed #F3EFE6; padding-bottom: 4px;"><span style="color:#000000 !important;">{m["name"]}</span><span style="font-weight: 700; color:#000000 !important;">{m["val"]}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    col_bottom_left, col_bottom_right = st.columns([1.4, 1.4])
    with col_bottom_left:
        st.markdown('<div class="dashboard-card"><div class="card-title">Xu hướng đĩa ăn bán chạy</div><div style="font-size:0.85rem; color:#555555 !important; margin-bottom:15px;">Báo cáo top món ăn sinh viên yêu thích nhất</div>', unsafe_allow_html=True)
        top_foods = [{"rank": "#1", "name": "Sườn xào chua ngọt", "rev": "75.000đ", "count": "3 lượt"}, {"rank": "#2", "name": "Rau muống xào tỏi", "rev": "15.000đ", "count": "3 lượt"}, {"rank": "#3", "name": "Canh bắp cải", "rev": "15.000đ", "count": "3 lượt"}]
        for food in top_foods:
            st.markdown(f'<div class="food-rank-row"><div style="display: flex; align-items: center;"><div class="rank-number">{food["rank"]}</div><div><div class="food-info-name" style="color:#000000 !important;">{food["name"]}</div><div class="food-info-revenue" style="color:#333333 !important;">Tích lũy: {food["rev"]}</div></div></div><div class="select-count-tag">{food["count"]}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_bottom_right:
        st.markdown('<div class="dashboard-card"><div class="card-title">Bán cơm nhanh tại quầy</div><div style="font-size:0.85rem; color:#555555 !important; margin-bottom:15px;">Ghi nhận đơn cơm nhanh không qua Camera</div>', unsafe_allow_html=True)
        st.markdown('<div class="quick-sell-price-box"><div><div style="font-size: 0.8rem; font-weight: 700; color: #555555 !important; text-transform: uppercase;">Giá khay cơm Kiosk hiện tại:</div><div style="font-size: 1.6rem; font-weight: 800; color: #000000 !important;">35.000đ</div></div></div>', unsafe_allow_html=True)
        st.markdown("<p style='font-weight: 600; margin-bottom: 5px; color:#000000 !important; font-family: \"Inter\", \"Arial\", sans-serif;'>Hình thức thanh toán tại quầy:</p>", unsafe_allow_html=True)
        payment_choice = st.radio(
            "Chọn hình thức thanh toán nhanh:",
            options=["💵 Tiền mặt", "💳 Quẹt thẻ (POS / Ngân hàng)", "📱 Quét mã QR Kiosk"],
            key="quick_payment_choice",
            label_visibility="collapsed"
        )
        
        # Xử lý QR cho phần bán nhanh tại quầy từ file local
        if payment_choice == "📱 Quét mã QR Kiosk":
            quick_qr_base64 = get_base64_encoded_image("my_qr.png")
            if quick_qr_base64:
                st.markdown(f"""
                <div class="qr-quick-container">
                    <img src="data:image/png;base64,{quick_qr_base64}" width="90" style="border-radius: 6px;"/>
                    <div style="font-size: 0.85rem; font-weight: 600; color: #333333 !important;">Quét mã cá nhân<br/>để nhận tiền đơn 35.000đ</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ Chưa có file 'my_qr.png'.")
            
        st.write("")
        if st.button("XÁC NHẬN BÁN NHANH", key="btn_quick_sell"):
            st.toast(f"🎉 Ghi nhận hóa đơn quầy 35.000đ thành công bằng hình thức **{payment_choice}**!", icon="💰")
        st.markdown('<div style="font-size: 0.8rem; color: #555555 !important; font-style: italic; margin-top: 15px;">* Hệ thống tự động ghi nhận món và xuất kết toán hóa đơn ra máy in quầy.</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
