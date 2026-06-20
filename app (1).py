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
    # Trả về Tên badge hiển thị bằng tiếng Việt chữ đen để đồng bộ yêu cầu
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
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;800&display=swap');
    .marketing-wrapper {
        font-family: 'Montserrat', sans-serif;
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
    .main-body-style { font-family: 'Inter', sans-serif; }
    .step-banner { background: #586F56; color: white !important; padding: 12px 20px; border-radius: 8px; font-weight: 600; font-size: 1.1rem; margin-bottom: 15px; margin-top: 10px; }
    .canteen-invoice-card { background-color: #FFFFFF; border: 1px solid #E0D4B7; border-radius: 24px; padding: 30px; box-shadow: 0 10px 25px rgba(0,0,0,0.04); }
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

    st.markdown("<h2 style='text-align: center; color: #000000 !important;'>HỆ THỐNG KIỂM TRA & THANH TOÁN KHAY CƠM TỰ ĐỘNG</h2>", unsafe_allow_html=True)
    st.write("---")

    col_left, col_right = st.columns([1.1, 1.3], gap="large")

    with col_left:
        st.markdown("<div class='step-banner'>📸 BƯỚC 1 & 2: THU THẬP & CĂN LỀ KHAY ĂN</div>", unsafe_allow_html=True)
        camera_file = st.camera_input("Chụp ảnh khay ăn trực tiếp")
        uploaded_file = st.file_uploader("Hoặc tải ảnh lên từ thiết bị", type=["jpg", "jpeg", "png"])
        st.markdown("<p style='color: #000000; font-weight: 600; margin-bottom: 2px;'>Góc xoay hiệu chỉnh tối ưu từ AI Co-pilot:</p>", unsafe_allow_html=True)
        rotation_mode = st.radio(
            "Góc xoay hiệu chỉnh tối ưu từ AI Co-pilot:",
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
            st.image(img_aligned, use_container_width=True, caption="Khay cơm chuẩn hóa đưa vào AI core")

        h, w, _ = img_aligned.shape
        regions = {
            "Cơm trắng": img_aligned[int(h*0.02):int(h*0.44), int(w*0.02):int(w*0.54)],
            "Canh": img_aligned[int(h*0.46):int(h*0.98), int(w*0.02):int(w*0.54)],
            "Món rau xào": img_aligned[int(h*0.02):int(h*0.32), int(w*0.56):int(w*0.98)],
            "Món chính": img_aligned[int(h*0.66):int(h*0.98), int(w*0.56):int(w*0.98)],
            "Nước chấm": img_aligned[int(h*0.34):int(h*0.64), int(w*0.56):int(w*0.98)]
        }

        with col_right:
            st.markdown("<div class='canteen-invoice-card'>", unsafe_allow_html=True)
            st.markdown("<div class='invoice-header'><div><h3 class='invoice-title'>🧾 KẾT QUẢ TÍNH TIỀN</h3><div class='invoice-subtitle'>AI kết xuất hóa đơn tự động</div></div><div class='model-badge'>EfficientNet Core V2</div></div>", unsafe_allow_html=True)

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

            # ĐỔI CHỮ TỔNG CỘNG THÀNH MÀU ĐEN TUYỆT ĐỐI KHÔNG BỊ Ô ĐEN CHE
            st.markdown(f"""
            <div style='background-color: #FFFDF6; border: 1px solid #EAE0C5; border-radius: 16px; padding: 20px; display: flex; justify-content: space-between; align-items: center; margin-top: 25px; margin-bottom: 25px;'>
                <span style='font-size: 1.1rem; font-weight: bold; color: #000000 !important;'>TỔNG CỘNG:</span>
                <span style='font-size: 2.3rem; font-weight: 800; color: #000000 !important;'>{total_bill:,}đ</span>
            </div>
            """, unsafe_allow_html=True)

            # Ô VUÔNG PHƯƠNG THỨC THANH TOÁN ĐÃ ĐƯỢC LÀM NỔI BẬT VÀ ẨN NÚT TRÒN
            st.markdown("<p style='font-weight: 700; margin-bottom: 2px; color:#000000 !important;'>💳 PHƯƠNG THỨC THANH TOÁN</p>", unsafe_allow_html=True)
            pay_option = st.radio(
                "Chọn hình thức thanh toán:",
                options=["💵 Tiền mặt", "📱 Quét mã QR", "🪪 Thẻ SV RFID"],
                key="payment_checkout",
                label_visibility="collapsed"
            )

            st.write("")
            if pay_option == "💵 Tiền mặt": st.success("💰 **Hệ thống sẵn sàng:** Vui lòng nhận tiền mặt và xác nhận hoàn tất đơn.")
            elif pay_option == "📱 Quét mã QR": st.info("📲 **Quét mã nhanh:** Hệ thống đã đồng bộ cổng QR động thanh toán.")
            elif pay_option == "🪪 Thẻ SV RFID": st.warning("💳 **Chờ quẹt thẻ:** Vui lòng áp thẻ sinh viên vào đầu đọc RFID.")

            st.write("")
            if st.button("XÁC NHẬN HOÀN TẤT HÓA ĐƠN", key="btn_confirm_invoice"):
                st.toast(f"🎉 Đã thanh toán {total_bill:,}đ thành công qua hình thức **{pay_option}**!", icon="✅")
            st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# TRANG 3: GÓC ẨM THỰC AI (DASHBOARD)
# ==========================================
with tabs[2]:
    dashboard_css = """
    <style>
    .kpi-card { background: #FFFFFF; border-radius: 20px; padding: 20px; box-shadow: 0 4px 15px rgba(165,145,120,0.05); border: 1px solid #F3EFE6; display: flex; justify-content: space-between; align-items: center; }
    .kpi-title { font-size: 0.75rem; font-weight: 700; color: #555555 !important; text-transform: uppercase; }
    .kpi-value { font-size: 1.8rem; font-weight: 700; color: #000000 !important; margin-top: 5px; }
    .kpi-sub { font-size: 0.8rem; color: #10B981 !important; font-weight: 600; margin-top: 5px; }
    .kpi-icon { width: 45px; height: 45px; border-radius: 12px; display: flex; align-items: center; justify-content: center; background: #EFF6F0; }
    .dashboard-card { background: #FFFFFF; border-radius: 24px; padding: 25px; box-shadow: 0 4px 15px rgba(165,145,120,0.05); border: 1px solid #F3EFE6; height: 100%; }
    .card-title { font-size: 1.25rem; font-weight: 700; color: #000000 !important; }
    .food-rank-row
