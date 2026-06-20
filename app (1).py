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

# TỐI ƯU GIAO DIỆN: Nền vàng nhẹ, toàn bộ tiêu đề & chữ màu đen, nút chọn nhỏ gọn
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

/* KHỐI MENU NGANG VÀ CÁC TIÊU ĐỀ ĐỔI THÀNH MÀU ĐEN */
div[data-testid="stTabs"] button {
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    padding: 12px 24px !important;
    color: #000000 !important; /* Màu chữ đen cho menu */
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    border-bottom: 3px solid #000000 !important;
}

h1, h2, h3, h4, h5, h6 {
    color: #000000 !important; /* Toàn bộ tiêu đề thành màu đen */
}

/* ĐỊNH DẠNG CÁC Ô CHỌN THANH TOÁN THÀNH NÚT HÌNH VUÔNG NHỎ GỌN */
div[data-testid="stRadio"] > div {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: wrap !important;
    gap: 10px !important;
    padding-top: 5px;
}
div[data-testid="stRadio"] label {
    background-color: #FFFFFF !important;
    border: 2px solid #D6C7A1 !important;
    border-radius: 8px !important; /* Bo góc nhỏ hơn trông vuông vắn hơn */
    padding: 8px 16px !important;  /* Thu nhỏ padding để nút gọn gàng */
    width: auto !important;        /* Không co giãn full chiều rộng */
    min-width: 110px !important;   /* Độ rộng tối thiểu vừa vặn chữ */
    text-align: center !important;
    cursor: pointer !important;
    transition: all 0.2s ease-in-out !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.04) !important;
}
/* Hiệu ứng khi hover vào ô chọn */
div[data-testid="stRadio"] label:hover {
    border-color: #000000 !important; 
    background-color: #FFFDF9 !important;
}
/* Style khi ô vuông nhỏ được click chọn */
div[data-testid="stRadio"] [data-checked="true"] ~ label {
    border-color: #000000 !important; /* Viền đen nổi bật */
    background-color: #EAEAEA !important; /* Nền xám nhạt tinh tế */
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15) !important;
}
div[data-testid="stRadio"] [data-checked="true"] ~ label p {
    font-weight: 700 !important;
    color: #000000 !important; /* Chữ đen khi chọn */
}
/* Ẩn hoàn toàn dấu chấm tròn radio mặc định */
div[data-testid="stRadio"] input[type="radio"] {
    display: none !important;
}
div[data-testid="stRadio"] div[data-testid="stMarkdownContainer"] p {
    margin: 0 !important;
    font-size: 0.95rem !important;
    color: #000000 !important;
}

/* ÉP TẤT CẢ CÁC ĐOẠN TEXT CHÍNH TRONG HÓA ĐƠN THÀNH MÀU ĐEN */
.food-title, .food-price, .invoice-title, .invoice-subtitle {
    color: #000000 !important;
}
</style>
"""
st.markdown(custom_ui_style, unsafe_allow_html=True)

# Khởi tạo Menu Ngang dạng Tabs
tabs = st.tabs(["🏠 Trang Chủ (Giới thiệu)", "📷 Hệ Thống Nhận Diện", "📊 Góc Ẩm Thực AI"])

# Bảng giá và Danh mục món ăn
PRICE_MAP = {
    "Cơm trắng": 10000, "Trứng chiên": 25000, "Khay inox (Trống)": 0,
    "Đậu hũ sốt cà": 25000, "Cá hú kho": 30000, "Thịt kho trứng": 30000,
    "Thịt kho": 25000, "Canh chua": 25000, "Sườn nướng": 30000,
    "Canh rau": 7000, "Rau xào": 10000
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
        if np.sum(np.abs(cv2.Sobel(top_half, cv2.CV_64F, 1, 0, ksize=3))) < np.sum(np.abs(cv2.Sobel(bottom_half, cv2.CV_64F, 1, 0, ksize=3))):
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
    <div style='font-family: sans-serif; background-image: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.85)), url("https://images.unsplash.com/photo-1544025162-d76694265947?ixlib=rb-4.0.3&auto=format&fit=crop&w=1920&q=80"); background-size: cover; background-position: center; padding: 60px; border-radius: 24px; margin-top: 10px;'>
        <h1 style='font-size: 3.5rem; font-weight: 800; color: #F39C12 !important; text-align: center; text-shadow: 2px 2px 4px rgba(0,0,0,0.8);'>CANTEEN AI SYSTEM</h1>
        <p style='font-size: 1.4rem; color: white !important; text-align: center; margin-bottom: 40px; text-shadow: 1px 1px 2px rgba(0,0,0,0.8);'>Giải pháp nhận diện khay cơm và thanh toán tự động bằng công nghệ Computer Vision</p>
    </div>
    """
    st.markdown(marketing_bg, unsafe_allow_html=True)
    st.write("")
    col1, col2, col3 = st.columns(3)
    with col1: st.info("🎯 **Chính xác cao**\n\nNhận diện nhanh chóng từng món ăn.")
    with col2: st.success("⚡ **Tốc độ chớp nhoáng**\n\nMọi thứ hoàn tất trong vài giây.")
    with col3: st.warning("📊 **Quản lý dễ dàng**\n\nHệ thống xuất hóa đơn tự động minh bạch.")

# ==========================================
# TRANG 2: HỆ THỐNG XỬ LÝ CHÍNH
# ==========================================
with tabs[1]:
    with st.spinner("⏳ Khởi động động cơ AI..."):
        model = init_model()

    st.markdown("<h2 style='text-align: center;'>HỆ THỐNG KIỂM TRA & THANH TOÁN KHAY CƠM TỰ ĐỘNG</h2>", unsafe_allow_html=True)
    st.write("---")

    col_left, col_right = st.columns([1.1, 1.3], gap="large")

    with col_left:
        # BỎ BANNER HÌNH CHỮ NHẬT - Chỉ giữ lại widget upload/camera trực tiếp
        camera_file = st.camera_input("Chụp ảnh khay ăn trực tiếp")
        uploaded_file = st.file_uploader("Hoặc tải ảnh lên từ thiết bị", type=["jpg", "jpeg", "png"])
        st.markdown("<p style='color: #000000; font-weight: 600; margin-bottom: 2px;'>Góc xoay hiệu chỉnh tối ưu từ AI Co-pilot:</p>", unsafe_allow_html=True)
        rotation_mode = st.radio(
            "Xoay khay:",
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
            st.markdown("<div style='background-color: #FFFFFF; border: 1px solid #E0D4B7; border-radius: 24px; padding: 30px; box-shadow: 0 10px 25px rgba(0,0,0,0.04);'>", unsafe_allow_html=True)
            st.markdown("<div style='display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #EAE0C5; padding-bottom: 15px; margin-bottom: 20px;'><div><h3 class='invoice-title'>🧾 KẾT QUẢ TÍNH TIỀN</h3><div class='invoice-subtitle'>AI kết xuất hóa đơn tự động</div></div><div style='background-color: #E2E8F0; color: #000000 !important; font-family: monospace; padding: 4px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: bold;'>EfficientNet Core V2</div></div>", unsafe_allow_html=True)

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

            # CHỮ TỔNG CỘNG MÀU ĐEN TUYỆT ĐỐI
            st.markdown(f"""
            <div style='background-color: #FFFDF6; border: 1px solid #EAE0C5; border-radius: 16px; padding: 20px; display: flex; justify-content: space-between; align-items: center; margin-top: 25px; margin-bottom: 25px;'>
                <span style='font-size: 1.1rem; font-weight: bold; color: #000000 !important;'>TỔNG CỘNG:</span>
                <span style='font-size: 2.3rem; font-weight: 800; color: #000000 !important;'>{total_bill:,}đ</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<p style='font-weight: 700; margin-bottom: 2px; color:#000000 !important;'>💳 PHƯƠNG THỨC THANH TOÁN</p>", unsafe_allow_html=True)
            pay_option = st.radio(
                "Chọn hình thức thanh toán:",
                options=["💵 Tiền mặt", "📱 Quét mã QR", "🪪 Thẻ SV RFID"],
                key="payment_checkout",
                label_visibility="collapsed"
            )

            st.write("")
            if st.button("XÁC NHẬN HOÀN TẤT HÓA ĐƠN", key="btn_confirm_invoice"):
                st.toast(f"🎉 Đã thanh toán {total_bill:,}đ qua hình thức **{pay_option}**!", icon="✅")
            st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# TRANG 3: GÓC ẨM THỰC AI (DASHBOARD)
# ==========================================
with tabs[2]:
    dashboard_css = """
    <style>
    .kpi-card { background: #FFFFFF; border-radius: 20px; padding: 20px; box-shadow: 0 4px 15px rgba(165,145,120,0.05); border: 1px solid #F3EFE6; display: flex; justify-content: space-between; align-items: center; }
    .kpi-title { font-size: 0.75rem; font-weight: 700; color: #000000 !important; text-transform: uppercase; }
    .kpi-value { font-size: 1.8rem; font-weight: 700; color: #000000 !important; margin-top: 5px; }
    .kpi-sub { font-size: 0.8rem; color: #10B981 !important; font-weight: 600; margin-top: 5px; }
    .dashboard-card { background: #FFFFFF; border-radius: 24px; padding: 25px; box-shadow: 0 4px 15px rgba(165,145,120,0.05); border: 1px solid #F3EFE6; height: 100%; }
    .food-rank-row { display: flex; align-items: center; justify-content: space-between; padding: 12px; background: #FDFCF7; border-radius: 16px; margin-bottom: 10px; border: 1px solid #F7F4EC; }
    </style>
    """
    st.markdown(dashboard_css, unsafe_allow_html=True)

    # 4 THẺ KPI HÀNG ĐẦU
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.markdown('<div class="kpi-card"><div><div class="kpi-title">Tổng doanh thu quầy</div><div class="kpi-value">374.500đ</div><div class="kpi-sub">↗ +12.4% so với hôm qua</div></div><div>🪙</div></div>', unsafe_allow_html=True)
    with col2: st.markdown('<div class="kpi-card"><div><div class="kpi-title">Tổng số khay bán ra</div><div class="kpi-value">13 khay</div></div><div style="background:#FDF5E6;">🥞</div></div>', unsafe_allow_html=True)
    with col3: st.markdown('<div class="kpi-card"><div><div class="kpi-title">Giá khay trung bình</div><div class="kpi-value">28.808đ</div></div><div>📈</div></div>', unsafe_allow_html=True)
    with col4: st.markdown('<div class="kpi-card"><div><div class="kpi-title">Cổng camera giám sát</div><div class="kpi-value" style="color:#2E7D32 !important;">CAM–01 ACTIVE</div></div><div>📷</div></div>', unsafe_allow_html=True)

    st.write("")
    col_chart_left, col_chart_right = st.columns([1.7, 1.1])
    with col_chart_left:
        st.markdown('<div class="dashboard-card"><h3>Xu Hướng Doanh Thu 7 Ngày Qua</h3>', unsafe_allow_html=True)
        chart_data = {"Ngày": ["03/06", "04/06", "05/06", "06/06", "07/06", "08/06", "09/06"], "Doanh thu ngày (đ)": [50000, 62000, 72000, 48000, 55000, 82000, 35000]}
        st.area_chart(data=chart_data, x="Ngày", y="Doanh thu ngày (đ)", color="#5D6B54")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_chart_right:
        st.markdown('<div class="dashboard-card"><h3>Phương thức thanh toán</h3>', unsafe_allow_html=True)
        pay_methods = [{"name": "🟢 Thẻ Sinh Viên", "val": "179.000đ"}, {"name": "🔴 Ví MoMo", "val": "138.500đ"}, {"name": "🔵 QR Ngân Hàng", "val": "27.000đ"}, {"name": "🟡 Tiền Mặt", "val": "30.000đ"}]
        for m in pay_methods:
            st.markdown(f'<div style="display: flex; justify-content: space-between; font-size: 0.95rem; margin-bottom: 12px; border-bottom: 1px dashed #F3EFE6; padding-bottom: 4px;"><span style="color:#000000 !important;">{m["name"]}</span><span style="font-weight: 700; color:#000000 !important;">{m["val"]}</span></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.write("")
    col_bottom_left, col_bottom_right = st.columns([1.4, 1.4])
    with col_bottom_left:
        st.markdown('<div class="dashboard-card"><h3>Xu hướng đĩa ăn bán chạy</h3>', unsafe_allow_html=True)
        top_foods = [{"rank": "#1", "name": "Sườn xào chua ngọt", "rev": "75.000đ", "count": "3 lượt"}, {"rank": "#2", "name": "Rau muống xào tỏi", "rev": "15.000đ", "count": "3 lượt"}, {"rank": "#3", "name": "Canh bắp cải", "rev": "15.000đ", "count": "3 lượt"}]
        for food in top_foods:
            st.markdown(f'<div class="food-rank-row"><div style="display: flex; align-items: center;"><div style="background:#5D6B54; color:white; font-weight:700; width:30px; height:30px; border-radius:8px; display:flex; align-items:center; justify-content:center; margin-right:12px;">{food["rank"]}</div><div><div style="color:#000000 !important; font-weight:600;">{food["name"]}</div><div style="color:#333333 !important; font-size:0.85rem;">Tích lũy: {food["rev"]}</div></div></div><div style="background:#E6ECE6; color:#000000 !important; font-weight:600; font-size:0.85rem; padding:6px 12px; border-radius:8px;">{food["count"]}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_bottom_right:
        st.markdown('<div class="dashboard-card"><h3>Bán cơm nhanh tại quầy</h3>', unsafe_allow_html=True)
        st.markdown('<div style="border: 1px solid #F3EFE6; background: #FAFAFA; border-radius: 16px; padding: 20px; margin-top: 15px; margin-bottom: 15px;">Ghi nhận đơn cơm nhanh: <b style="font-size:1.5rem; color:#000000;">35.000đ</b></div>', unsafe_allow_html=True)
        
        st.markdown("<p style='font-weight: 600; margin-bottom: 5px; color:#000000 !important;'>Hình thức thanh toán tại quầy:</p>", unsafe_allow_html=True)
        payment_choice = st.radio(
            "Chọn hình thức thanh toán nhanh:",
            options=["💵 Tiền mặt", "💳 Quẹt thẻ (POS)"],
            key="quick_payment_choice",
            label_visibility="collapsed"
        )
        
        st.write("")
        if st.button("XÁC NHẬN BÁN NHANH", key="btn_quick_sell"):
            st.toast(f"🎉 Ghi nhận hóa đơn quầy 35.000đ qua **{payment_choice}**!", icon="💰")
        st.markdown('</div>', unsafe_allow_html=True)
