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

        html_items = ""

        ai_console_logs = []

        item_counter = 1



        for region_name, region_img in regions.items():

            if region_img.shape[0] == 0 or region_img.shape[1] == 0:

                continue



            img_resized = cv2.resize(region_img, (224, 224))

            img_batch = np.expand_dims(img_resized, axis=0).astype('float32')

            img_batch = preprocess_input(img_batch)



            predictions = model.predict(img_batch, verbose=0)

            predicted_class_idx = np.argmax(predictions[0])

            confidence = np.max(predictions[0]) * 100



            food_name = CLASS_NAMES[predicted_class_idx]

            price = PRICE_MAP[food_name]



            if food_name == "Khay inox (Trống)" or region_name == "Nước chấm":

                if food_name != "Khay inox (Trống)":

                    ai_console_logs.append(f"⚡ [Detector]: Bỏ qua khu vực phụ '{region_name}' ({food_name}).")

                continue



            total_bill += price

            badge_text, bg_color, text_color = get_food_badge(food_name)



            import base64

            from io import BytesIO

            pil_region = Image.fromarray(region_img)

            buffered = BytesIO()

            pil_region.save(buffered, format="JPEG")

            img_str = base64.b64encode(buffered.getvalue()).decode()



            html_items += f"""

            <div class='food-item-row'>

                <div class='food-item-left'>

                    <div class='food-item-img-container'>

                        <img src='data:image/jpeg;base64,{img_str}' class='food-item-img' />

                        <div class='food-number-tag'>#{item_counter}</div>

                    </div>

                    <div class='food-details'>

                        <div class='food-title'>{food_name}</div>

                        <div class='badge-container'>

                            <span class='category-badge' style='background-color: {bg_color}; color: {text_color} !important;'>{badge_text}</span>

                            <span class='accuracy-badge'>Acc: {confidence:.0f}%</span>

                        </div>

                    </div>

                </div>

                <div class='food-price'>{price:,}đ</div>

            </div>

            """

            ai_console_logs.append(f"✔ [CNN Core]: Nhận diện thành công '{food_name}' tại {region_name} - Acc: {confidence:.2f}%")

            item_counter += 1



        with col_right:

            # Tạo khối giao diện 3 phương thức thanh toán mới

            invoice_html = f"""

            <div class='step-banner'>&nbsp;🧾 BƯỚC 3: HÓA ĐƠN ĐIỆN TỬ &amp; THANH TOÁN</div>

            <div class='canteen-invoice-card'>

                <div class='invoice-header'>

                    <div>

                        <h3 class='invoice-title'>Hóa Đơn Tổng Canteen</h3>

                        <div class='invoice-subtitle'>Vé ăn chiết khấu học đường trường đại học</div>

                    </div>

                    <div style='text-align: right; display: flex; flex-direction: column; align-items: flex-end; gap: 8px;'>

                        <span class='receipt-btn'>📄 BIÊN LAI</span>

                        <span class='model-badge'>MODEL: SIMULATION CNN (EFFICIENTNET)</span>

                    </div>

                </div>

                {html_items}

                <div class='total-box'>

                    <span class='total-label'>TỔNG CỘNG:</span>

                    <span class='total-price-value'>{total_bill:,}đ</span>

                </div>

                <div class='payment-title'>HÌNH THỨC CHIẾT KHẤU THẺ / SMART-QR</div>

                <div class='payment-options-grid'>

                    <div class='payment-choice-box pay-selected'>Thẻ SV RFID</div>

                    <div class='payment-choice-box'>Tiền mặt</div>

                    <div class='payment-choice-box'>Quét mã</div>

                </div>

            </div>

            """

            st.markdown(invoice_html, unsafe_allow_html=True)



            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("🛒 XÁC NHẬN & THANH TOÁN NGAY", use_container_width=True, type="primary"):

                st.toast("🎉 Giao dịch thành công!")

                st.balloons()



            st.markdown("<br>", unsafe_allow_html=True)

            console_content = "".join([f"<div style='margin-bottom: 4px; color: #8BE9FD;'>{log}</div>" for log in ai_console_logs])

            st.markdown(f"<div class='terminal-console'>{console_content}</div>", unsafe_allow_html=True)



# ------------------------------------------

# TRANG 3: GÓC ẨM THỰC AI

# ------------------------------------------

# ------------------------------------------

# TRANG 3: THỐNG KÊ DOANH THU & BÁN NHANH (Thay cho Góc Ẩm Thực AI)

# ------------------------------------------

elif page == "Góc Ẩm Thực AI":

    dashboard_css = """

    <style>

    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"], .stApp {

        font-family: 'Inter', sans-serif;

        background-color: #FDF9F0 !important;

    }

    

    /* Toàn bộ chữ mặc định thành màu tối của giao diện */

    p, span, h1, h2, h3, h4, div { color: #4A3E3D !important; }



    /* CSS cho 4 thẻ KPI hàng đầu */

    .kpi-container {

        display: flex; gap: 20px; margin-bottom: 25px;

    }

    .kpi-card {

        flex: 1; background: #FFFFFF; border-radius: 20px; padding: 20px;

        box-shadow: 0 4px 15px rgba(165, 145, 120, 0.05);

        border: 1px solid #F3EFE6; display: flex; justify-content: space-between; align-items: center;

    }

    .kpi-title { font-size: 0.75rem; font-weight: 700; color: #A59E92 !important; text-transform: uppercase; letter-spacing: 0.5px; }

    .kpi-value { font-size: 1.8rem; font-weight: 700; color: #2D392E !important; margin-top: 5px; }

    .kpi-sub { font-size: 0.8rem; color: #10B981 !important; font-weight: 600; margin-top: 5px; }

    .kpi-sub-gray { font-size: 0.8rem; color: #A59E92 !important; margin-top: 5px; }

    .kpi-icon { width: 45px; height: 45px; border-radius: 12px; display: flex; align-items: center; justify-content: center; }



    /* Nội dung các Card lớn */

    .dashboard-card {

        background: #FFFFFF; border-radius: 24px; padding: 25px;

        box-shadow: 0 4px 15px rgba(165, 145, 120, 0.05); border: 1px solid #F3EFE6; height: 100%;

    }

    .card-title { font-size: 1.3rem; font-weight: 700; color: #2D392E !important; margin-bottom: 5px; }

    .card-subtitle { font-size: 0.85rem; color: #A59E92 !important; margin-bottom: 20px; }



    /* CSS Hàng món ăn bán chạy */

    .food-rank-row {

        display: flex; align-items: center; justify-content: space-between;

        padding: 15px; background: #FDFCF7; border-radius: 16px; margin-bottom: 12px; border: 1px solid #F7F4EC;

    }

    .rank-number {

        background: #5D6B54; color: white !important; font-weight: 700; width: 32px; height: 32px;

        border-radius: 8px; display: flex; align-items: center; justify-content: center; margin-right: 15px;

    }

    .food-info-name { font-weight: 600; font-size: 1rem; color: #2D392E !important; }

    .food-info-revenue { font-size: 0.8rem; color: #A59E92 !important; margin-top: 2px; }

    .select-count-tag { background: #E6ECE6; color: #435241 !important; font-weight: 600; font-size: 0.85rem; padding: 6px 14px; border-radius: 8px; }



    /* CSS Cột bán cơm nhanh */

    .alert-banner {

        background: #FFF7EE; border: 1px solid #FFE7CF; border-radius: 12px; padding: 12px 15px;

        display: flex; justify-content: space-between; align-items: center; margin-top: 15px; margin-bottom: 20px;

    }

    .quick-sell-price-box {

        border: 1px solid #F3EFE6; background: #FAFAFA; border-radius: 16px; padding: 20px;

        display: flex; justify-content: space-between; align-items: center; margin-top: 15px;

    }

    .quick-sell-label { font-size: 0.8rem; font-weight: 700; color: #A59E92 !important; text-transform: uppercase; }

    .quick-sell-value { font-size: 1.6rem; font-weight: 800; color: #2D392E !important; }

    

    /* Custom style cho Nút bấm của nhà bếp */

    div.stButton > button {

        background-color: #435241 !important; color: white !important;

        border-radius: 12px !important; border: none !important;

        padding: 12px 24px !important; font-weight: 600 !important; font-size: 1rem !important;

        width: 100%; transition: all 0.2s;

    }

    div.stButton > button:hover { background-color: #333F31 !important; transform: translateY(-1px); }

    

    .footer-note { font-size: 0.8rem; color: #A59E92 !important; font-style: italic; margin-top: 25px; line-height: 1.4; }

    </style>

    """

    st.markdown(dashboard_css, unsafe_allow_html=True)



    # ------------------------------------------

    # HÀNG 1: 4 THẺ KPI TRÊN CÙNG

    # ------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    

    with col1:

        st.markdown("""

        <div class="kpi-card">

            <div>

                <div class="kpi-title">Tổng doanh thu quầy</div>

                <div class="kpi-value">374.500đ</div>

                <div class="kpi-sub">↗ +12.4% so với hôm qua</div>

            </div>

            <div class="kpi-icon" style="background: #EFF6F0;"><span style="font-size: 20px;">🪙</span></div>

        </div>

        """, unsafe_allow_html=True)

        

    with col2:

        st.markdown("""

        <div class="kpi-card">

            <div>

                <div class="kpi-title">Tổng số khay bán ra</div>

                <div class="kpi-value">13 khay</div>

                <div class="kpi-sub-gray">Đã thanh toán kiểm soát</div>

            </div>

            <div class="kpi-icon" style="background: #FDF5E6;"><span style="font-size: 20px;">🥞</span></div>

        </div>

        """, unsafe_allow_html=True)

        

    with col3:

        st.markdown("""

        <div class="kpi-card">

            <div>

                <div class="kpi-title">Giá khay trung bình</div>

                <div class="kpi-value">28.808đ</div>

                <div class="kpi-sub-gray">Báo cáo thực đơn tự động</div>

            </div>

            <div class="kpi-icon" style="background: #F0F7FF;"><span style="font-size: 20px;">📈</span></div>

        </div>

        """, unsafe_allow_html=True)

        

    with col4:

        st.markdown("""

        <div class="kpi-card">

            <div>

                <div class="kpi-title">Cổng camera giám sát</div>

                <div class="kpi-value" style="font-size: 1.4rem; margin-top: 8px;">CAM–01 ACTIVE</div>

                <div class="kpi-sub" style="color: #2E7D32 !important;">🟢 17/17 món được nạp</div>

            </div>

            <div class="kpi-icon" style="background: #E8F5E9;"><span style="font-size: 20px;">📷</span></div>

        </div>

        """, unsafe_allow_html=True)



    st.write("")



    # ------------------------------------------

    # HÀNG 2: XU HƯỚNG DOANH THU & PHƯƠNG THỨC THANH TOÁN

    # ------------------------------------------

    col_chart_left, col_chart_right = st.columns([1.7, 1.1])

    

    with col_chart_left:

        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)

        st.markdown('<div class="card-title">Xu Hướng Doanh Thu 7 Ngày Qua</div>', unsafe_allow_html=True)

        st.markdown('<div class="card-subtitle">Dữ liệu được cập nhật thực tế từ quầy thông tin</div>', unsafe_allow_html=True)

        

        # Tạo Mock Data biểu đồ đường theo đúng hình ảnh mẫu

        chart_data = {

            "Ngày": ["03/06", "04/06", "05/06", "06/06", "07/06", "08/06", "09/06"],

            "Doanh thu ngày (đ)": [50000, 62000, 72000, 48000, 55000, 82000, 0]

        }

        st.area_chart(data=chart_data, x="Ngày", y="Doanh thu ngày (đ)", color="#5D6B54")

        st.markdown('</div>', unsafe_allow_html=True)

        

    with col_chart_right:

        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)

        st.markdown('<div class="card-title">Phương thức thanh toán</div>', unsafe_allow_html=True)

        st.markdown('<div class="card-subtitle">Tỷ lệ sử dụng dòng tiền</div>', unsafe_allow_html=True)

        

        # Thiết lập cột hiển thị dữ liệu thanh toán trực quan

        st.markdown("""

        <div style="text-align: center; margin-bottom: 20px; padding: 15px; background: #FAF8F5; border-radius: 50%; width: 120px; height: 120px; margin: 0 auto 20px auto; border: 6px solid #5D6B54; display: flex; flex-direction: column; justify-content: center;">

            <span style="font-size: 0.7rem; color: #A59E92 !important; font-weight: bold;">DOANH SỐ</span>

            <span style="font-size: 1.05rem; font-weight: 700; color: #2D392E !important;">374.500đ</span>

        </div>

        """, unsafe_allow_html=True)

        

        pay_methods = [

            {"name": "🟢 Thẻ Sinh Viên", "val": "179.000đ"},

            {"name": "🔴 Ví MoMo", "val": "138.500đ"},

            {"name": "🔵 QR Ngân Hàng", "val": "27.000đ"},

            {"name": "🟡 Tiền Mặt", "val": "30.000đ"}

        ]

        for m in pay_methods:

            st.markdown(f"""

            <div style="display: flex; justify-content: space-between; font-size: 0.95rem; margin-bottom: 8px; border-bottom: 1px dashed #F3EFE6; padding-bottom: 4px;">

                <span style="font-weight: 500;">{m['name']}</span>

                <span style="font-weight: 700; color: #2D392E !important;">{m['val']}</span>

            </div>

            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)



    st.write("")



    # ------------------------------------------

    # HÀNG 3: XU HƯỚNG ĐĨA ĂN BÁN CHẠY & BÁN CƠM NHANH TẠI QUẦY

    # ------------------------------------------

    col_bottom_left, col_bottom_right = st.columns([1.4, 1.4])

    

    with col_bottom_left:

        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)

        st.markdown('<div class="card-title">Xu hướng đĩa ăn bán chạy</div>', unsafe_allow_html=True)

        st.markdown('<div class="card-subtitle">Báo cáo top 5 món ăn sinh viên yêu mến nhất</div>', unsafe_allow_html=True)

        

        top_foods = [

            {"rank": "#1", "name": "Sườn xào chua ngọt", "rev": "75.000đ", "count": "3 lượt chọn"},

            {"rank": "#2", "name": "Rau muống xào tỏi", "rev": "15.000đ", "count": "3 lượt chọn"},

            {"rank": "#3", "name": "Canh bắp cải", "rev": "15.000đ", "count": "3 lượt chọn"},

            {"rank": "#4", "name": "Trứng rán hành tây", "rev": "24.000đ", "count": "3 lượt chọn"},

            {"rank": "#5", "name": "Ba chỉ heo luộc", "rev": "54.000đ", "count": "3 lượt chọn"}

        ]

        

        for food in top_foods:

            st.markdown(f"""

            <div class="food-rank-row">

                <div style="display: flex; align-items: center;">

                    <div class="rank-number">{food['rank']}</div>

                    <div>

                        <div class="food-info-name">{food['name']}</div>

                        <div class="food-info-revenue">Doanh thu tích lũy: {food['rev']}</div>

                    </div>

                </div>

                <div class="select-count-tag">{food['count']}</div>

            </div>

            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

        

    with col_bottom_right:

        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)

        st.markdown('<div class="card-title">Bán cơm nhanh tại quầy nhanh</div>', unsafe_allow_html=True)

        st.markdown('<div class="card-subtitle">Cách ghi nhận đơn cơm sinh viên nhanh không qua Camera</div>', unsafe_allow_html=True)

        

        st.write("Nhân viên có thể nhanh chóng chọn khay cơm mẫu đã nạp sẵn ở tab Kiosk và trực tiếp ghi nhận hóa đơn thanh toán tiền mặt/Momo để hệ thống kiểm soát thống kê doanh số.")

        

        # Thẻ tổng kết hóa đơn hôm nay

        st.markdown("""

        <div class="alert-banner">

            <span style="font-weight: 600; color: #C2780E !important;">ℹ️ Tổng kết doanh số quầy hôm nay</span>

            <span style="background: #FFFFFF; border: 1px solid #FFE7CF; padding: 2px 10px; border-radius: 6px; font-weight: bold; font-size: 0.85rem; color: #C2780E !important;">0 bills</span>

        </div>

        """, unsafe_allow_html=True)

        

        # Thẻ hiển thị giá khay cơm mẫu đang chờ

        st.markdown("""

        <div class="quick-sell-price-box">

            <div>

                <div class="quick-sell-label">Giá khay cơm Kiosk đang chờ:</div>

                <div class="quick-sell-value">35.000đ</div>

            </div>

            <div></div>

        </div>

        """, unsafe_allow_html=True)

        

        st.write("")

        # Nút bấm tích hợp tính năng Streamlit thực tế

        if st.button("BÁN CƠM NHANH BẰNG TIỀN MẶT"):

            st.toast("🎉 Đã ghi nhận hóa đơn 35.000đ thành công!", icon="💰")

            

        st.markdown("""

        <div class="footer-note">

            * Sau khi nhấn bán nhanh bằng tiền mặt, hệ thống tự động ghi nhận món lên hóa đơn và in kết toán hóa đơn.

        </div>

        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

