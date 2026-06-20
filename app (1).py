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
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Montserrat', sans-serif;
    }
    .stApp {
        background-color: #FFFDF9;
    }
    .st-emotion-cache-1jicfl2 {
        padding: 2rem 3rem;
    }
    .step-banner {
        background: linear-gradient(135deg, #FF7E5F 0%, #FEB47B 100%);
        color: white;
        padding: 12px 20px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.2rem;
        margin-bottom: 15px;
        margin-top: 25px;
        box-shadow: 0 4px 6px rgba(255, 126, 95, 0.2);
        display: flex;
        align-items: center;
        gap: 10px;
    }
    [data-testid="stFileUploadDropzone"] {
        background-color: #ffffff !important;
        border: 2px dashed #FF7E5F !important;
        border-radius: 12px !important;
    }
    [data-testid="stFileUploadDropzone"] div {
        color: #2C3E50 !important;
    }
    .stRadio label, .stRadio p {
        font-weight: 600 !important;
        color: #2C3E50 !important;
    }
    .receipt-container {
        background-color: #FEF9E7;
        border: 2px dashed #F39C12;
        border-radius: 10px;
        padding: 5px;
        box-shadow: inset 0 0 10px rgba(0,0,0,0.05);
    }
    h1 {
        color: #D35400 !important;
        text-align: center;
        font-weight: 800 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .subtitle-text {
        text-align: center;
        font-size: 1.1rem;
        color: #7F8C8D;
        margin-bottom: 30px;
    }
    .food-label {
        text-align: center;
        font-weight: 700;
        color: #2C3E50;
        margin-top: 10px;
        margin-bottom: 5px;
        font-size: 1.1rem;
    }
    </style>
    """
    st.markdown(app_bg, unsafe_allow_html=True)

    with st.spinner("⏳ Khởi động động cơ AI, vui lòng đợi trong giây lát..."):
        model = init_model()

    st.markdown("<h1>Khu Vực Thanh Toán Thu Ngân</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle-text'>Xác nhận khay ăn tự động • Nhanh chóng • Chính xác</p>", unsafe_allow_html=True)
    
    st.write("---")

    col_input, col_preview = st.columns([1.2, 1], gap="large")

    with col_input:
        st.markdown("<div class='step-banner'>📸 BƯỚC 1: TẢI HOẶC CHỤP ẢNH KHAY CƠM</div>", unsafe_allow_html=True)
        # THÊM WIDGET CAMERA
        camera_file = st.camera_input("Chụp ảnh trực tiếp từ Camera")
        uploaded_file = st.file_uploader("Hoặc kéo thả ảnh/chọn từ thiết bị (Hỗ trợ: jpg, png)", type=["jpg", "jpeg", "png"])
        
        st.markdown("<div class='step-banner'>🔄 BƯỚC 2: TÙY CHỈNH GÓC NHÌN</div>", unsafe_allow_html=True)
        rotation_mode = st.radio(
            "Cài đặt căn lề tự động của AI (Chỉ can thiệp khi cần thiết):",
            ("Tự động chỉnh hướng", "Giữ nguyên (0°)", "Xoay 90° theo chiều KĐH (CW)", "Xoay 90° ngược chiều KĐH (CCW)", "Xoay 180°"),
            horizontal=True
        )

    # Ưu tiên sử dụng ảnh chụp từ camera nếu có, ngược lại dùng ảnh upload
    active_file = camera_file if camera_file is not None else uploaded_file

    if active_file is not None:
        image = Image.open(active_file).convert('RGB')
        img_array = np.array(image)

        with col_preview:
            st.markdown("<div class='step-banner'>🎯 KẾT QUẢ CĂN LỀ AI</div>", unsafe_allow_html=True)
            with st.spinner("⏳ Trí tuệ nhân tạo đang xoay lật và tối ưu hóa ảnh..."):
                if rotation_mode == "Tự động chỉnh hướng":
                    img_aligned = auto_align_tray(img_array)
                elif rotation_mode == "Xoay 90° theo chiều KĐH (CW)":
                    img_aligned = cv2.rotate(img_array, cv2.ROTATE_90_CLOCKWISE)
                elif rotation_mode == "Xoay 90° ngược chiều KĐH (CCW)":
                    img_aligned = cv2.rotate(img_array, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif rotation_mode == "Xoay 180°":
                    img_aligned = cv2.rotate(img_array, cv2.ROTATE_180)
                else:
                    img_aligned = img_array
                
                st.image(img_aligned, use_container_width=True)

        st.write("---")
        
        st.markdown("<div class='step-banner'>🍱 BƯỚC 3: KẾT QUẢ QUÉT TỪNG NGĂN</div>", unsafe_allow_html=True)
        
        h, w, _ = img_aligned.shape
        regions = {
            "Ngăn Cơm": img_aligned[int(h*0.02):int(h*0.44), int(w*0.02):int(w*0.54)],
            "Ngăn Canh": img_aligned[int(h*0.46):int(h*0.98), int(w*0.02):int(w*0.54)],
            "Ngăn Món 1": img_aligned[int(h*0.02):int(h*0.32), int(w*0.56):int(w*0.98)],
            "Ngăn Món 2": img_aligned[int(h*0.34):int(h*0.64), int(w*0.56):int(w*0.98)],
            "Ngăn Món 3": img_aligned[int(h*0.66):int(h*0.98), int(w*0.56):int(w*0.98)]
        }
        
        total_bill = 0
        receipt_lines = []
        ai_messages = []
        cols = st.columns(5)
        
        for idx, (region_name, region_img) in enumerate(regions.items()):
            with cols[idx]:
                if region_img.shape[0] == 0 or region_img.shape[1] == 0:
                    st.error("⚠️ Lỗi cắt ảnh")
                    continue
                    
                img_resized = cv2.resize(region_img, (224, 224))
                img_batch = np.expand_dims(img_resized, axis=0).astype('float32')
                img_batch = preprocess_input(img_batch)
                
                predictions = model.predict(img_batch, verbose=0)
                predicted_class_idx = np.argmax(predictions[0])
                confidence = np.max(predictions[0]) * 100
                
                food_name = CLASS_NAMES[predicted_class_idx]
                price = PRICE_MAP[food_name]
                total_bill += price
                
                st.image(region_img, use_container_width=True)
                st.markdown(f"<div class='food-label'>{region_name}</div>", unsafe_allow_html=True)
                
                if predicted_class_idx == 2:
                    st.info(f"⚪ Trống \n\n {confidence:.1f}%")
                    receipt_lines.append(f"• {region_name}: Trống (0đ)")
                elif confidence > 65:
                    st.success(f"✅ {food_name} \n\n {price:,}đ")
                    receipt_lines.append(f"• {region_name}: {food_name} ({price:,}đ)")
                else:
                    st.warning(f"⚠️ {food_name}? \n\n {price:,}đ")
                    receipt_lines.append(f"• {region_name}: {food_name} [Cần xác nhận] ({price:,}đ)")
                
                ai_messages.append(f"🛰️ [Co-pilot]: Phát hiện {food_name} tại {region_name}, độ tin cậy {confidence:.1f}%.")

        st.write("---")
        
        # BƯỚC 4: HÓA ĐƠN VÀ CONSOLE KIỂU MỚI ĐẸP HƠN
        col_receipt, col_console = st.columns([1, 1])
        
        with col_receipt:
            st.markdown("<div class='step-banner'>🧾 BƯỚC 4: HÓA ĐƠN ĐIỆN TỬ</div>", unsafe_allow_html=True)
            st.markdown("<div class='receipt-container'>", unsafe_allow_html=True)
            st.text_area("Chi tiết hóa đơn:", value="\n".join(receipt_lines), height=150, disabled=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown(
                f"""
                <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); padding: 20px; border-radius: 12px; text-align: center; color: white; box-shadow: 0 4px 10px rgba(56, 239, 125, 0.3); margin-top: 15px; margin-bottom: 15px;">
                    <h4 style="color: white !important; margin: 0; font-weight: 600; font-size: 1.1rem;">TỔNG THANH TOÁN</h4>
                    <h1 style="color: white !important; font-size: 2.8rem; margin: 5px 0 0 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.2); font-weight: 800;">{total_bill:,} ₫</h1>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            if st.button("🖨️ XUẤT HÓA ĐƠN", use_container_width=True, type="primary"):
                st.toast("✅ Giao dịch thành công!")
                st.balloons()

        with col_console:
            st.markdown("<div class='step-banner'>🤖 AI CO-PILOT CONSOLE</div>", unsafe_allow_html=True)
            st.info("\n\n".join(ai_messages))

# ------------------------------------------
# TRANG 3: GÓC ẨM THỰC AI (ĐÃ HOÀN THIỆN ẢNH VIỆT NAM & HIỆU ỨNG PHÁT SÁNG)
# ------------------------------------------
elif page == "Góc Ẩm Thực AI":
    # CSS Custom tạo hiệu ứng phát sáng Neon chuyển màu lung linh cho thẻ món ăn
    glow_css = """
    <style>
    [data-testid="stVVerticalBlock"] > div:has(div.stElementContainer) {
        transition: all 0.3s ease-in-out;
    }
    /* Áp dụng hiệu ứng viền phát sáng viễn tưởng cho các Container món ăn */
    .stElementContainer:has(.food-card-glow) {
        border-radius: 12px;
        box-shadow: 0 0 15px rgba(255, 126, 95, 0.5), inset 0 0 10px rgba(254, 180, 123, 0.2);
        border: 1px solid rgba(255, 126, 95, 0.4);
        padding: 10px;
        background: #ffffff;
    }
    </style>
    """
    st.markdown(glow_css, unsafe_allow_html=True)

    st.markdown("<h1>✨ GÓC GỢI Ý MÓN NGON TỪ AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Để AI giúp bạn chọn một bữa ăn đầy đủ dinh dưỡng và đậm đà bản sắc cơm Việt hôm nay!</p>", unsafe_allow_html=True)
    st.write("---")

    # Dữ liệu hình ảnh được thay đổi chính xác sang phong cách Cơm Văn Phòng / Cơm Bình Dân Việt Nam
    suggestions = [
        {
            "name": "Sườn cốt lết nướng", 
            "desc": "Sườn cốt lết được đập mềm, ướp sả mật ong nướng vàng xém cạnh, chuẩn vị cơm tấm văn phòng.", 
            "appeal": "⭐⭐⭐⭐⭐ (Siêu hấp dẫn)",
            "image": "https://i.ytimg.com/vi/picf2oCcsb0/maxresdefault.jpg"
        },
        {
            "name": "Thịt kho tàu (Thịt kho trứng)", 
            "desc": "Thịt ba chỉ vuông vức kho r rục cùng hột vịt, nước kho dừa màu cánh gián đậm đà đưa cơm.", 
            "appeal": "⭐⭐⭐⭐⭐ (Khó cưỡng)",
            "image": "https://s.yimg.com/fz/api/res/1.2/c5fsSJl9N4niV2tMd7EapQ--~C/YXBwaWQ9c3JjaGRkO2ZpPWZpbGw7aD00MTI7cHhvZmY9NTA7cHlvZmY9MTAwO3E9ODA7c3M9MTt3PTM4OA--/https://i.pinimg.com/736x/7a/52/a2/7a52a2036d5c39dc9f8d9bf642cdd30e.jpg"
        },
        {
            "name": "Cá kho", 
            "desc": "Khúc cá kho ngậy được kho kẹo trong tộ đất với tóp mỡ, hành lá và tiêu sọ cay nồng.", 
            "appeal": "⭐⭐⭐⭐ (Đậm đà dân dã)",
            "image": "https://i.ytimg.com/vi/BfD1KwGwkqA/maxresdefault.jpg"
        },
        {
            "name": "Trứng chiên", 
            "desc": "Trứng vịt đánh bông chiên dày hành thơm phức, ngoài rìa xém giòn, bên trong mềm xốp.", 
            "appeal": "⭐⭐⭐ (Nhanh gọn, bắt miệng)",
            "image": "https://cdn2.fptshop.com.vn/unsafe/1920x0/filters:format(webp):quality(75)/2023_10_18_638332629116893732_thiet-ke-chua-co-ten-3.jpg"
        },
        {
            "name": "Canh chua", 
            "desc": "Nước canh chua thanh nhẹ từ me, nấu kèm đậu bắp, cà chua và giá đỗ chuẩn vị cơm bình dân.", 
            "appeal": "⭐⭐⭐⭐ (Giải nhiệt sảng khoái)",
            "image": "https://www.cooking-therapy.com/wp-content/uploads/2020/12/Canh-Chua-13.jpg"
        }
    ]

    col1, col2 = st.columns(2)
    for i, item in enumerate(suggestions):
        with (col1 if i % 2 == 0 else col2):
            # Tạo khối container có class định danh để nhận hiệu ứng phát sáng CSS ở trên
            with st.container(border=True):
                st.markdown("<div class='food-card-glow'></div>", unsafe_allow_html=True)
                st.image(item["image"], use_container_width=True, caption=f"Món ăn thực tế: {item['name']}")
                st.subheader(item["name"])
                st.write(item["desc"])
                st.caption(f"*Đánh giá từ AI:* {item['appeal']}")
                if st.button(f"Chọn {item['name']}", key=f"btn_{i}", use_container_width=True):
                    st.success(f"Đã ghi chú! AI hệ thống đã thêm {item['name']} vào thực đơn lý tưởng của bạn.")

    st.write("---")
    st.info("💡 *Mẹo nhỏ từ AI:* Bộ đôi Combo bất bại của dân văn phòng: **Sườn cốt lết nướng** kết hợp cùng **Canh chua dọc mùng** sẽ mang lại nguồn năng lượng tối đa!")
