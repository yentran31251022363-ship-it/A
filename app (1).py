# ------------------------------------------
# TRANG 3: THỐNG KÊ DOANH THU & BÁN NHANH (Thay cho Góc Ẩm Thực AI)
# ------------------------------------------
elif page == "Góc Ẩm Thực AI":
    dashboard_css = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght=400;500;600;700&display=swap');
    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif;
        background-color: #FDF9F0 !important;
    }
    
    /* Chữ mặc định cho toàn trang */
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
    /* Đổi màu chữ số thứ tự (Rank) thành MÀU TRẮNG trên nền rêu tối */
    .rank-number {
        background: #435241; color: #FFFFFF !important; font-weight: 700; width: 32px; height: 32px;
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
        display: flex; justify-content: space-between; align-items: center; margin-top: 15px; margin-bottom: 20px;
    }
    .quick-sell-label { font-size: 0.8rem; font-weight: 700; color: #A59E92 !important; text-transform: uppercase; }
    .quick-sell-value { font-size: 1.6rem; font-weight: 800; color: #2D392E !important; }
    
    /* Thiết lập chữ MÀU TRẮNG cho nút bấm chính trên nền rêu tối */
    div.stButton > button {
        background-color: #435241 !important; color: #FFFFFF !important;
        border-radius: 12px !important; border: none !important;
        padding: 14px 24px !important; font-weight: 600 !important; font-size: 1rem !important;
        width: 100%; transition: all 0.2s;
    }
    div.stButton > button:hover { background-color: #333F31 !important; transform: translateY(-1px); }
    
    /* Style cho Radio button (Lựa chọn thanh toán) gọn gàng hơn */
    div[data-testid="stRadio"] label {
        font-weight: 500 !important; color: #2D392E !important;
    }
    
    .footer-note { font-size: 0.8rem; color: #A59E92 !important; font-style: italic; margin-top: 20px; line-height: 1.4; }
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
        
        st.markdown("""
        <div style="text-align: center; margin-bottom: 20px; padding: 15px; background: #FAF8F5; border-radius: 50%; width: 120px; height: 120px; margin: 0 auto 20px auto; border: 6px solid #435241; display: flex; flex-direction: column; justify-content: center;">
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
    # HÀNG 3: XU HƯỚNG ĐĨA ĂN BÁN CHẠY & LỰA CHỌN THANH TOÁN TẠI QUẦY
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
        st.markdown('<div class="card-title">Bán cơm nhanh tại quầy</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-subtitle">Lựa chọn phương thức ghi nhận nhanh đơn cơm sinh viên</div>', unsafe_allow_html=True)
        
        st.write("Nhân viên chọn khay cơm mẫu đã nạp sẵn ở tab Kiosk và trực tiếp chọn hình thức thanh toán dưới đây để đồng bộ dữ liệu hệ thống.")
        
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
        
        # SỬA ĐỔI: Thêm lựa chọn thanh toán bằng Radio Button của Streamlit
        payment_choice = st.radio(
            "CHỌN PHƯƠNG THỨC THANH TOÁN:",
            ["💵 Thanh toán tiền mặt", "💳 Quẹt thẻ (Thẻ SV / Momo / QR)"],
            index=0,
            horizontal=True
        )
        
        st.write("")
        # Nút hành động tương ứng với phương thức lựa chọn
        if st.button("XÁC NHẬN BÁN NHANH"):
            if "tiền mặt" in payment_choice:
                st.toast("🎉 Đã ghi nhận thanh toán TIỀN MẶT thành công: 35.000đ", icon="💵")
            else:
                st.toast("🎉 Đã ghi nhận thanh toán QUẸT THẺ thành công: 35.000đ", icon="💳")
            
        st.markdown("""
        <div class="footer-note">
            * Sau khi nhấn nút xác nhận, hệ thống tự động lưu hóa đơn, cập nhật biểu đồ xu hướng doanh thu và xuất lệnh in kết toán quầy.
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
