import os

# CẤU HÌNH ĐƯỜNG DẪN MODEL
# Nếu chạy trên máy cá nhân/Streamlit, để model cùng thư mục với code
MODEL_PATH = "canteen_model_STAGE1.keras" 

# THÔNG SỐ ẢNH DÀNH CHO EFFICIENTNET
IMG_SIZE = (224, 224)

# BẢNG GIÁ THỰC TẾ
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

# DANH SÁCH MÓN ĂN (Phải khớp thứ tự thư mục train)
CLASS_NAMES = [
    "Cơm trắng",          # 0
    "Trứng chiên",        # 1
    "Khay inox (Trống)",  # 2
    "Đậu hũ sốt cà",      # 3
    "Cá hú kho",          # 4
    "Thịt kho trứng",     # 5
    "Thịt kho",           # 6
    "Canh chua",          # 7
    "Sườn nướng",         # 8
    "Canh rau",           # 9
    "Rau xào"             # 10
]
