import config

class BillingSystem:
    @staticmethod
    def generate_receipt(predictions_dict):
        """
        Nhận vào Dict dạng: {"Cơm": "Cơm trắng", "Món mặn": "Thịt kho"}
        Trả về tổng tiền và danh sách text hóa đơn.
        """
        total_bill = 0
        receipt_lines = []

        for region_name, food_name in predictions_dict.items():
            # Truy xuất giá từ config
            price = config.PRICE_MAP.get(food_name, 0)
            total_bill += price
            
            if food_name == "Khay inox (Trống)":
                receipt_lines.append(f"• {region_name}: Trống (0đ)")
            else:
                receipt_lines.append(f"• {region_name}: {food_name} ({price:,}đ)")

        return total_bill, receipt_lines
