import cv2
import numpy as np

class TrayProcessor:
    @staticmethod
    def auto_align_tray(img):
        """Tự động xoay khay cơm về khuôn dọc chuẩn dựa trên vách ngăn"""
        h, w, _ = img.shape

        # BƯỚC 1: Xử lý ảnh ngang
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

        # BƯỚC 2: Xử lý ảnh dọc bị lộn ngược
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

    @staticmethod
    def manual_rotate(img, mode):
        """Xoay thủ công theo yêu cầu người dùng"""
        if mode == "Xoay 90° theo chiều KĐH (CW)":
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif mode == "Xoay 90° ngược chiều KĐH (CCW)":
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif mode == "Xoay 180°":
            return cv2.rotate(img, cv2.ROTATE_180)
        return img

    @staticmethod
    def crop_regions(img_aligned):
        """Cắt ảnh thành 5 ngăn với tỷ lệ chuẩn và Safe Margin 2%"""
        h, w, _ = img_aligned.shape
        
        regions = {
            "Cơm": img_aligned[int(h*0.02):int(h*0.44), int(w*0.02):int(w*0.54)],
            "Canh": img_aligned[int(h*0.46):int(h*0.98), int(w*0.02):int(w*0.54)],
            "Rau": img_aligned[int(h*0.02):int(h*0.32), int(w*0.56):int(w*0.98)],
            "Nước chấm": img_aligned[int(h*0.34):int(h*0.64), int(w*0.56):int(w*0.98)],
            "Món mặn": img_aligned[int(h*0.66):int(h*0.98), int(w*0.56):int(w*0.98)]
        }
        return regions
