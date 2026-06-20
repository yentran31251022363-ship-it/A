import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import cv2
import config

class FoodClassifier:
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            self.model = tf.keras.models.load_model(config.MODEL_PATH)
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"⚠️ Error loading model: {e}")

    def predict_region(self, region_img):
        """Nhận ảnh đã cắt, đưa qua EfficientNet và trả về Tên món & Độ tin cậy"""
        if self.model is None or region_img.shape[0] == 0 or region_img.shape[1] == 0:
            return "Lỗi cắt ảnh", 0.0

        img_resized = cv2.resize(region_img, config.IMG_SIZE)
        img_batch = np.expand_dims(img_resized, axis=0).astype('float32')
        img_batch = preprocess_input(img_batch)

        predictions = self.model.predict(img_batch, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0]) * 100

        food_name = config.CLASS_NAMES[predicted_class_idx]
        return food_name, confidence
