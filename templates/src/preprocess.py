# src/preprocess.py
import cv2
import numpy as np
from PIL import Image

def preprocess_image(image_path, img_size=(256, 256)):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(img_size)
    image_array = np.array(image) / 255.0  # Normalize
    return image_array
