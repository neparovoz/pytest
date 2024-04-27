import pytest
import os
import cv2
import numpy as np
import tensorflow as tf
from src.preprocess.preprocess import preprocess_image

def test_preprocess_image():
    img_path = 'input_images/AgACAgIAAxkBAAIBNGYrxljN-swncAMi7f3fUsw5S4X_AAKg1DEbDgZgSciBsGfBWPX0AQADAgADeQADNAQ'
    img = load_image(img_path)
    processed_img = preprocess_image(img)
    assert np.max(processed_img) <= 1.0
    assert np.min(processed_img) >= -1.0

def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 127.5 - 1
    img = np.expand_dims(img, 0)
    img = tf.convert_to_tensor(img)
    return img