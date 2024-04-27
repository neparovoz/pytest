import cv2
from src.preprocess.preprocess import *


def test_load_image():
    img = load_image("AgACAgIAAxkBAAIBNGYrxljN-swncAMi7f3fUsw5S4X_AAKg1DEbDgZgSciBsGfBWPX0AQADAgADeQADNAQ.jpg")
    assert img.shape == (1, 1280, 958, 3)