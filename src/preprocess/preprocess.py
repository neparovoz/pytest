import os
import cv2
import numpy as np
import tensorflow as tf
import telebot
import matplotlib.pyplot as plt

def load_image(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 127.5 - 1
    img = np.expand_dims(img, 0)
    img = tf.convert_to_tensor(img)
    return img

def preprocess_image(img, target_dim=224):
    shape = tf.cast(tf.shape(img)[1:-1], tf.float32)
    min_side = min(shape)
    scale = target_dim / min_side
    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = tf.image.resize_with_crop_or_pad(img, target_dim, target_dim)
    return img