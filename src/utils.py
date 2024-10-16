import os
import numpy as np
import cv2

IMG_SIZE = 128

def load_image(image_path):
    """Load an image and resize it to the required dimensions."""
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    return img_resized

def normalize_image(img):
    """Normalize pixel values of an image."""
    return img / 255.0

def save_model(model, filename):
    """Save a trained model to disk."""
    model.save(filename)

def load_processed_data():
    """Load preprocessed data from disk."""
    X_train = np.load('data/processed/X_train.npy')
    X_val = np.load('data/processed/X_val.npy')
    y_train = np.load('data/processed/y_train.npy')
    y_val = np.load('data/processed/y_val.npy')
    return X_train, X_val, y_train, y_val
