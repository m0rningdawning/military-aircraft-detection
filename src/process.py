import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

IMG_SIZE = 128  
# exterminate that
CATEGORIES = ['fighter_jet', 'helicopter', 'bomber', 'drone']  
DATASET_PATH = 'data/kaggle_dataset/'  

def load_and_preprocess_images():
    data = []
    labels = []

    for category in CATEGORIES:
        path = os.path.join(DATASET_PATH, category)
        class_label = CATEGORIES.index(category)
        
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path)
                img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                data.append(img_resized)
                labels.append(class_label)
            except Exception as e:
                print(f"Error loading image {img_name}: {e}")

    data = np.array(data) / 255.0  
    labels = np.array(labels)
    return data, labels

def split_data(data, labels):
    return train_test_split(data, labels, test_size=0.2, random_state=42)

if __name__ == "__main__":
    X, y = load_and_preprocess_images()
    X_train, X_val, y_train, y_val = split_data(X, y)
    np.save('data/processed/X_train.npy', X_train)
    np.save('data/processed/X_val.npy', X_val)
    np.save('data/processed/y_train.npy', y_train)
    np.save('data/processed/y_val.npy', y_val)
