import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

CROP_DIR = '../data/raw/crop/'
PROCESSED_DIR = '../data/ready/'

IMG_SIZE = 128


def load_images_and_labels():
    images = []
    labels = []
    classes = os.listdir(CROP_DIR)
    class_mapping = {name: idx for idx, name in enumerate(
        classes)}

    for class_name in classes:
        class_path = os.path.join(CROP_DIR, class_name)
        if os.path.isdir(class_path):
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    img_resized = cv2.resize(
                        img, (IMG_SIZE, IMG_SIZE))
                    images.append(img_resized)
                    labels.append(class_mapping[class_name])

    images = np.array(images)
    labels = np.array(labels)
    return images, labels, classes


def split_and_save_data():
    images, labels, classes = load_images_and_labels()

    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=42)

    np.save(os.path.join(PROCESSED_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(PROCESSED_DIR, 'X_val.npy'), X_val)
    np.save(os.path.join(PROCESSED_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(PROCESSED_DIR, 'y_val.npy'), y_val)

    with open(os.path.join(PROCESSED_DIR, 'classes.npy'), 'wb') as f:
        np.save(f, classes)


if __name__ == '__main__':
    split_and_save_data()
    print("Data has been preprocessed and saved.")
