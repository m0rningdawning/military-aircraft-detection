# import os
# import numpy as np
# import cv2
# from sklearn.model_selection import train_test_split

# CROP_DIR = '../data/raw/crop/'

# IMG_SIZE = 128  

# def load_images_and_labels():
#     images = []
#     labels = []
    
#     classes = os.listdir(CROP_DIR)
#     class_mapping = {name: idx for idx, name in enumerate(classes)}
    
#     print(f"Found {len(classes)} classes.")
    
#     for class_name in classes:
#         class_path = os.path.join(CROP_DIR, class_name)
        
#         if os.path.isdir(class_path):
#             print(f"Processing class: {class_name}")
            
#             for img_name in os.listdir(class_path):
#                 img_path = os.path.join(class_path, img_name)
                
#                 img = cv2.imread(img_path)
#                 if img is None:
#                     print(f"Image {img_name} could not be loaded. Skipping...")
#                     continue 
                
#                 img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                
#                 images.append(img_resized)
#                 labels.append(class_mapping[class_name])
    
#     images = np.array(images)
#     labels = np.array(labels)
    
#     return images, labels, class_mapping

# def split_and_save_dataset():
#     X, y, class_mapping = load_images_and_labels()
    
#     print(f"Total images loaded: {X.shape[0]}")
    
#     X = X / 255.0
    
#     X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
#     np.save('../data/ready/X_train.npy', X_train)
#     np.save('../data/ready/X_val.npy', X_val)
#     np.save('../data/ready/y_train.npy', y_train)
#     np.save('../data/ready/y_val.npy', y_val)
    
#     np.save('../data/ready/classes.npy', class_mapping)
    
#     print(f"Training set: {X_train.shape[0]} samples")
#     print(f"Validation set: {X_val.shape[0]} samples")

# if __name__ == "__main__":
#     split_and_save_dataset()

import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

CROP_DIR = '../data/raw/crop/'
IMG_SIZE = 128 

def load_images_and_labels():
    images = []
    labels = []
    
    classes = os.listdir(CROP_DIR)
    class_mapping = {name: idx for idx, name in enumerate(classes)}
    
    print(f"Found {len(classes)} classes.")
    
    for class_name in classes:
        class_path = os.path.join(CROP_DIR, class_name)
        
        if os.path.isdir(class_path):
            print(f"Processing class: {class_name}")
            
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Image {img_name} could not be loaded. Skipping...")
                    continue 
                
                img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                
                images.append(img_resized)
                labels.append(class_mapping[class_name])
    
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels, class_mapping

def split_and_save_dataset():
    X, y, class_mapping = load_images_and_labels()
    
    print(f"Total images loaded: {X.shape[0]}")
    
    X = X / 255.0

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    np.save('../data/ready/X_train.npy', X_train)
    np.save('../data/ready/X_val.npy', X_val)
    np.save('../data/ready/y_train.npy', y_train)
    np.save('../data/ready/y_val.npy', y_val)
    
    np.save('../data/ready/classes.npy', class_mapping)

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")

if __name__ == "__main__":
    split_and_save_dataset()
