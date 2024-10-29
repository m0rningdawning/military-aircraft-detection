# import numpy as np
# from tensorflow.keras.models import load_model
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns

# X_val = np.load('../data/ready/X_val.npy')
# y_val = np.load('../data/ready/y_val.npy')

# classes = np.load('../data/ready/classes.npy')

# model = load_model('../models/aircraft_classifier.h5')

# y_pred = model.predict(X_val)
# y_pred_classes = np.argmax(y_pred, axis=1)

# print("Classification Report:\n")
# print(classification_report(y_val, y_pred_classes, target_names=classes))

# cm = confusion_matrix(y_val, y_pred_classes)
# plt.figure(figsize=(10, 8))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix')
# plt.show()

import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

X_val_path = '../data/ready/X_val.npy'
y_val_path = '../data/ready/y_val.npy'

def load_data():
    X_val = np.load(X_val_path)
    y_val = np.load(y_val_path)
    return X_val, y_val

def evaluate():
    model = load_model('../models/aircraft_classifier.h5')
    
    X_val, y_val = load_data()
    
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print(classification_report(y_val, y_pred_classes))
    
    cm = confusion_matrix(y_val, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_val), yticklabels=np.unique(y_val))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    evaluate()
