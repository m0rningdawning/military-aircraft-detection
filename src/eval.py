import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

X_val = np.load('data/processed/X_val.npy')
y_val = np.load('data/processed/y_val.npy')

model = load_model('models/aircraft_classifier.h5')

y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)

print("Classification Report:\n")
# change that shi 
print(classification_report(y_val, y_pred_classes, target_names=['Fighter Jet', 'Helicopter', 'Bomber', 'Drone']))

cm = confusion_matrix(y_val, y_pred_classes)
plt.figure(figsize=(8, 6))
# and that
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fighter Jet', 'Helicopter', 'Bomber', 'Drone'], yticklabels=['Fighter Jet', 'Helicopter', 'Bomber', 'Drone'])
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
