import numpy as np
from tensorflow.keras.utils import to_categorical
from model import create_model

X_train = np.load('../data/ready/X_train.npy')
X_val = np.load('../data/ready/X_val.npy')
y_train = np.load('../data/ready/y_train.npy')
y_val = np.load('../data/ready/y_val.npy')

y_train = to_categorical(y_train, num_classes=73)
y_val = to_categorical(y_val, num_classes=73)

model = create_model()

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=32)

model.save('../models/aircraft_classifier.h5')

np.save('../models/training_history.npy', history.history)
