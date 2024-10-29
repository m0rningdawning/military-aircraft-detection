# import numpy as np
# from tensorflow.keras.utils import to_categorical
# from model import create_model

# X_train = np.load('../data/ready/X_train.npy')
# X_val = np.load('../data/ready/X_val.npy')
# y_train = np.load('../data/ready/y_train.npy')
# y_val = np.load('../data/ready/y_val.npy')

# y_train = to_categorical(y_train, num_classes=73)
# y_val = to_categorical(y_val, num_classes=73)

# model = create_model()

# history = model.fit(X_train, y_train, validation_data=(
#     X_val, y_val), epochs=20, batch_size=32)

# model.save('../models/aircraft_classifier.h5')

# np.save('../models/training_history.npy', history.history)

# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.applications import EfficientNetB3
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
# from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
# from sklearn.model_selection import train_test_split

# IMG_SIZE = 128 
# BATCH_SIZE = 32
# EPOCHS = 20
# DATA_DIR = '../data/ready/'

# X_train = np.load(DATA_DIR + 'X_train.npy')
# y_train = np.load(DATA_DIR + 'y_train.npy')
# X_val = np.load(DATA_DIR + 'X_val.npy')
# y_val = np.load(DATA_DIR + 'y_val.npy')

# X_train, X_val = X_train / 255.0, X_val / 255.0

# base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
# base_model.trainable = False 

# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# x = Dropout(0.5)(x)
# x = Dense(256, activation='relu')(x)
# x = Dropout(0.5)(x)
# predictions = Dense(len(np.unique(y_train)), activation='softmax')(x)

# model = Model(inputs=base_model.input, outputs=predictions)

# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=1e-6)

# history = model.fit(
#     X_train, y_train,
#     validation_data=(X_val, y_val),
#     epochs=EPOCHS,
#     batch_size=BATCH_SIZE,
#     callbacks=[early_stopping, reduce_lr]
# )

# model.save('../models/efficientnet_aircraft_classifier.h5')

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow.keras.preprocessing.image as image
import matplotlib.pyplot as plt

IMG_SIZE = 128  
BATCH_SIZE = 40
EPOCHS = 30
DATA_DIR = '../data/ready/'

X_train = np.load(DATA_DIR + 'X_train.npy')
y_train = np.load(DATA_DIR + 'y_train.npy')
X_val = np.load(DATA_DIR + 'X_val.npy')
y_val = np.load(DATA_DIR + 'y_val.npy')

train_datagen = image.ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
    Dense(256, kernel_regularizer=regularizers.l2(0.016),
          activity_regularizer=regularizers.l1(0.006),
          bias_regularizer=regularizers.l1(0.006), 
          activation='relu'),
    Dropout(rate=0.45, seed=123),
    Dense(len(np.unique(y_train)), activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)

train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)

history = model.fit(
    train_generator,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    verbose=1,
    callbacks=[early_stopping, reduce_lr]
)

model.save('../models/efficientnet_aircraft_classifier.h5')

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.show()
