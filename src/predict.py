import numpy as np
import cv2
from tensorflow.keras.models import load_model

IMG_SIZE = 128
# change that shi to 73
CATEGORIES = ['Fighter Jet', 'Helicopter', 'Bomber', 'Drone']

model = load_model('models/aircraft_classifier.h5')

def predict_image(image_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_array = np.expand_dims(img_resized, axis=0) 
    img_array = img_array / 255.0  

    # Predict
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    
    return CATEGORIES[class_index]

if __name__ == "__main__":
    # that as well
    image_path = 'path_to_test_image.jpg' 
    predicted_class = predict_image(image_path)
    print(f"The predicted class is: {predicted_class}")
