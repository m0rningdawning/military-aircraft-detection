import numpy as np
import cv2
from tensorflow.keras.models import load_model

IMG_SIZE = 128

model = load_model('../models/aircraft_classifier.h5')

classes = np.load('../data/ready/classes.npy')


def predict_image(image_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_array = np.expand_dims(img_resized, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)

    return classes[class_index]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Military Aircraft Classifier Prediction')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to the image to be classified')
    args = parser.parse_args()

    predicted_class = predict_image(args.image)
    print(f"The predicted class is: {predicted_class}")
