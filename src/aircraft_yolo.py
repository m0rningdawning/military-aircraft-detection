from ultralytics import YOLO
import cv2

model = YOLO('original.pt')

image_paths = [
    '../datasets/military/images/aircraft_val/cd7aaef78e9a0efd458aa3b8385f1e23.jpg',
    '../datasets/military/images/aircraft_val/ffa565e4f573979207ea525e63c204d5.jpg',
    '../datasets/military/images/aircraft_val/ffa8d739b58bbf6b65e6d5dd9b2e8a69.jpg',
    '../datasets/military/images/aircraft_val/ff600521dbd019802c3ada1bbe19ad8e.jpg',
    '../datasets/military/images/aircraft_val/ffff0595d9b782e9cd3c537529bf6027.jpg',
    '../datasets/other/lol.png',
    '../datasets/other/apache.png',
    '../datasets/other/rq4.png',
]

screen_width = 1920
screen_height = 1080


def resize_to_screen(img, screen_width, screen_height):
    img_height, img_width = img.shape[:2]
    scale_width = screen_width / img_width
    scale_height = screen_height / img_height
    scale = min(scale_width, scale_height)
    new_width = int(img_width * scale)
    new_height = int(img_height * scale)
    return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)


current_index = 0

while True:
    results = model(source=image_paths[current_index], conf=0.40)
    img = results[0].plot()

    resized_img = resize_to_screen(img, screen_width, screen_height)

    cv2.imshow('Prediction', resized_img)

    key = cv2.waitKey(0)

    if key == ord('n'):
        current_index = (current_index + 1) % len(image_paths)
    elif key == ord('p'):
        current_index = (current_index - 1) % len(image_paths)
    elif key == 27:
        break

cv2.destroyAllWindows()
