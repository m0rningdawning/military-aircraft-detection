from ultralytics import YOLO
import cv2

model = YOLO('best.pt')

image_paths = [
    '../data/raw/crop/A10/1a3ffc625ff0b3eb5c3a9349bfb9bc27_1.jpg',
    '../data/raw/crop/F18/2acc060b1ac622d4cc24680567415f52_0.jpg',
    '../data/raw/crop/F16/0cd99a9ee135c7618006540f5b6d9b1b_0.jpg',
    '../data/raw/crop/A400M/3bf06ff19f1aeb86c98a85fd9cb4973f_2.jpg',
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
    results = model(source=image_paths[current_index], conf=0.25)
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
