from ultralytics import YOLO

model = YOLO('original.pt')

results = model(source=0, show=True, conf=0.25)