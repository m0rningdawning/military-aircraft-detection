from ultralytics import YOLO

model = YOLO("yolo8n.pt")

if __name__ == '__main__':
    results = model.train(
        data="aircraft_names.yaml",
        epochs=3,
        imgsz=640,
    )

metrics = model.val()

results = model(
    "../datasets/military/images/aircraft_train/0a09ec2be619a9dcbb8a72e101782a5e.jpg")
results[0].show()

path = model.export(format="onnx")
