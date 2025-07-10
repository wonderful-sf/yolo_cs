
from ultralytics import YOLO

model = YOLO('yolo11n.pt')
model.train(
    data = 'data.yaml',
    batch = 8,
    device = 0,
    epochs = 200,
    workers= 0
)