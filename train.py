
from ultralytics import YOLO

model = YOLO('yolo11n.pt')
model.train(
    data = 'data.yaml',
    device = 0,
    epochs = 500,
    workers= 0
)