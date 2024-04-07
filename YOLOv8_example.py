
from ultralytics import YOLO

model = YOLO('best.pt')

results = model(source=1, show=True, conf=0.3, save=True)
