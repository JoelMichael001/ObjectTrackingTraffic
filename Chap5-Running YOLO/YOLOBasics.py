import cv2
from ultralytics import  YOLO

model = YOLO('../YOLOWeights/yolov8n.pt')
results = model("Images/1.png", show=True)
cv2.waitKey(0)