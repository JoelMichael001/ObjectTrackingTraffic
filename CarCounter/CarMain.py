import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from cvzone.SerialModule import SerialObject
from time import sleep

from sort import *
arduino = SerialObject("COM5")

# Video source
cap = cv2.VideoCapture(1)  # For Webcam
cap.set(3, 1280)
cap.set(4, 720)
# cap = cv2.VideoCapture("../Videos/motorbikes.mp4")  # For Video

# YOLO model
model = YOLO("../Yolo-Weights/yolov8l.pt")

# Class names
classNames = ["person", "bicycle", "Car", "motorbike", "aeroplane", "Bus", "train", "Truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Mask
Mask = cv2.imread("r (1).png")
tracker = Sort(max_age=20, min_hits=2, iou_threshold=0.3)

# Initialize variables for counting objects
active_object_ids = set()
total_objects_in_frame = 0
delay1=3


while True:
    success, img = cap.read()
        # Apply mask
    imgRegion = cv2.bitwise_and(img, Mask)

    # Object detection
    results = model(imgRegion, stream=True)
    detections = np.empty((0, 5))
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Filter by object type and confidence
            if currentClass in ["Car", "Bus", "Truck"] and conf > 0.3:
                cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    # Object tracking
    resultsTracker = tracker.update(detections)
    current_frame_ids = set()

    for result in resultsTracker:
        x1, y1, x2, y2, obj_id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        # Draw bounding boxes and IDs
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5, colorR=(255, 0, 0))
        cvzone.putTextRect(img, f'{int(obj_id)}', (max(0, x1), max(35, y1)),
                           scale=1, thickness=1, offset=3)

        # Update IDs for the current frame
        current_frame_ids.add(int(obj_id))

    # Update object counts
    new_objects = current_frame_ids - active_object_ids
    exited_objects = active_object_ids - current_frame_ids

    active_object_ids = current_frame_ids
    total_objects_in_frame = len(active_object_ids)

    # Display the object count
    cvzone.putTextRect(img, f'Active Objects: {total_objects_in_frame}', (50, 50), scale=2, thickness=2, colorR=(0, 255, 0))
    if total_objects_in_frame <= 4 :
        arduino.sendData([0])
    else:
        arduino.sendData([1])

    cv2.imshow("Image", img)
    cv2.waitKey(1)  # Press 'q' to exit
