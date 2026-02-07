import cv2
import time
from ultralytics import YOLO
import importlib

#spec = importlib.util.spec.from_file_location()

model = YOLO("best.pt", task="detect")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, rawframe = cap.read()
    #frame = cv2.resize(rawframe, (320, 320))

    if success:
        frame = cv2.resize(rawframe, (230, 230))
        results = model(frame)[0]
        
        boxes = results.boxes[results.boxes.conf > 0.5]
        
        #annotatedFrame = results.plot()
        #cv2.imshow("YOLO Inference", annotatedFrame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
cap.release()
cv2.destroyAllWindows()