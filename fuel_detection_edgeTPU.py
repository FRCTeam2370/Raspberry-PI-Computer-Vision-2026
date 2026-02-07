import cv2
from ultralytics import YOLO

model_input_size = 352
# camera_fov should be in radians
camera_fov= 1.195551

model = YOLO("1_29_26_full_integer_quant_edgetpu.tflite", task="detect")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, rawframe = cap.read()

    if success:
        frame = cv2.resize(rawframe, (model_input_size, model_input_size))

        results = model(frame, conf=0.5)[0]

        #print(results)
        boxes = results.boxes[results.boxes.conf > 0.3]
        for b in boxes:
            tx, ty, tw, th = b.xywh[0]
            x = tx.item()
            w = tw.item()
            # target position is from - half of the camera fov to + half the camera fov
            target_position = (x + (w/2) - (model_input_size/2))
            radians = -2*target_position/model_input_size*camera_fov
            # Radians should be positive to the left and negative to the right
            print(radians)
        annotatedFrame = results.plot()
        cv2.imshow("YOLO Inference", annotatedFrame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    #break
cap.release()
cv2.destroyAllWindows()