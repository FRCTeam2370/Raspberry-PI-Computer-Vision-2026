import cv2 # uv add open-cv
from ultralytics import YOLO # uv add ultralytics
import ntcore # uv add pyntcore https://robotpy.readthedocs.io/en/2023.5/install/pyntcore.html
#from networktables import NetworkTables
import time
import math

# ======
# Setup Constants
# ======
MODEL_INPUT_SIZE = 352
CAMERA_FOV_HORIZONTAL = 1.2137   # CAMERA_FOV_HORIZONTAL should be in radians
CAMERA_FOV_VERTICAL = 0.7636   # CAMERA_FOV_HORIZONTAL should be in radians
CAMERA_CENTER_PIZEL_OFFSET = -40
CAMERA_INPUT_INDEX = 1
MODEL_PATH = "/home/josh/Documents/ibots/2_12_26.2_full_integer_quant_edgetpu.tflite"
DOUBLE_DETECTION_CLOSENESS_TOLERENCES = 0.1
CLUMP_CLOSENESS_TOLERENCES = 0.3
DISTANCE_WEIGHT_SCALAR = 1

# Set up NetworkTable
inst = ntcore.NetworkTableInstance.getDefault()
inst.startClient4("Raspberry Pi")
inst.setServerTeam(2370)
table = inst.getTable("fuelCV") 

# Set up CV model
model = YOLO(MODEL_PATH, task="detect")

# Connect to camera
connected_to_camera = False

while not connected_to_camera:
    try:
        cap = cv2.VideoCapture(0)

        if cap.isOpened():
            connected_to_camera = True
    except:
        connected_to_camera = False

while cap.isOpened():
    success, rawframe = cap.read()

    if success:
        frame = cv2.resize(rawframe, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))

        results = model(frame, conf=0.6)[0]

        yaw_radians = []
        pitch_radians = []
        distances = []
        weights = []

        #print(results)

        boxes = results.boxes[results.boxes.conf > 0.7]
        table.putNumber("number_of_fuel", boxes.__len__())

        for b in boxes:
            tx, ty, tw, th = b.xywh[0]
            x = tx.item()
            y = ty.item()
            w = tw.item()
            h = th.item()


            # target position is from - half of the camera fov to + half the camera fov
            target_position_x = (x + (w/2) - (MODEL_INPUT_SIZE/2)) + CAMERA_CENTER_PIZEL_OFFSET
            target_position_y = (x + (h/2) - (MODEL_INPUT_SIZE/2))

            # Scale pixel position of bounding box to radians. Radians should be positive to the left and negative to the right
            yaw_radian = (-2*target_position_x/MODEL_INPUT_SIZE*CAMERA_FOV_HORIZONTAL)
            pitch_radian = (-2*target_position_x/MODEL_INPUT_SIZE*CAMERA_FOV_VERTICAL)

            # Calculate distance based on bounding box lengths
            distance_height = 60/h
            distance_width = 34.4/w
            if distance_width > distance_height:
                distance = distance_width
            else:
                distance = distance_height

            # Calculate distance weight
            weight = -DISTANCE_WEIGHT_SCALAR * distance + 10
            # Add proximity weight
            i = 0
            for old_detection_radian in yaw_radians:
                if abs(old_detection_radian - yaw_radian) < CLUMP_CLOSENESS_TOLERENCES and abs(pitch_radians[i] - pitch_radian) < CLUMP_CLOSENESS_TOLERENCES:
                    # Yaw and distance are  close, exclude value
                    weight += 1
                i+=1

            # Filter double detections
            i = 0
            keep = True
            for old_detection_radian in yaw_radians:
                if abs(old_detection_radian - yaw_radian) < DOUBLE_DETECTION_CLOSENESS_TOLERENCES and abs(pitch_radians[i] - pitch_radian) < DOUBLE_DETECTION_CLOSENESS_TOLERENCES:
                    # Yaw and distance are  close, exclude value
                    keep = False
                i+=1
            if keep:
                yaw_radians.append(yaw_radian)
                pitch_radians.append(pitch_radian)
                distances.append(distance)
                weights.append(weight)
            
        table.putNumberArray("yaw_radians", yaw_radians)
        table.putNumberArray("distance", distances)
        table.putNumberArray("weights", weights)
        print(f"Filtered: {len(yaw_radians)}, Yaw Radians: {yaw_radians}, Weights: {weights}")

        annotatedFrame = results.plot()
        cv2.imshow("YOLO Inference", annotatedFrame)
    
        if cv2.waitKey(1) == ord('q'):
            print("Exiting Program...")
            break

cap.release()
cv2.destroyAllWindows()