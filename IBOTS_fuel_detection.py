import cv2 # uv add open-cv
from ultralytics import YOLO # uv add ultralytics
import ntcore # uv add pyntcore https://robotpy.readthedocs.io/en/2023.5/install/pyntcore.html
#from networktables import NetworkTables
import time
import math

# ======
# Setup Constants
model_input_size = 352
camera_fov = 1.195551   # camera_fov should be in radians
camera_input_index = 1
model_path = "/home/josh/Documents/ibots/2_12_26.2_full_integer_quant_edgetpu.tflite"
mount_height = 0.2 # The camera mount height should be in meters

# Set up NetworkTable
# Get the default NetworkTable instance
inst = ntcore.NetworkTableInstance.getDefault()
inst.startClient4("Raspberry Pi")
inst.setServerTeam(2370)
table = inst.getTable("fuelCV") 

# Set up video feed + CV model
model = YOLO(model_path, task="detect")

connected_to_camera = False

while not connected_to_camera:
    try:
        cap = cv2.VideoCapture(0)

        if cap.isOpened():
            connected_to_camera = True
    except:
        connected_to_camera = False

while cap.isOpened():
    #for _ in range(5):
    success, rawframe = cap.read()

    if success:
        frame = cv2.resize(rawframe, (model_input_size, model_input_size))

        results = model(frame, conf=0.5)[0]

        yaw_radians = []
        pitch_radians = []
        distances = []

        #print(results)

        boxes = results.boxes[results.boxes.conf > 0.6]
        table.putNumber("number_of_fuel", boxes.__len__())

        for b in boxes:
            tx, ty, tw, th = b.xywh[0]
            x = tx.item()
            y = ty.item()
            w = tw.item()
            h = th.item()

            distance_height = 60/h
            distance_width = 34.4/w
            print(f"Distance: Width: {distance_width}, Height: {distance_height}")
            if distance_width > distance_height:
                distances.append(distance_width)
            else:
                distances.append(distance_height)

            # target position is from - half of the camera fov to + half the camera fov
            target_position_x = (x + (w/2) - (model_input_size/2))
            target_position_x = (x + (h/2) - (model_input_size/2))
            # Scale pixel position of bounding box to radians. Radians should be positive to the left and negative to the right
            yaw_radians.append(-2*target_position_x/model_input_size*camera_fov)
            pitch_radians.append(-2*target_position_x/model_input_size*camera_fov)

            table.putNumberArray("yaw_radians", yaw_radians)
            table.putNumberArray("pitch_radians", pitch_radians)
            table.putNumberArray("distance", distances)
            
        #annotatedFrame = results.plot()
        #cv2.imshow("YOLO Inference", annotatedFrame)
    
        #if cv2.waitKey(1) == ord('q'):
            #print("Exiting Program...")
            #break
cap.release()
cv2.destroyAllWindows()