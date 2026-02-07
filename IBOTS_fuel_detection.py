import cv2 # uv add open-cv
from ultralytics import YOLO # uv add ultralytics
import ntcore # uv add pyntcore https://robotpy.readthedocs.io/en/2023.5/install/pyntcore.html
import time
import math

# ======
# Setup Constants
model_input_size = 352
camera_fov = 1.195551   # camera_fov should be in radians
camera_input_index = 1
model_path = "/home/josh/Documents/ibots/1_29_26_full_integer_quant_edgetpu.tflite"
mount_height = 0.2 # The camera mount height should be in meters

# Set up network table
networktable_instance = ntcore.NetworkTableInstance.getDefault()
table = networktable_instance.getTable("fuelCV")

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
    success, rawframe = cap.read()

    if success:
        frame = cv2.resize(rawframe, (model_input_size, model_input_size))

        results = model(frame, conf=0.5)[0]

        #print(results)

        boxes = results.boxes[results.boxes.conf > 0.3]

        for b in boxes:
            tx, ty, tw, th = b.xywh[0]
            x = tx.item()
            y = ty.item()
            w = tw.item()
            h = th.item()
            # target position is from - half of the camera fov to + half the camera fov
            target_position_x = (x + (w/2) - (model_input_size/2))
            target_position_x = (x + (h/2) - (model_input_size/2))
            # Scale pixel position of bounding box to radians. Radians should be positive to the left and negative to the right
            yaw_radians = -2*target_position_x/model_input_size*camera_fov
            pitch_radians = -2*target_position_x/model_input_size*camera_fov

            if h < 0:
                target_on_ground = True
            else:
                target_on_ground = False

            distance = mount_height/math.tan(pitch_radians)

            print(f"{yaw_radians} {pitch_radians}")
            table.putNumber("yaw_radians", yaw_radians)
            table.putNumber("pitch_radians", pitch_radians)
            table.putBoolean("target_on_ground", target_on_ground)
            table.putNumber("distance", distance)
            
        #annotatedFrame = results.plot()
        #cv2.imshow("YOLO Inference", annotatedFrame)
        
        if cv2.waitKey(1) == ord('q'):
            print("Let's get out of here...")
            break
    #break
cap.release()
#cv2.destroyAllWindows()