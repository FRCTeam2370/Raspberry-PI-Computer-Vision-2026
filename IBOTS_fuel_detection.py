import cv2 # uv add open-cv
from ultralytics import YOLO # uv add ultralytics
import ntcore # uv add pyntcore https://robotpy.readthedocs.io/en/2023.5/install/pyntcore.html
import math
import time

# ======
# Setup Constants
# ======
MODEL_INPUT_SIZE = 352
MODEL_DETECTION_CONF_THRESHOLD = 0.6
CAMERA_FOV_HORIZONTAL = 0.6981   # CAMERA_FOV_HORIZONTAL should be in radians
CAMERA_FOV_VERTICAL = 0.4363   # CAMERA_FOV_VERTICAL should be in radians
CAMERA_INPUT_INDEX = 1
MODEL_PATH = "/home/josh/Documents/ibots/2_16_26_last_full_integer_quant_edgetpu.tflite"
DOUBLE_DETECTION_CLOSENESS_TOLERENCES = 0.01
CLUMP_CLOSENESS_TOLERENCES = 0.35 # CLUMP_CLOSENESS_TOLERENCES should be in meters
CLUMP_WEIGHT_MULTIPLIER = 1.3
DISTANCE_WEIGHT_SCALAR = 1
BALL_DIAMETER = 0.1501
PITCH_WHERE_TARGET_IS_NOT_ON_GROUND = -0.01 # Self explanatory (oh, and its in radians)

# Set up NetworkTable
inst = ntcore.NetworkTableInstance.getDefault()
inst.startClient4("Raspberry Pi")
inst.setServerTeam(2370)
table = inst.getTable("fuelCV") 

# Set up CV model
loaded = False
while not loaded:
    try:
        model = YOLO(MODEL_PATH, task="detect")
        loaded = True
    except ValueError as e:
        print(f"Loading model failed: {e}")

# Connect to camera
connected_to_camera = False

while not connected_to_camera:
    try:
        cap = cv2.VideoCapture(0)

        if cap.isOpened():
            connected_to_camera = True
    except:
        connected_to_camera = False

def getBallPos(camera_dir, robot_dir, distance_to_ball, x_init, y_init):
    target_dir = robot_dir + camera_dir

    x_offset = math.cos(target_dir)*distance_to_ball
    y_offset = math.sin(target_dir)*distance_to_ball

    x_pos = x_init + x_offset
    y_pos = y_init + y_offset

    return [x_pos, y_pos]

while cap.isOpened():
    success, rawframe = cap.read()

    if success:
        frame = cv2.resize(rawframe, (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE))
        results = model(frame, conf=MODEL_DETECTION_CONF_THRESHOLD)[0]

        yaw_radians = []
        pitch_radians = []
        distances = []
        weights = []

        # For Weight Annotations:
        image_x = []
        image_y = []

        #print(results)

        boxes = results.boxes[results.boxes.conf > MODEL_DETECTION_CONF_THRESHOLD]
        table.putNumber("number_of_fuel", boxes.__len__())

        # Collect Camera Field Position from Network Tables Format: x,y,rotation (rad)
        robot_position = [0.0,0.0,0.0]
        robot_position = table.getEntry("Camera Pose").getDoubleArray(robot_position)
        print(robot_position)

        final_boxes = []
        ball_positions_x = []       
        ball_positions_y = []

        for b in boxes:
            tx, ty, tw, th = b.xywh[0]
            x = tx.item()
            y = ty.item()
            w = tw.item()
            h = th.item()

            # target position is from - half of the camera fov to + half the camera fov
            target_position_x = (x + (w/2) - (MODEL_INPUT_SIZE/2))# + CAMERA_CENTER_PIZEL_OFFSET
            target_position_y = (y + (h/2) - (MODEL_INPUT_SIZE/2))

            # Scale pixel position of bounding box to radians. Radians should be positive to the left and negative to the right
            yaw_radian = (-2*target_position_x/MODEL_INPUT_SIZE*CAMERA_FOV_HORIZONTAL)
            pitch_radian = (target_position_y/MODEL_INPUT_SIZE*CAMERA_FOV_VERTICAL)

            # Calculate distance based on bounding box lengths
            distance_height = (MODEL_INPUT_SIZE/h * BALL_DIAMETER / 2) / math.sin(CAMERA_FOV_VERTICAL / 2)/2
            distance_width = (MODEL_INPUT_SIZE/w * BALL_DIAMETER / 2) / math.sin(CAMERA_FOV_HORIZONTAL / 2)/2
            distance = round(distance_height, 2)
            #print(f"Pitch: {pitch_radian}, Dist: {distance}")

            # Calculate ball position x,y
            ball_position = getBallPos(yaw_radian, robot_position[2], distance, robot_position[0], robot_position[1])

            # Calculate distance weight
            weight = -DISTANCE_WEIGHT_SCALAR * distance + 10

            # Add proximity weight
            old_ball_positions_x = ball_positions_x.copy()
            old_ball_positions_y = ball_positions_y.copy()
            for i in reversed(range(len(weights))):
                if abs(old_ball_positions_x[i] - ball_position[0]) < CLUMP_CLOSENESS_TOLERENCES and abs(old_ball_positions_y[i] - ball_position[1]) < CLUMP_CLOSENESS_TOLERENCES:
                    # balls are close, clump datapoints
                    ball_position[0] = (ball_positions_x[i] + ball_position[0])/2
                    ball_position[1] = (ball_positions_y[i] + ball_position[1])/2
                    weight = weight * CLUMP_WEIGHT_MULTIPLIER

                    yaw_radians.pop(i)
                    pitch_radians.pop(i)
                    distances.pop(i)
                    weights.pop(i)
                    ball_positions_y.pop(i)
                    ball_positions_x.pop(i)

                    # For Weight Annotatins
                    x = max((image_x[i], x), key=abs)
                    y = max((image_y[i], y), key=abs)
                    image_x.pop(i)
                    image_y.pop(i)

            # Filter double detections
            i = 0
            keep = True
            for old_distances in yaw_radians:
                if abs(old_distances - yaw_radian) < DOUBLE_DETECTION_CLOSENESS_TOLERENCES and abs(pitch_radians[i] - pitch_radian) < DOUBLE_DETECTION_CLOSENESS_TOLERENCES:
                    keep = False
                i+=1
            # Filter out fuel not on ground
            if pitch_radian < PITCH_WHERE_TARGET_IS_NOT_ON_GROUND:
                keep = False

            if keep:
                yaw_radians.append(yaw_radian)
                pitch_radians.append(pitch_radian)
                distances.append(distance)
                weights.append(round(weight,2))
                # For Weight Annotations:
                image_x.append(x)
                image_y.append(y)
                final_boxes.append(b)
                ball_positions_x.append(ball_position[0])
                ball_positions_y.append(ball_position[1])

        # Order by weight
        if len(weights) > 0:
            sortedValues = sorted(zip(weights, ball_positions_x, ball_positions_y), reverse=True)
            weights, ball_positions_x, ball_positions_y = zip(*sortedValues)
            print(weights)
        table.putNumberArray("yaw_radians", yaw_radians)
        table.putNumberArray("distance", distances)
        table.putNumberArray("weights", weights)
        table.putNumberArray("ball_position_x", ball_positions_x)
        table.putNumberArray("ball_position_y", ball_positions_y)

        # Show Feed
        results.boxes = final_boxes
        print(f"Filtered: {len(yaw_radians)}, Yaw Radians: {yaw_radians}, Weights: {weights}")
        annotatedFrame = results.plot()
        # Annotate weights
        for i in range(len(yaw_radians)):
            x,y = image_x[i], image_y[i]
            annotation = f"Weight: {weights[i]}"
            #annotation = f"Distance: {distances[i]}"
            annotation = f"Radians: {pitch_radians[i]}"
            cv2.putText(annotatedFrame, annotation, (round(x), round(y)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255 , 0, 0), 1, cv2.LINE_AA)
        cv2.imshow("Processed Feed", annotatedFrame)
        if cv2.waitKey(1) == ord('q'):
            print("Exiting Program...")
            break

cap.release()
cv2.destroyAllWindows()