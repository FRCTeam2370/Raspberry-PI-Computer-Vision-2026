import cv2 # uv add open-cv
from ultralytics import YOLO # uv add ultralytics
import supervision as sv
import ntcore # uv add pyntcore https://robotpy.readthedocs.io/en/2023.5/install/pyntcore.html
import math

# ======
# Setup Constants
# ======
MODEL_INPUT_SIZE = 352
MODEL_DETECTION_CONF_THRESHOLD = 0.6
CAMERA_FOV_HORIZONTAL = 0.6981   # CAMERA_FOV_HORIZONTAL should be in radians
CAMERA_FOV_VERTICAL = 0.4363   # CAMERA_FOV_VERTICAL should be in radians
CAMERA_INPUT_INDEX = 1
MODEL_PATH = "/home/josh/Documents/ibots/2_16_26_.2_full_integer_quant_edgetpu.tflite"
DOUBLE_DETECTION_CLOSENESS_TOLERENCES = 0.01
CLUMP_CLOSENESS_TOLERENCES = 0.35 # CLUMP_CLOSENESS_TOLERENCES should be in meters
CLUMP_WEIGHT_MULTIPLIER = 1.3
DISTANCE_WEIGHT_SCALAR = 1
BALL_DIAMETER = 0.1501

# Set up CV model
model = YOLO(MODEL_PATH, task="detect")
tracker = sv.ByteTrack()
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

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
        results = model(frame, conf=MODEL_DETECTION_CONF_THRESHOLD)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = tracker.update_with_detections(detections)

        labels = [
            model.names[class_id]
            for class_id
            in detections.class_id
        ]

        annotated_image = bounding_box_annotator.annotate(
            scene=frame, detections=detections)
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections, labels=labels)

        cv2.imshow("Supervision Inference", annotated_image)
        #sv.plot_image(annotated_image)

        if cv2.waitKey(1) == ord('q'):
            print("Exiting Program...")
            break

cap.release()
cv2.destroyAllWindows()