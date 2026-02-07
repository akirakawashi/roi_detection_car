import cv2
from ultralytics import YOLO
from shapely.geometry import Polygon, Point, box
import numpy as np

MODEL_PATH = '/home/estaid/dev/prediction_Car/yolov8x.pt'
VIDEO_INPUT_PATH = '/home/estaid/dev/prediction_Car/data/cvtest.avi'
VIDEO_OUTPUT_PATH = 'output_video.mp4'

# ROI
TOP_LEFT = (0, 440)
BOTTOM_LEFT = (0, 1514)
BOTTOM_RIGHT = (1677, 1514)
TOP_RIGHT = (1677, 440)
ROI_POLYGON = Polygon([TOP_LEFT, BOTTOM_LEFT, BOTTOM_RIGHT, TOP_RIGHT])

MODEL = YOLO(MODEL_PATH)

# classes to detect
CLASSES = (2, 7)

def draw_filtered_boxes(frame, filtered_results):
    """
    Draw bounding boxes and labels on the frame for filtered results.
    :param frame: source frame
    :param filtered_results: list of objects that passed filtering
    """
    for r in filtered_results:
        x1, y1, x2, y2 = map(int, r[:4])
        conf = round(float(r[4]), 2)
        cls = int(r[5])
        
        color = (0, 255, 0)  # green
        thickness = 3
        font_scale = 2
        text_color = (0, 0, 255) #red
        
        # Draw bounding box around the object
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Put class label and confidence
        label = f"{cls}: {conf}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)

def process_frame(frame):
    """
    Process a single frame: run detection, filter results by ROI, and draw detected objects.
    :param frame: input frame
    :return: frame with drawn objects
    """
    global MODEL, CLASSES
    
    results = MODEL.predict(frame, classes=CLASSES)[0]
    
    # Filter objects by region of interest (ROI)
    filtered_results = []
    for r in results.boxes.data.tolist():
        x1, y1, x2, y2, conf, cls = map(float, r[:6])
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        # Check if the bounding box center is inside the ROI
        if ROI_POLYGON.contains(Point(center_x, center_y)):
            filtered_results.append(r)
    
    # Draw bounding boxes and labels on the frame
    draw_filtered_boxes(frame, filtered_results)
    
    return frame

if __name__ == "__main__":
    cap = cv2.VideoCapture(VIDEO_INPUT_PATH)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create a VideoWriter object for output
    out = cv2.VideoWriter(
        VIDEO_OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        processed_frame = process_frame(frame)
        out.write(processed_frame)

    cap.release()
    out.release()