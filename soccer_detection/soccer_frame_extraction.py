import cv2
import os
import numpy as np
from aws_s3_utility import download_from_s3

#def analyze_video_from_s3(s3_key):
#    local_video_path = 'temp_video.mp4'
#    download_from_s3(s3_key, local_video_path)
    
    # Call frame_extract function here
#    extract_key_frames(local_video_path)

#def download_from_s3(s3_key, local_path):
#    s3 = boto3.client('s3')
#    try:
#        s3.download_file(S3_BUCKET_NAME, s3_key, local_path)
#        print(f"File {s3_key} downloaded to {local_path}")
#    except NoCredentialsError:
#        print("Credentials not available")

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load YOLO model and class labels
def load_yolo_model():
    config_path = "yolov4.cfg"  # Path to YOLO config file
    weights_path = "yolov4.weights"  # Path to YOLO weights file
    class_labels_path = "coco.names"  # Path to class labels

    with open(class_labels_path, 'r') as f:
        class_labels = f.read().strip().split('\n')

    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    return net, class_labels

# Compute edge map using Canny edge detector
def compute_edge_map(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_frame, threshold1=40, threshold2=100)
    return edges

# Compare edge maps using normalized difference
def compare_edge_maps(edges1, edges2):
    diff = cv2.absdiff(edges1, edges2)
    diff_percentage = np.sum(diff > 0) / edges1.size
    return diff_percentage

# Detect objects in a frame
def detect_objects(frame, net, class_labels, confidence_threshold=0.5, nms_threshold=0.4):
    #setting up frame to be used with YOLOv4
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getUnconnectedOutLayersNames()
    outputs = net.forward(layer_names)

    h, w = frame.shape[:2]
    boxes, confidences, class_ids = [], [], []

    #processes each detection in the output
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            #filtering the bounding boxes to ensure they stay within the threshold
            if confidence > confidence_threshold:
                box = detection[0:4] * np.array([w, h, w, h])
                (center_x, center_y, width, height) = box.astype("int")
                x = int(center_x - (width / 2))
                y = int(center_y - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    result = []
    for i in indices.flatten():
        result.append({
            "box": boxes[i],
            "confidence": confidences[i],
            "class_id": class_ids[i],
            "label": class_labels[class_ids[i]],
        })
    return result

# Main function to extract key frames
def extract_key_frames_with_detection(video_path, output_dir, net, class_labels, threshold=0.05):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    frame_count = 0
    saved_count = 0
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Unable to read the first frame.")
        cap.release()
        return

    prev_edges = compute_edge_map(prev_frame)

    while cap.isOpened():
        ret, curr_frame = cap.read()
        if not ret:
            break

        curr_edges = compute_edge_map(curr_frame)
        difference = compare_edge_maps(prev_edges, curr_edges)

        detections = detect_objects(curr_frame, net, class_labels)
        soccer_ball_detected = any(d["label"] == "sports ball" for d in detections)
        players_detected = any(d["label"] == "person" for d in detections)

        if (soccer_ball_detected or players_detected) and difference >= threshold:
            frame_name = f"{output_dir}/frame_{frame_count:04d}.jpg"
            cv2.imwrite(frame_name, curr_frame)
            print(f"Saved {frame_name} with edge difference {difference:.2f}")
            saved_count += 1
            prev_edges = curr_edges  # Update previous edges only when saving a frame

        frame_count += 1

    cap.release()
    print(f"Processing complete. {saved_count} frames saved to {output_dir}.")

# Parameters
video_path = "input_videos/compress-scoccer_analysis_1.mp4"
output_dir = "value_frames"
threshold = 0.035

# Load YOLO model
net, class_labels = load_yolo_model()

# Extract key frames
extract_key_frames_with_detection(video_path, output_dir, net, class_labels, threshold)