import cv2
import os
import boto3
import numpy as np

# Function to download the video from S3
def download_video_from_s3(bucket_name, s3_key, local_file_path):
    s3 = boto3.client('s3')
    try:
        print(f"Downloading {s3_key} from bucket {bucket_name}...")
        s3.download_file(bucket_name, s3_key, local_file_path)
        print("Download complete!")
    except Exception as e:
        print(f"Error downloading file: {e}")
        raise

# Upload a file to an S3 bucket
def upload_to_s3(local_file_path, bucket_name, s3_file_path):
    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(local_file_path, bucket_name, s3_file_path)
        print(f"Uploaded {local_file_path} to s3://{bucket_name}/{s3_file_path}")
    except Exception as e:
        print(f"Error uploading {local_file_path} to S3: {e}")

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

#Logic for creating a dynamic threshold value based on video 
def compute_threshold(differences):
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)
    dynamic_threshold = mean_diff + ( .9 * std_diff) # Adjust the multiplier (e.g., 0.5) to tune sensitivity
    return dynamic_threshold

# Detect objects in a frame
def detect_objects(frame, net, class_labels, confidence_threshold=0.5, nms_threshold=0.4):
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getUnconnectedOutLayersNames()
    outputs = net.forward(layer_names)

    h, w = frame.shape[:2]
    boxes, confidences, class_ids = [], [], []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
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

# Main function to extract key frames and upload to S3
def extract_key_frames_to_s3_with_detection(video_path, output_dir, bucket_name, s3_folder, net, class_labels, sample_size = 50, update_interval = 100):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    frame_count = 0
    saved_count = 0
    differences = []

 
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Unable to read the first frame.")
        cap.release()
        return

    prev_edges = compute_edge_map(prev_frame)

#determine dynamic threshold based on first 50 frames
    print("calculating initial threshold...")
    for _ in range(sample_size):
        ret, curr_frame = cap.read()
        if not ret:
            break

        curr_edges = compute_edge_map(curr_frame)
        difference = compare_edge_maps(prev_edges, curr_edges)
        differences.append(difference)
    
    threshold = compute_threshold(differences)
    print (f"initial dynamic threshold {threshold: .4f}")

#reset frames to 0 before performing actual frame extraction
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, prev_frame = cap.read()
    prev_edges = compute_edge_map(prev_frame)
    differences = []

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
            frame_name = f"frame_{frame_count:04d}.jpg"
            local_path = os.path.join(output_dir, frame_name)
            s3_path = f"{s3_folder}/{frame_name}"

            cv2.imwrite(local_path, curr_frame)
            upload_to_s3(local_path, bucket_name, s3_path)
            saved_count += 1
            prev_edges = curr_edges

        if frame_count % update_interval == 0 and len(differences) > 10:
            threshold = compute_threshold(differences)
            print(f"Updated threshold at frame {frame_count}: {threshold: .4f}")
            differences = []

        frame_count += 1

    cap.release()
    print(f"Processing complete. {saved_count} frames saved to {output_dir}.")

def main():
    
    # Parameters
    local_video_path = "Soccer_Part_1.mp4"
    output_dir = "value_frames"
    bucket_name = "frame-storage-capstone-project"
    s3_video_key = "Soccer_Part_1.mp4"
    s3_folder = "soccer-key-frames"
    sample_size = 50
    update_interval = 100

    # Download the video from S3
    download_video_from_s3(bucket_name, s3_video_key, local_video_path)

    # Load YOLO model
    net, class_labels = load_yolo_model()

    # Extract frames and upload to S3
    extract_key_frames_to_s3_with_detection(local_video_path, output_dir, bucket_name, s3_folder, net, class_labels, sample_size, update_interval)

if __name__ == '__main__':
    main()
