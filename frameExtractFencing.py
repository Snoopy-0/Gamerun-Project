import cv2
import os
import boto3
import numpy as np

# Compute edge map using Canny edge detector
def compute_edge_map(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_frame, threshold1=80, threshold2=180)
    return edges

# Compare edge maps using normalized difference
def compare_edge_maps(edges1, edges2):
    diff = cv2.absdiff(edges1, edges2)
    diff_percentage = np.sum(diff > 0) / edges1.size
    return diff_percentage

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

#Logic for creating a dynamic threshold value based on video 
def compute_Threshold(differences):
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)
    dynamic_threshold = mean_diff + (1.2 * std_diff) # Adjust the multiplier (e.g., 0.5) to tune sensitivity
    return dynamic_threshold

#main function to extract key frames based on edge differences
def extract_key_frames_to_s3(video_path, output_dir, bucket_name, s3_folder, sample_size = 50, update_interval = 100):
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

    #dynamic threshold gets created here
    differences = []

    print("Calculating initial dynamic threshold...")

    #gets sample_size number of frames to calculate first dynamic threshold
    for _ in range(sample_size):
        ret, curr_frame = cap.read()
        if not ret:
            break

        curr_edges = compute_edge_map(curr_frame)
        difference = compare_edge_maps(prev_edges, curr_edges)
        differences.append(difference)
        prev_edges = curr_edges

    #makes the threshold the computed threshold, prints the threshold, then goes back to start of video.
    threshold = compute_Threshold(differences)
    print(f"Initial dynamic threshold: {threshold: .4f}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, prev_frame = cap.read()
    prev_edges = compute_edge_map(prev_frame)

    while cap.isOpened():
        ret, curr_frame = cap.read()
        if not ret:
            break

        curr_edges = compute_edge_map(curr_frame)
        difference = compare_edge_maps(prev_edges, curr_edges)

        if difference >= threshold:
            frame_name = f"frame_{frame_count:04d}.jpg"
            local_path = os.path.join(output_dir, frame_name)
            s3_path = f"{s3_folder}/{frame_name}"
            
            cv2.imwrite(local_path, curr_frame)
            upload_to_s3(local_path, bucket_name, s3_path)
            saved_count += 1
            prev_edges = curr_edges  # Update previous edges only when saving a frame

        frame_count += 1

    cap.release()
    print(f"Processing complete. {saved_count} frames saved to {output_dir}.")

def main():
    # Parameters 
    local_video_path = "XXXX.mp4"
    output_dir = "value_frames"
    bucket_name = "frame-storage-capstone-project"
    s3_video_key = "Fencing_Part_1.mp4"  # Replace with GameRun S3 bucket name
    s3_folder = "fencing-videos"  # S3 folder to save the frames  
    sample_size = 50
    update_interval = 100

    # Download the video from S3
    download_video_from_s3(bucket_name, s3_video_key, local_video_path)
    #Extract frames and Upload to S3
    extract_key_frames_to_s3(local_video_path, output_dir, bucket_name, s3_folder, sample_size, update_interval)

if __name__ == '__main__':
    main()
