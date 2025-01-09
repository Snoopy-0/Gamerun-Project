import cv2
import os
import numpy as np

# Compute edge map using Canny edge detector
def compute_edge_map(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_frame, threshold1=50, threshold2=150)
    return edges

# Compare edge maps using normalized difference
def compare_edge_maps(edges1, edges2):
    diff = cv2.absdiff(edges1, edges2)

    diff_percentage = np.sum(diff > 0) / edges1.size
    return diff_percentage

#main function to extract key frames based on edge differences
def extract_key_frames_edge(video_path, output_dir, threshold=0.05):
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

        print(f"Frame {frame_count}: Edge difference = {difference:.5f}")  # Debugging

        if difference >= threshold:
            frame_name = f"{output_dir}/frame_{frame_count:04d}.jpg"
            cv2.imwrite(frame_name, curr_frame)
            print(f"Saved {frame_name} with edge difference {difference:.2f}")
            saved_count += 1
            prev_edges = curr_edges  # Update previous edges only when saving a frame

        frame_count += 1

    cap.release()
    print(f"Processing complete. {saved_count} frames saved to {output_dir}.")

# Parameters
video_path = "input_videos\Fencing_Part_1_compress.mp4"  
output_dir = "value_frames"  
threshold = 0.08  # Set a percentage threshold for edge differences (higher = less saved frames/ lower = more saved frames)

extract_key_frames_edge(video_path, output_dir, threshold)
