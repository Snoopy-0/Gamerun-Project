from flask import Flask, request, jsonify
import os
import threading
from FrameExtractFencing import download_video_from_s3, extract_key_frames_to_s3
from FrameExtractSoccer import download_video_from_s3 as download_soccer_video_from_s3
from FrameExtractSoccer import load_yolo_model, extract_key_frames_to_s3_with_detection

app = Flask(__name__)

# API to process fencing videos
@app.route('/fencing', methods=['POST'])
def process_fencing():
    try:
        data = request.json
        bucket_name = data['bucket_name']
        s3_video_key = data['s3_video_key']
        output_dir = data['output_dir']
        s3_folder = data['s3_folder']
        threshold = data.get('threshold', 0.035)

        # Local file path to download the video
        local_video_path = os.path.join(output_dir, os.path.basename(s3_video_key))

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Download video from S3
        download_video_from_s3(bucket_name, s3_video_key, local_video_path)

        # Run frame extraction
        threading.Thread(target=extract_key_frames_to_s3, args=(local_video_path, output_dir, bucket_name, s3_folder, threshold)).start()

        return jsonify({"message": "Fencing video processing started."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# API to process soccer videos
@app.route('/soccer', methods=['POST'])
def process_soccer():
    try:
        data = request.json
        bucket_name = data['bucket_name']
        s3_video_key = data['s3_video_key']
        output_dir = data['output_dir']
        s3_folder = data['s3_folder']
        threshold = data.get('threshold', 0.035)

        # Local file path to download the video
        local_video_path = os.path.join(output_dir, os.path.basename(s3_video_key))

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Download video from S3
        download_soccer_video_from_s3(bucket_name, s3_video_key, local_video_path)

        # Load YOLO model
        net, class_labels = load_yolo_model()

        # Run frame extraction with object detection
        threading.Thread(target=extract_key_frames_to_s3_with_detection, args=(local_video_path, output_dir, bucket_name, s3_folder, net, class_labels, threshold)).start()

        return jsonify({"message": "Soccer video processing started."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

