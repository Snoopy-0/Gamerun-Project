import requests
import datetime
import random

def randomNumber():
    return random.randint(10000, 99999)

now = datetime.datetime.now()
randNum = randomNumber()

url = "http://127.0.0.1:5000/fencing"
data = {
    "bucket_name": "frame-storage-capstone-project",
    "s3_video_key": "Fencing_Part_1.mp4",
    "output_dir": "value_frames",
    "s3_folder" : f"{randNum}--{now.strftime('%Y-%m-%d_%H:%M:%S')}",
    "threshold" : 0.085
}

response = requests.post(url, json=data)
print(response.json())
