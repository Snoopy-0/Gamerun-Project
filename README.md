# Sports Video AI Analysis Tool

## Overview
This project is designed to assist coaches and sports enthusiasts in analyzing sports videos efficiently. It fetures a python based API that allows for frame extraction of sports footage using object detection models such as canny and YOLOv4. These allow for accurate determination of zero value frames versus frames of value. API calls are sports specific, as different methods are used to detect value frames depending on sport. 

---

## Features
- **Key Frame Detection:** Automatically identifies the most significant frames in a video.
- **Frame Extraction:** Extracts identified frames for further analysis.
- **AI Integration:** Prepares frames for AI-driven insights, such as player performance analysis or tactical review.
- **API Implementation:** Supports AWS S3 for cloud storage and retrieval of processed data.

---

## Installation

### 1. Clone the repository:
```bash
git clone https://github.com/yourusername/Gamerun-Project.git
cd Gamerun-Project
```
### 2. Install dependencies:

Ensure you have Python installed. Install all required dependencies using:
```bash
pip install -r requirements.txt
```
The requirements.txt file includes:
opencv-python
numpy
flask
boto3

### 3. Download YOLOv4 Files:

The project requires specific YOLOv4 model files. Download the following from the YOLOv4 GitHub repository:

- [yolov4.cfg](https://github.com/kiyoshiiriemon/yolov4_darknet/blob/master/cfg/yolov4.cfg)
- [yolov4.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights)
- [coco.names](https://github.com/kiyoshiiriemon/yolov4_darknet/blob/master/data/coco.names)

These files can also be found here:
[YOLOv4 Darknet GitHub Repository](https://github.com/kiyoshiiriemon/yolov4_darknet)

Place them in the application directory within the project.
## How to Run
#### 1. Configure AWS:

Run the following command to configure AWS credentials:
```bash
aws configure
```
Enter the AWS Access Key, Secret Key, and Region when prompted.
#### 2. Start the Frame Extraction API:

Run:
```bash
python frameExtractAPI.py
```
This script is the API and will wait for calls to the API
#### 3. Make API Calls:

Run one of the following API scripts depending on the sport:
```bash
python fencingAPIcall.py
```
or
```bash
python soccerAPIcall.py
```
