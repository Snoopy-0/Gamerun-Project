# Sports Video AI Analysis Tool

## Overview
This project is designed to assist coaches and sports enthusiasts in analyzing sports videos efficiently. It features a Python-based program that identifies key frames in a video and extracts them for AI-based analysis. The goal is to enable performance assessment, strategy evaluation, and actionable insights from sports footage.

---

## Features
- **Key Frame Detection:** Automatically identifies the most significant frames in a video.
- **Frame Extraction:** Extracts identified frames for further analysis.
- **AI Integration:** Prepares frames for AI-driven insights, such as player performance analysis or tactical review.
- **API Implimentation:** Allows for use with an AWS S3 buckets for cloud storage. 

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Gamerun-Project.git
   cd Gamerun-Project

## Requirements:
   - pip install opencv-python
   - pip install numpy
   - pip install flask boto3
   - yolov4.cfg, yolov4.weights, & coco.names are necessary and can be downloaded from the YOLOv4 github repository
        - yolov4 files can be found here: https://github.com/kiyoshiiriemon/yolov4_darknet these files are necessary for the model to be configured in our program.

## How to run
   - First, clone our github repository and get the required yolov4 files
   - Download and configure AWS using AWS configure in terminal. input the public and secret keys and set region
   - Next, run the frameExtractAPI.py file
   - Finally, run either fencingAPIcall.py or soccerAPIcall.py. results will be inside aws s3 bucket.
