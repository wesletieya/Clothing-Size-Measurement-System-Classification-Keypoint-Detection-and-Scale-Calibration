# Clothing Size Measurement System: Classification, Keypoint Detection, and Scale Calibration

## Project Overview

This project aims to provide an automated system for measuring clothing dimensions in online marketplaces. Using a combination of image classification, keypoint detection, and scale calibration, the system generates clothing measurements in millimeters, improving customer confidence and enhancing the online shopping experience.

## Features

- **Clothing Classification**: Automatically classifies the type of clothing (e.g., upper body, lower body, full body).
- **Keypoint Detection**: Detects key points (landmarks) on clothing, such as collar width, shoulder width, and waist width.
- **Scale Calibration**: Uses a reference object (e.g., a card) of known dimensions to determine real-world measurements from pixel data.
- **Real-World Size Calculation**: Converts pixel measurements to millimeters, providing users with accurate clothing dimensions.

## Dataset

We used several datasets, including the **Fashion Landmark Detection in the Wild (FLD)** dataset and the **Clothes Dataset** from Kaggle, which includes over 5,000 images across more than 20 clothing categories.

- **Fashion Landmark Detection in the Wild (FLD)**: Contains labeled landmarks for different clothing types, enabling precise measurements.
  [fashion-landmarks](https://github.com/liuziwei7/fashion-landmarks)
- **Clothes Dataset (Kaggle)**: A diverse dataset with over 5,000 images for model training and validation.
  [Clothing dataset](https://github.com/alexeygrigorev/clothing-dataset)

# Project Pipeline: Clothing Size Measurement System

## 1. Data Collection and Preprocessing

- **Image Data**: Collected from datasets such as the Fashion Landmark Detection in the Wild and the Clothes Dataset from Kaggle.
- **Labeling**: Images are labeled with clothing types (e.g., upper body, lower body, full body) and keypoints (e.g., collar, shoulder, waist, hem).
- **Data Augmentation**: Utilized ImageDataGenerators to augment the dataset with transformations like rotation, flipping, and scaling to enhance model generalization.

## 2. Clothing Classification Model

- **Base Model**: MobileNetV2 pre-trained on ImageNet is employed for feature extraction.
- **Custom Layers**: Added layers (Dense, Dropout) for classification, predicting the clothing type (e.g., upper body, lower body, full body).
- **Output**: Model outputs a class label representing the type of clothing.

## 3. Keypoint Detection Model

- **Custom CNN Model**: Built a Sequential model for keypoint detection, identifying crucial landmarks on clothing items.
- **Keypoint Coordinates**: Outputs pixel coordinates for predefined keypoints.

## 4. Scale Calibration

- **Reference Object Detection**: Detects a reference object (e.g., a card) in the image, with known real-world dimensions.
- **Zoom Factor Calculation**: Calculates the scale factor between the pixel dimensions of the reference object and its real-world dimensions.
- **Conversion to Real-World Measurements**: Converts pixel-based distances between keypoints into millimeters using the calculated zoom factor.

## 5. Measurement Calculation

- **Keypoint Distances**: Measures distances between detected landmarks (collar width, shoulder width, waist width) in pixels.
- **Real-World Measurements**: Converts these distances into millimeters using the scale calibration data.

## 6. Additional Techniques

- **Pose ResNet-50**: Experimented with Pose ResNet-50 for landmark detection. The code for this approach is included in the repository for reference.
- **Clothing Extraction and Landmark Detection**: Used YOLO for bounding box detection to extract clothing from images. Subsequently, a pre-trained HRNet model is utilized for landmark detection on the extracted clothing regions.

## 7. Pretrained Models

- **Pretrained Models**: All pretrained models used in this project are available for download from the following Google Drive link: [Download Pretrained Models]([https://drive.google.com/drive/folders/your-folder-id](https://drive.google.com/drive/folders/1Ku9bNEhJeYNyXNBoHCk89ygAAfkMCeXn?usp=drive_link). 

