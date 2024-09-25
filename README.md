# Clothing Size Measurement System: Classification, Keypoint Detection, and Scale Calibration

## Project Overview

This project aims to provide an automated system for measuring clothing dimensions in online marketplaces. Using a combination of image classification, keypoint detection, and scale calibration, the system generates accurate clothing measurements in millimeters, improving customer confidence and enhancing the online shopping experience.

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
  [Clothing dataset]([https://github.com/username/repo-name](https://github.com/alexeygrigorev/clothing-dataset))

## Model Architecture

The system employs a deep learning model built with **MobileNetV2** as a base, combined with additional layers for keypoint detection and classification:

