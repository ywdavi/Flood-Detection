# Flood Detection Semantic Segmentation

## Description
This repository contains my project on semantic segmentation using the **[FloodNet Dataset](https://ieeexplore.ieee.org/document/9460988)**, which was completed in the second semester of my Master's program. The goal of the project is to build a robust model capable of detecting flood damages from aerial images captured after natural disasters. The trained model is integrated into a real-time **Streamlit** dashboard to assist in visualizing flood damage in various areas.

## Objectives
- Perform semantic segmentation on aerial images from the **FloodNet Dataset** to detect different types of damage.
- Train various state-of-the-art models such as **U-Net** and **PSP-Net** for segmentation.
- Develop a **Streamlit dashboard** for real-time flood damage detection and visualization.
  
## Dataset
The **FloodNet Dataset** provides high-resolution UAS imagery with detailed semantic annotations of damages collected after **Hurricane Harvey**. The dataset contains 2343 images split into training (60%), validation (20%), and test (20%) sets.

## Key Steps

### Data Preprocessing:
- Resized images and masks to 256x256 pixels and normalized RGB values between 0 and 1.
- Applied data augmentation techniques including horizontal and vertical flipping and random rotations to prevent overfitting.

### Model Training:
- Trained **U-Net** and **PSP-Net** models with **ResNet101** backbones for segmentation tasks.
- Employed **Dice Loss** to optimize the models, ensuring better overlap between predicted segmentation masks and ground truth.

### Model Evaluation:
- Both models were trained for 25 epochs and evaluated using metrics such as **Mean IoU**, **Mean Dice Score**, and **Accuracy**.
- **U-Net** achieved better segmentation results with an IoU of 57.57% and a Dice Score of 67.75%, while **PSP-Net** had a slightly lower performance.

## Results
- The **U-Net** model provided better overall performance, especially in segmenting flooded areas.
- Class-wise analysis showed challenges in detecting smaller objects like pools and vehicles, but flooded buildings and roads were identified correctly.
  
**Evaluation Metrics:**
| Model   | Mean IoU | Mean Dice | Mean Accuracy |
|---------|----------|-----------|---------------|
| U-Net   | 57.57%   | 67.75%    | 72.67%        |
| PSP-Net | 57.57%   | 62.86%    | 69.24%        |

## Dashboard
A **Streamlit** dashboard was developed to visualize the results in real time. Users can upload images and select between different segmentation models to view the predicted flood damage. The dashboard also includes tools to adjust transparency and overlay segmentations on the original images.

<div align="center">
	<img width="40%" alt="Screenshot 2024-10-18 at 22 02 16" src="https://github.com/user-attachments/assets/2817f235-b06b-4c30-bac5-7026d7abaec3">
	<img width="10%" alt="Screenshot 2024-10-18 at 22 15 17" src="https://github.com/user-attachments/assets/cd45f3d7-fd93-41c0-b87e-c0dd754d5cca">
	<img width="40%" alt="Screenshot 2024-10-18 at 22 05 06" src="https://github.com/user-attachments/assets/8043a3d1-12a6-46bd-9169-82c0cd295109">
</div>

## Content
- [6397.jpg](https://github.com/ywdavi/FloodDetection/blob/main/6397.jpg) and [7606.jpg](https://github.com/ywdavi/FloodDetection/blob/main/7606.jpg): Placeholder images. 
- [FPN](https://github.com/ywdavi/FloodDetection/blob/main/FPN) and [Unet](https://github.com/ywdavi/FloodDetection/blob/main/Unet): Segmentation models (.pth).
- [FloodDetection.py](https://github.com/ywdavi/FloodDetection/blob/main/FloodDetection.py): Streamlit dashboard.
- [requirements.txt](https://github.com/ywdavi/FloodDetection/blob/main/requirements.txt): Required libraries for the dashboard.
- [FloodDetection_Davide_Vettore.ipynb](https://github.com/ywdavi/FloodDetection/blob/main/FloodDetection_Davide_Vettore.ipynb): Training of the segmentation models.

## References
- [FloodNet Dataset Repository](https://github.com/BinaLab/FloodNet-Supervised_v1.0)
- [FloodNet Paper](https://ieeexplore.ieee.org/document/9460988)

_P.S. these models are trained on resnet34 backbone to not exceed 100MB github limit_"


