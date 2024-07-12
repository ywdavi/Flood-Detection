# **Semantic Segmentation on the FloodNet Dataset**
## *Advanced Computational Techniques for Big Imaging and Signal Data*

**Name and Surname:** Davide Vettore  
**ID:** 868855  
**Date:** July 15, 2024

> _Natural disasters are becoming more frequent and severe, threatening human health and infrastructure. Accurate and timely information is crucial for effective disaster management. Small unmanned aerial systems (UAS) with affordable sensors can quickly collect thousands of images, even in difficult-to-reach areas, which helps in rapid response and recovery. However, analyzing these large datasets to extract useful information remains a significant challenge._

The **FloodNet Dataset** provides high-resolution UAS imagery with detailed semantic annotations of damages collected after **Hurricane Harvey**. The whole dataset has 2343 images, divided into training (60%), validation (20%), and test (20%) sets. The semantic segmentation labels include: Background, Building Flooded, Building Non-Flooded, Road Flooded, Road Non-Flooded, Water, Tree, Vehicle, Pool and Grass. The **goal** is to create and train a strong model for semantic segmentation. This model will be used in a dashboard to detect flood damages in real-time.

**_Content:_**
- [6397.jpg](https://github.com/ywdavi/FloodDetection/blob/main/6397.jpg) and [7606.jpg](https://github.com/ywdavi/FloodDetection/blob/main/7606.jpg): Placeholder images. 
- [FPN](https://github.com/ywdavi/FloodDetection/blob/main/FPN) and [Unet](https://github.com/ywdavi/FloodDetection/blob/main/Unet): Segmentation models (`.pth`).
- [FloodDetection.py](https://github.com/ywdavi/FloodDetection/blob/main/FloodDetection.py): Streamlit dashboard.
- [requirements.txt](https://github.com/ywdavi/FloodDetection/blob/main/requirements.txt): Required libraries for the dashboard.
- [FloodDetection_Davide_Vettore.ipynb](https://github.com/ywdavi/FloodDetection/blob/main/FloodDetection_Davide_Vettore.ipynb): Training of the segmentation models.

**_References:_**
- [FloodNet Dataset Repository](https://github.com/BinaLab/FloodNet-Supervised_v1.0)
- [FloodNet Paper](https://ieeexplore.ieee.org/document/9460988)
