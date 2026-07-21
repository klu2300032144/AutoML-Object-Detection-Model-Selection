# AutoML-Based Object Detection Model Selection

## Overview

This project compares three deep learning object detection models:

- YOLOv8n
- YOLOv8s
- SSD (Single Shot MultiBox Detector)

The comparison is performed using the COCO128 dataset to determine which model provides the best detection performance.

---

## Features

- Object detection using YOLOv8n
- Object detection using YOLOv8s
- Object detection using SSD
- Model comparison using mAP50
- Performance graph generation

---

## Technologies Used

- Python
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- NumPy
- Matplotlib

---

## Dataset

COCO128 Dataset

The dataset contains images and corresponding labels for object detection.

---

## Project Structure

```
AutoML-Object-Detection-Model-Selection
│
├── coco128/
│   ├── images/
│   ├── labels/
│   ├── compare_models.py
│   ├── models_graph.py
│   ├── data.yaml
│   ├── LICENSE
│   └── README.txt
│
├── ssd_train.py
├── ssd_test.py
├── ssd_training_graph.py
└── .gitignore
```

---

## Results

| Model | mAP50 Accuracy |
|-------|---------------:|
| YOLOv8n | 0.59 |
| YOLOv8s | **0.70** |
| SSD | 0.40 |

### Best Performing Model

**YOLOv8s**

---

## Future Improvements

- Train on larger datasets
- Add Faster R-CNN comparison
- Add EfficientDet comparison
- Hyperparameter tuning

---

## Author

Bhagya Sri Voora
