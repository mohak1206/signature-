# 🖊️ Signature Detection with YOLOv8

> Automated signature presence/absence detection using state-of-the-art object detection — trained on 7,000+ images with 92%+ mAP50 accuracy.

---

## 📌 Overview

This project fine-tunes a **YOLOv8m** (medium) model to detect whether a signature is present or absent in document images. It is designed for use cases like automated form validation, document processing pipelines, and compliance checks.

---

## ✨ Key Results

| Metric         | Validation | Test     |
|----------------|------------|----------|
| **mAP50**      | 0.9106     | 0.9240   |
| **mAP50-95**   | 0.7276     | 0.7348   |
| **Precision**  | 0.9229     | 0.8639   |
| **Recall**     | 0.8313     | 0.9204   |

---

## 🗂️ Dataset

- **Source:** [Roboflow — signature-no-signature](https://roboflow.com/)
- **Total Images:** 7,004
- **Custom Split:**

| Split      | Images | Percentage |
|------------|--------|------------|
| Train      | 4,902  | 70%        |
| Validation | 1,401  | 20%        |
| Test       | 701    | 10%        |

---

## ⚙️ Setup

### 1. Clone & Install Dependencies

```bash
pip install roboflow ultralytics
```

### 2. Mount Google Drive (Colab)

```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.makedirs('/content/drive/MyDrive/signature_yolov8', exist_ok=True)
```

### 3. Download Dataset

```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("space-3jvwp").project("signature-no-signature")
version = project.version(1)
dataset = version.download("yolov8")
```

> ⚠️ Replace `YOUR_API_KEY` with your actual Roboflow API key.

### 4. Verify GPU

```python
import torch
print("GPU Available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU ⚠️")
```

---

## 🏋️ Training

```python
from ultralytics import YOLO

model = YOLO('yolov8m.pt')

model.train(
    data='path/to/data.yaml',
    epochs=50,
    imgsz=640,
    batch=8,
    project='signature_detection',
    name='run1',
    patience=10
)
```

**Training Configuration:**

| Parameter   | Value      |
|-------------|------------|
| Architecture | YOLOv8m   |
| Epochs      | 50         |
| Image Size  | 640 × 640  |
| Batch Size  | 8          |
| Hardware    | Tesla T4 (Google Colab) |

---

## 📊 Evaluation

```python
# Validate on validation set
metrics = model.val()

# Evaluate on test set
test_results = model.val(data='path/to/data.yaml', split='test')
```

---

## 🔍 Inference

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('best.pt')

# Run inference
results = model.predict('path/to/image.jpg', conf=0.5)

for r in results:
    print("Bounding Boxes:", r.boxes.xyxy)
    print("Confidence Scores:", r.boxes.conf)
    print("Class Labels:", r.boxes.cls)
    # r.save(filename='predicted_image.jpg')  # Save annotated image
```

---

## 💾 Saving the Model

```python
from google.colab import files
files.download('/content/runs/detect/train/weights/best.pt')
```

The best model weights (`best.pt`) are saved after training and can be downloaded for deployment or inference.

---

## 🚀 Future Work

- [ ] Experiment with larger model variants (`yolov8l`, `yolov8x`) for improved accuracy
- [ ] Apply advanced data augmentation to handle varied signature styles
- [ ] Handle edge cases: partial signatures, stamps, and handwritten annotations
- [ ] Build a REST API or web application for real-time document signature detection
- [ ] Explore model quantization for edge deployment

---

## 🛠️ Tech Stack

- [Roboflow](https://roboflow.com/)
- Google Colab (Tesla T4 GPU)

---

## 📄 License

This project is for research and educational purposes. Dataset usage is subject to [Roboflow's terms of service](https://roboflow.com/terms).
