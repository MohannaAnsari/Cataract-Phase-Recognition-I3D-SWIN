# Cataract Surgery Phase Recognition — I3D + Swin Transformer Classifiers

This repository implements two **clip-based** surgical phase recognition models:

### 1) **I3D / SlowR50 Classifier**  
A video backbone from PyTorchVideo with a custom classification head.

### 2) **Swin / VideoMAE Transformer Classifier**  
A Vision Transformer–based spatiotemporal model for clip-level classification  
(using the same training pipeline).

All models operate on short video clips (C×T×H×W) and predict one of **10 cataract phases**.

---

## 🧠 Models Included

### ✔ I3D (SlowR50)
File: `src/i3d_model.py`  
Features:
- Pretrained SlowR50 backbone  
- Dropout multilayer classification head  
- Label smoothing  
- Weighted cross-entropy  

### ✔ Swin Transformer / VideoMAE  
File: `notebooks/swin_model.ipynb`  
Features:
- Pretrained VideoMAE or Swin-v2 backbone  
- Frozen → gradual unfreezing strategy  
- Stronger spatial modeling  
- 32-frame clip input

Both models share the same:
- dataset (`dataset.py`)
- training logic (`train_clip.py`)
- evaluation logic (`evaluate_clip.py`)

---

## 📁 Repository Structure

- src/

  dataset.py
  
  eda.py
  
  i3d_main.py
  
  train_clip.py
  
  evaluate_clip.py
  
  i3d_model.py
  
  utils.py

- data/

    annotations.csv
  
    annotations_segments.csv
  
    phases.csv
  
    videos.csv
  
- notebooks/

  swin_model.ipynb
  
- reports/

  phase1_summary.md
  
  phase2_summary.md



---

## 🚀 Training the Swin Classifier

notebooks/swin_model.ipynb

(You can use this notebook.)

---

## 🚀 Training the I3D Classifier

python src/train_clip.py --model i3d


---

## 📦 Requirements

- torch
- torchvision
- timm
- pytorchvideo
- numpy
- opencv-python
- matplotlib
- pandas
- seaborn
- scipy
- sklearn
- tqdm
- pyyaml

