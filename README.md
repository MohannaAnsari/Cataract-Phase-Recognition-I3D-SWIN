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
File: `src/swin_model.ipynb`  
(**You should paste your Phase 2 notebook implementation here**)  
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

src/
dataset.py
eda.py
phase1_main.py
train_clip.py
evaluate_clip.py
i3d_model.py
swin_model.py # <-- added
utils.py
notebooks/
phase_2.ipynb # Swin/VideoMAE experiments
reports/
phase1_summary.md
phase2_summary.md



---

## 🚀 Training the Swin Classifier

python src/train_clip.py --model swin

(You can add a CLI argument or just switch in code.)

---

## 🚀 Training the I3D Classifier

python src/train_clip.py --model i3d


---

## 📦 Requirements

torch
torchvision
timm
pytorchvideo
numpy
opencv-python
matplotlib
pandas
seaborn
scipy
sklearn
tqdm
pyyaml

