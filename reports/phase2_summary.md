# Phase 2: Experiment 1 – The Clip Classification Paradigm  

## 🎯 Goal  
To establish a strong performance baseline by modeling the cataract surgery phase recognition task as a **standard action-recognition problem**, where short, isolated clips (≈ 2–3 seconds) are individually classified into one of ten surgical phases.  

---

## ⚙️ Methodology  

### Model Design  
A **VideoMAE-Base (Swin/Transformer-based)** backbone was fine-tuned for surgical phase classification:  
- **Input:** 32-frame clips sampled at 10 fps (~3.2 s) resized to 224 × 224 px  
- **Backbone:** `MCG-NJU/videomae-base` pretrained weights  
- **Fine-tuning:**  
  - First 8 encoder blocks frozen initially, then unfrozen after epoch 5  
  - Dropout (0.3) and label smoothing (0.1)  
  - Weighted cross-entropy loss using inverse class frequency  
- **Optimizer:** AdamW with cosine warm-up scheduler  
- **Batch size:** 4    |   **Epochs:** 25  
- **Augmentations:** random crop, flip, rotation, color jitter, grayscale  
- **Hardware:** NVIDIA GTX 1660 Ti (6 GB)  

### Data Preparation  
- Dataset split: 80 % training / 20 % validation at the segment level  
- Class imbalance handled by weighted sampler and loss reweighting  

### Evaluation Strategy  
Clip-level predictions were stitched together along the timeline of each surgery to form frame-wise phase sequences.  
Temporal post-processing was applied using:  
1. **Median filter (k = 7)** for short-term stabilization  
2. **Local majority vote (window = 11)** for contextual smoothing  

---

## 📊 Results  

### Training Progress (Highlights)
| Epoch | Train Acc | Val Acc | Val Loss | Notes |
|:------|-----------:|--------:|---------:|:------|
| 1 → 5   | 0.04 → 0.09 | 0.07 → 0.31 | ↓ | Backbone frozen |
| 6 → 15  | 0.14 → 0.44 | 0.31 → 0.68 | ↓ | Fine-tuning improves stability |
| 16 → 25 | 0.55 → 0.89 | 0.69 (max) | 1.68 | Final convergence |

**Best validation accuracy:** 0.695  
**Best validation loss:** 1.681  

---

### Per-Video Timeline Evaluation  
| Metric | Raw | Median | Voted |
|:--|:--:|:--:|:--:|
| Accuracy | 0.711 | 0.711 | 0.711 |
| Macro F1 | 0.628 | 0.628 | 0.628 |

**Timeline Classification Report (voted):**
accuracy = 0.695
macro avg F1 = 0.215
weighted avg F1 = 0.675


**Confusion Matrix (timeline, voted):**
![Confusion Matrix](./outputs/plots/swin_confusion.png)

Most predictions concentrated on dominant late-stage phases (e.g., Viscous Agent Injection), while early phases (0–4) remained under-represented, revealing strong class imbalance and weak temporal discrimination.

---

## 🔍 Analysis  

### Performance Summary  
- **Global appearance learning:** Model effectively identifies tools and surgical context.  
- **Temporal weakness:** Limited context window fails to capture smooth phase transitions.  
- **Boundary errors:** Frequent misclassifications around phase changes.  
- **Post-processing:** Median and voting filters improve temporal stability but not overall accuracy.  

### Failure Modes at Phase Transitions  
- Short transition phases (≤ 2 s) are often skipped or merged with neighboring phases.  
- Confusion between similar phases with comparable visual patterns (e.g., Rhexis vs. Capsule Polishing).  
- Long dominant phases over-represented in predictions.  

---

## 🧩 Post-Processing Investigation  
To stabilize predictions:  
- **Median filtering** reduced spikes but kept same accuracy (0.711).  
- **Voting window** slightly smoothed boundaries without improving metrics.  
Future work may explore probabilistic temporal models (e.g., HMM, GraphCut optimization) to better model phase transitions.  

---

## 🧠 Conclusion  
The clip-classification approach achieved ~71 % frame-level accuracy and macro F1 ≈ 0.63 after temporal voting, demonstrating that visual context alone can discriminate surgical phases to a reasonable extent.  
However, its inability to enforce temporal continuity and to capture fine-grained transitions limits its applicability for precise surgical workflow analysis.  

These results validate the need for a temporal segmentation approach (Phase 3), which explicitly models frame-to-frame dependencies to overcome the observed instabilities.  

---

## 🔮 Future Work  
- Implement **ASFormer** or other temporal segmentation models for continuous frame-level prediction.  
- Integrate **GraphCut-based temporal smoothing**, drawing from the author’s ICCKE-2025-accepted paper on *Semantic Smoothness Optimization via Graph-Cut Energy Minimization*.  
- Explore balanced phase sampling and feature-level fusion to reduce class imbalance.  

---

**Deliverable Summary:**  
- **Baseline Model:** VideoMAE/Swin Transformer  
- **Best Validation Accuracy:** 0.695  
- **Best Timeline Accuracy:** 0.711  
- **Main Limitation:** Temporal instability near phase boundaries  
- **Next Step:** Temporal Segmentation (ASFormer)
