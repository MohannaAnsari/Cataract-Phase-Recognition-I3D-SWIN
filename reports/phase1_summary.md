# Phase 1 – Foundation & Data-Driven Hypothesis  
**Aras Lab Onboarding Project: Cataract Surgery Phase Recognition**  
*Author:* Mohanna Ansari  *Date:* Oct24

---

## 1. Objective
The first phase of this project focuses on understanding the **Cataract-101 surgical-video dataset** and forming a **data-driven hypothesis** about which modeling strategy—clip-based classification or temporal segmentation—will better recognize surgical phases.  
To achieve this, the dataset’s structure was analyzed, the distribution and transitions of surgical phases were explored, and a reproducible data-processing pipeline was built to support both later experiments.

---

## 2. Dataset Overview
**Cataract-101** contains **101 anonymized cataract-surgery videos**, recorded at **25 fps** with a resolution of **720 × 540 px** and an average duration of roughly eight minutes.  
Each surgery is annotated by ophthalmic experts into **ten quasi-standardized phases** that span the entire procedure.

Only the **starting frame of each phase** is given in the annotations, so each full segment was reconstructed by pairing a start frame with the next start frame—or with the video’s end.  
Metadata also includes each surgeon’s ID and **experience level** (1 = low, 2 = high), allowing later exploration of inter-surgeon variability.

---

## 3. Exploratory Data Analysis (EDA)

### 3.1 Class Distribution
The dataset shows a clear **class imbalance** (Figure 2).  
Phases such as *Viscous Agent Injection* and *Irrigation and Aspiration* appear far more often than *Hydrodissection* or *Capsule Polishing*.  
Across all videos, the most common phase appears roughly **2–3 times more frequently** than the rarest one.  
This imbalance will require class weighting or balanced sampling during training.

---

### 3.2 Phase Durations
Reconstructed durations vary strongly across phases.  
Long, stable phases—like *Phacoemulsification* and *Irrigation and Aspiration*—typically last **40–60 seconds**, while short transitional steps—such as *Incision* or *Viscous Agent Injection*—often last **under 15 seconds**.  
This suggests a mix of **steady visual contexts** and **rapid transitions**, each demanding different modeling behavior.

---

### 3.3 Temporal Transitions
The **phase-transition matrix** (Figure 1) shows that most surgeries follow a largely sequential order, with strong diagonal dominance—once a phase begins, it usually persists.  
However, several repeated transitions exist, particularly for *Viscous Agent Injection*, which is often performed twice.  
This indicates a **non-linear workflow** and confirms that **temporal continuity** and **context** are essential for accurate modeling.

---

#### *Figure 1 – Phase Transition Matrix*  
![Phase Transition Matrix](./outputs/plots/transition_matrix.png)

#### *Figure 2 – Number of Segments per Phase*  
![Number of Segments per Phase](./outputs/plots/class_balance.png)
---

## 4. Data Processing Pipeline
To support both modeling paradigms, a unified preprocessing pipeline was developed with these key components:

1. **Metadata Integration** – merges `videos.csv`, `phases.csv`, and `annotations.csv`.  
2. **Segment Reconstruction** – converts phase start frames into continuous intervals.  
3. **Clip Sampler** – extracts fixed-length clips (e.g., 16 frames @ 3 fps) from each segment.  
4. **Flexible Dataset Class** – PyTorch `Dataset` returning either  
   - *(Clip mode)* → `(video clip, label)` for classification, or  
   - *(Sequence mode)* → `(frame sequence, label sequence)` for segmentation.  
5. **Configuration File** – `config.yaml` defines FPS, clip length, and resize dimensions for reproducibility.

This design ensures that later models share the same data pipeline, allowing a fair comparison between paradigms.

---

## 5. Key Findings

| Observation | Evidence | Implication |
|--------------|-----------|-------------|
| **Strong class imbalance** | Class distribution plot | Requires weighting or augmentation |
| **Long stable phases** | Duration statistics | Classification models may perform well here |
| **Short transitional phases** | < 15 s mean duration | Segmentation models needed for fine temporal resolution |
| **Repeated phases** | Transition matrix | Workflow is non-linear; context awareness is critical |
| **Temporal continuity** | Diagonal dominance | Time-aware models can exploit smoothness |

---

## 6. Hypothesis
> Because cataract surgeries include mostly long, stable phases but also brief transitional ones that sometimes repeat, a **temporal segmentation model** (such as MS-TCN or ASFormer) is expected to outperform a simple clip-based classifier near phase boundaries and repeated injections by enforcing temporal consistency.  
> The clip-based classifier, however, may still achieve strong performance on the visually stable phases that dominate each video.

---

## 7. Deliverables Produced

| Deliverable | Description | File |
|--------------|--------------|------|
| Class distribution plot | Frequency of each surgical phase | `class_balance.png` |
| Phase transition matrix | Temporal relationships between phases | `transition_matrix.png` |
| Duration statistics | Mean and variance of phase durations | `duration_stats.csv` |
| Data pipeline implementation | Unified PyTorch dataset for clip & segmentation modes | `src/dataset.py` |
| Phase 1 report | This summary document | `phase1_summary.md` |

---

## 8. Conclusion
This phase established a detailed understanding of the dataset and produced a robust, reusable pipeline for later experiments.  
The analysis revealed that cataract surgeries exhibit both predictable temporal patterns and irregular transitions, motivating the use of **temporal models** that explicitly account for time continuity.  
These findings directly guide the modeling choices and evaluation strategies in Phases 2 and 3.

---
