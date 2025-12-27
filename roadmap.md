# ClearQuote CV Engineer Assignment — End-to-End Roadmap (Validated & Edge-Ready)

## 1. Problem Overview

This assignment is an **image classification problem** optimized for **edge/mobile deployment**. The objective is to classify a single image of a vehicle into one of the following viewpoints:

- Front
- FrontLeft
- RearLeft
- Rear
- RearRight
- FrontRight
- Background / Unknown (7th class)

The final solution must balance **accuracy (per-class F1 score)** with **low latency, small model size, and deployability**, ensuring it runs reliably in ClearQuote’s mobile inference environment. fileciteturn2file0

---

## 2. High-Level Objectives

- Build a robust multi-class image classifier
- Optimize the model for mobile/edge inference
- Export the model to `.tflite`
- Provide a clean, reproducible inference script (`test_predict.py`)
- **Verify correctness, performance, and deployability before submission**
- Document the full workflow clearly for execution on a fresh system

---

## 3. Recommended End-to-End Pipeline

```
Raw Images + VIA JSON
        ↓
Label Extraction & Mapping
        ↓
Train / Validation / Test Split (80 / 10 / 10)
        ↓
Data Augmentation (label-aware)
        ↓
Model Training (MobileNetV3 / EfficientNet-Lite)
        ↓
Offline Evaluation (Precision / Recall / F1)
        ↓
SavedModel Export
        ↓
TFLite Conversion + Quantization
        ↓
TFLite ↔ SavedModel Verification
        ↓
Latency & Memory Validation (CPU / Edge)
        ↓
Batch Inference via test_predict.py
        ↓
Predictions CSV
```

---

## 4. Phase 1 — Dataset Preparation & Analysis

### 4.1 Dataset Overview (Validated)

**Dataset Statistics:**
- Total folders: 61
- Total images: 3,943
- Annotation format: VIA (VGG Image Annotator) JSON

**Viewpoint Distribution (from heuristic analysis):**
| Viewpoint | Count | Percentage |
|-----------|-------|------------|
| FrontLeft | 855 | 21.7% |
| FrontRight | 764 | 19.4% |
| RearRight | 613 | 15.5% |
| RearLeft | 510 | 12.9% |
| Background | 472 | 12.0% |
| Front | 346 | 8.8% |
| Rear | 336 | 8.5% |

**Class Imbalance**: FrontLeft/FrontRight have ~2.5x more samples than Front/Rear. Apply class weights.

### 4.2 Label Extraction Strategy (Deterministic Heuristics)

Since no explicit viewpoint labels exist, infer labels from annotated car parts:

**Part Identity → Viewpoint Vote Mapping:**
```python
FRONT_PARTS = {'frontheadlamp', 'frontbumper', 'frontbumpergrille', 'logo', 
               'bonnet', 'frontws', 'towbarcover', 'lowerbumpergrille', 'frontbumpercladding'}
REAR_PARTS = {'taillamp', 'tailgate', 'rearbumper', 'rearws', 'lefttaillamp', 
              'righttaillamp', 'antenna', 'rearbumpercladding'}
LEFT_PARTS = {'leftheadlamp', 'leftfoglamp', 'leftfrontdoor', 'leftreardoor',
              'leftwa', 'leftrunningboard', 'leftqpanel', 'leftorvm', 'leftfender', 
              'leftapillar', 'lefttaillamp'}
RIGHT_PARTS = {'rightheadlamp', 'rightfoglamp', 'rightfrontdoor', 'rightreardoor',
               'rightwa', 'rightrunningboard', 'rightqpanel', 'rightorvm', 'rightfender',
               'rightapillar', 'righttaillamp'}
DAMAGE_PARTS = {'scratch', 'dent', 'dirt', 'd2', 'bumpertorn', 'crack'}  # Ignore for viewpoint
```

**Label Resolution Algorithm:**
1. Count votes for each axis: `front_votes`, `rear_votes`, `left_votes`, `right_votes`
2. Primary axis: `Front` if front > rear, `Rear` if rear > front, tie-break to Front
3. Secondary axis: `Left` if left > right, `Right` if right > left, empty if equal
4. Combine: `primary + secondary` → `FrontLeft`, `RearRight`, etc.
5. Handle `partial_` prefix by stripping it before matching

**Note:** Important annotation patterns observed:
- `partial_frontbumper`, `partial_bonnet` → strip prefix, treat as full part
- `sensor`, `wiper`, `doorhandle`, `tyre` → neutral parts, don't indicate viewpoint

### 4.3 Background / Unknown Class (Automatic Detection)

**Sources identified from dataset analysis:**
- Empty annotations (`regions == []`): 57 images
- Damage-only images (only `scratch`, `dent`, `dirt`, `d2`): 21 images
- Total automatic Background candidates: ~78+ images

**Detection logic:**
```python
def is_background(regions):
    if len(regions) == 0:
        return True
    identities = {r['region_attributes'].get('identity', '').lower() for r in regions}
    non_damage = identities - DAMAGE_PARTS
    return len(non_damage) < 2  # Too few meaningful viewpoint indicators
```

Purpose: Reduce false positives when test images are random/irrelevant.

### 4.4 Dataset Splitting

- Stratified split to preserve class balance:
  - Training: 80%
  - Validation: 10%
  - Local Test: 10%
- Save split files explicitly:
  - `train.csv`, `val.csv`, `test.csv`

### 4.5 Class Imbalance Handling

If class imbalance is detected:
- Apply class weights during training, or
- Oversample minority classes, or
- Use Focal Loss

---

## 5. Phase 2 — Data Augmentation Strategy

To simulate real mobile capture conditions:

### Safe Augmentations
- Brightness and contrast jitter
- Hue and saturation variation
- Gaussian noise or slight blur
- Small rotations (±10–15°)
- Random scaling and cropping

### Conditional Augmentations (Label-Aware)

- **Horizontal Flip**
  - Must swap labels:
    - FrontLeft ↔ FrontRight
    - RearLeft ↔ RearRight

Augmentations must be implemented inside the training data pipeline to ensure reproducibility.

---

## 6. Phase 3 — Model Selection & Training

### 6.1 Architecture Recommendation

Optimized for mobile CPUs:

- **MobileNetV3-Small** (preferred for speed)
- **EfficientNet-Lite** (preferred for accuracy)

### 6.2 Training Configuration

- Input size: 224 × 224
- Framework: TensorFlow / Keras
- Loss Function: Categorical Crossentropy (or Focal Loss if needed)
- Optimizer: Adam
- Batch size: 16–64

### 6.3 Training Strategy

1. Freeze backbone
2. Train classification head (6–8 epochs)
3. Unfreeze backbone
4. Fine-tune entire model (3–6 epochs)

### 6.4 Monitoring & Evaluation

- Track **per-class Precision, Recall, and F1 score** on validation set
- Save the best-performing model
- Generate `results.txt` containing:
  - Precision, Recall, F1 per class
  - Confusion matrix summary
  - Observed failure modes

---

## 7. Phase 4 — Model Export & TFLite Conversion

### 7.1 SavedModel Export

- Export trained model as TensorFlow `SavedModel`
- Save `labels.txt` alongside the model

### 7.2 TFLite Conversion Options

- Float16 Quantization (recommended default)
- Dynamic Range Quantization
- Full INT8 Quantization (requires representative dataset)

### 7.3 Functional Verification

- Run identical samples through:
  - TensorFlow SavedModel
  - TFLite Interpreter
- Verify **top-1 prediction agreement ≥ 95%**
- Investigate discrepancies due to preprocessing or quantization

---

## 8. Phase 5 — Inference Script (`test_predict.py`)

### 8.1 Purpose

Provide a simple, robust script that allows ClearQuote to run inference on a folder of images using the TFLite model.

### 8.2 Functional Requirements

- Command-line interface
- Accepts:
  - Path to `.tflite` model
  - Path to `labels.txt`
  - Path to image folder
- Applies identical preprocessing as training
- Handles float32 and uint8 TFLite models
- Processes images efficiently without memory overflow

### 8.3 Output

Generate a Pandas DataFrame with:

```
image_name | prediction | score
```

Save as `predictions.csv`

---

## 9. Phase 6 — Validation & Edge Readiness (Critical for Success)

### 9.1 Offline Model Validation

Before submission, validate the model using:

- Per-class **Precision, Recall, and F1 score**
- Confusion matrix analysis (detect systematic swaps)
- Robustness testing on:
  - Background / Unknown images
  - Low-light and overexposed images
  - Partial occlusions and rotated inputs

### 9.2 TFLite Runtime Validation

- Measure average inference latency using:
  - Desktop CPU (baseline)
  - Android device or emulator (preferred)
- Record:
  - Device model / CPU
  - Average latency per image
  - Model size

### 9.3 Pass Criteria (Self-Check)

Recommended internal thresholds:

- Per-class F1 ≥ 0.80 (aim ≥ 0.85 where possible)
- SavedModel ↔ TFLite top-1 agreement ≥ 95%
- Stable inference without crashes on large image folders

Document these results in `results.txt`.

---

## 10. Deliverables

### Required Files

- `test_predict.py`
- `models/model.tflite`
- `models/saved_model/labels.txt`
- `readme.txt`
- `requirements.txt`

### Optional (Strongly Recommended)

- `results.txt` (metrics + failure analysis)
- Sample `predictions.csv`
- Smoke-test script for quick verification

---

## 11. Expected Outputs

| Component | Output |
|---------|--------|
| Training | SavedModel + labels.txt |
| Conversion | Quantized `.tflite` model |
| Evaluation | Precision / Recall / F1 per class |
| Inference | predictions.csv |

---

## 12. Packaging Instructions

- Ensure all scripts run on a clean system
- Avoid hardcoded paths
- Provide clear setup and execution instructions
- Final archive name:

```
product_overlay_firstname_lastname.zip
```

---

## 13. Success Criteria

- High per-class F1 score
- Verified TFLite correctness vs SavedModel
- Acceptable inference latency on CPU / mobile
- Clean, reproducible, grader-friendly codebase
- Clear documentation and execution steps

---

This updated document integrates **validation strategy, deployability checks, and edge-readiness verification**, ensuring the solution not only trains well locally but also passes ClearQuote’s evaluation environment reliably.

