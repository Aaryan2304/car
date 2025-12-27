# ClearQuote Vehicle Viewpoint Classifier - Copilot Instructions

## Project Overview

This is an **edge-deployable image classification** project for classifying vehicle images into 7 viewpoints:
- `Front`, `FrontLeft`, `FrontRight`, `Rear`, `RearLeft`, `RearRight`, `Background/Unknown`

Target: Mobile/edge inference using TensorFlow Lite with high F1 scores per class.

## Dataset Structure

```
dataset/
  {folder_id}/
    via_region_data.json    # VIA annotations (polygon regions for car parts)
    *.jpg / *.jpeg          # Vehicle images
```

### VIA JSON Annotation Format
- Use `filename` field as the actual image filename (not the JSON key which includes size)
- `regions[]` contains polygon annotations with `identity` labels (e.g., `tyre`, `frontbumper`, `leftheadlamp`)
- `shape_attributes.all_points_x/y` contain absolute pixel coordinates
- **For this task**: Classify entire image into viewpoint category based on visible parts, not individual region annotations

### Label Extraction Strategy (Deterministic Heuristics)

**Step 1: Part Identity → Vote Mapping** (based on actual dataset analysis)
```python
FRONT_PARTS = {'frontheadlamp', 'frontbumper', 'frontbumpergrille', 'logo', 
               'bonnet', 'frontws', 'towbarcover', 'lowerbumpergrille', 'frontbumpercladding'}
REAR_PARTS = {'taillamp', 'tailgate', 'rearbumper', 'rearws', 'lefttaillamp', 
              'righttaillamp', 'antenna', 'rearbumpercladding'}
LEFT_PARTS = {'leftheadlamp', 'leftfoglamp', 'leftfrontdoor', 'leftreardoor',
              'leftwa', 'leftrunningboard', 'leftqpanel', 'leftfrontdoorcladding',
              'leftreardoorcladding', 'leftorvm', 'leftfender', 'leftapillar', 'lefttaillamp'}
RIGHT_PARTS = {'rightheadlamp', 'rightfoglamp', 'rightfrontdoor', 'rightreardoor',
               'rightwa', 'rightrunningboard', 'rightqpanel', 'rightorvm', 'rightfender',
               'rightapillar', 'righttaillamp'}
DAMAGE_PARTS = {'scratch', 'dent', 'dirt', 'd2', 'bumpertorn', 'bumpertear', 
                'bumperdent', 'crack', 'clipsbroken', 'rust'}  # Ignore for viewpoint
```

**Step 2: Vote Counting**
For each image, count votes: `front_votes`, `rear_votes`, `left_votes`, `right_votes`

**Step 3: Label Resolution**
```python
# Primary axis (front vs rear)
if front_votes > rear_votes:
    primary = 'Front'
elif rear_votes > front_votes:
    primary = 'Rear'
else:
    primary = 'Front' if front_votes > 0 else None  # tie-break: prefer Front

# Secondary axis (left vs right)
if left_votes > right_votes:
    secondary = 'Left'
elif right_votes > left_votes:
    secondary = 'Right'
else:
    secondary = None  # pure front/rear view

# Combine: 'Front', 'FrontLeft', 'RearRight', etc.
label = primary + (secondary or '') if primary else 'Background'
```

**Step 4: Centroid-Based Tie-Breaking (Optional Enhancement)**
When `left_votes == right_votes` and both > 0:
- Compute average X-centroid of left-side parts vs right-side parts
- If left centroids are more prominent (closer to image center), choose Left

### Background/Unknown Class Detection
**Automatic detection** (no manual folder needed):
```python
def is_background(regions):
    if len(regions) == 0:
        return True
    # Filter out tiny decorative regions (< 1% of image area)
    significant_regions = [r for r in regions if region_area(r) > MIN_AREA_THRESHOLD]
    return len(significant_regions) < 2  # Too few meaningful parts
```
- Images with `regions == []` → Background
- Images with only damage annotations (`scratch`, `dent`, `dirt`) → Background
- Ambiguous combinations with < 2 significant car parts → Background candidate

## Key Implementation Patterns

### Data Pipeline
1. Parse all `via_region_data.json` files to extract image filenames
2. Apply voting heuristics to infer viewpoint label from `identity` fields
3. Detect Background class via empty/sparse annotations
4. Create stratified splits: 80% train / 10% val / 10% test
5. Save as CSV: `filename,label`

### Data Augmentation (Label-Aware)
```python
# Horizontal flip MUST swap labels:
# FrontLeft <-> FrontRight
# RearLeft <-> RearRight
# Front and Rear remain unchanged
```

### Model Architecture
- Preferred: **MobileNetV3-Small** (speed) or **EfficientNet-Lite** (accuracy)
- Input size: 224×224
- Framework: TensorFlow/Keras
- Training: Freeze backbone → train head (6-8 epochs) → fine-tune (3-6 epochs)

### Class Imbalance
- Apply class weights or use Focal Loss if imbalance detected
- Background class is auto-generated from sparse annotations (see above)

## Dataset Statistics (from analysis)

```
Total folders: 61
Total images: 3,943

Viewpoint Distribution (from heuristics):
  FrontLeft:  855 images
  FrontRight: 764 images  
  RearRight:  613 images
  RearLeft:   510 images
  Background: 472 images (includes 57 empty + 21 damage-only)
  Front:      346 images
  Rear:       336 images

Top Part Identities: sensor(3069), scratch(2669), tyre(2566), 
  doorhandle(2157), logo(1958), frontbumper(1958)
```

**Note**: Class imbalance exists - FrontLeft has 2.5x more samples than Front/Rear.

## Development Setup

**Using existing `gpu` conda environment (TensorFlow CPU):**

```bash
# Activate environment
conda activate gpu

# Verify TensorFlow installation
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"

# Project structure to create:
├── data_preparation.py      # JSON parsing, label extraction, splits
├── train.py                 # Model training with augmentation
├── convert_tflite.py        # SavedModel → TFLite conversion
├── test_predict.py          # Inference script (required deliverable)
├── models/
│   ├── saved_model/
│   │   └── labels.txt
│   └── model.tflite
├── train.csv, val.csv, test.csv
└── requirements.txt
```

## Required Deliverables

| File | Purpose |
|------|---------|
| `test_predict.py` | Inference script (CLI: model path, labels, image folder) |
| `models/model.tflite` | Quantized TFLite model |
| `models/saved_model/labels.txt` | Label mapping |
| `readme.txt` | Setup, architecture, training params |
| `requirements.txt` | Python dependencies |
| `predictions.csv` | Output format: `image_name,prediction,score` |

## Inference Script Requirements (`test_predict.py`)

```python
# Must handle:
# - Command-line args: --model, --labels, --images
# - Both float32 and uint8 TFLite models
# - Same preprocessing as training
# - Batch processing without memory overflow
# - Output: pandas DataFrame saved as predictions.csv
```

## TFLite Conversion

1. Export TensorFlow SavedModel with `labels.txt`
2. Convert with Float16 quantization (default) or INT8 (requires representative dataset)
3. **Verify**: Top-1 prediction agreement ≥ 95% between SavedModel and TFLite

## Validation Criteria

- Per-class F1 ≥ 0.80 (target ≥ 0.85)
- SavedModel ↔ TFLite agreement ≥ 95%
- Test on edge cases: low-light, partial occlusion, rotated inputs
- Document results in `results.txt`

## Packaging

- Archive name: `product_overlay_firstname_lastname.zip`
- Avoid hardcoded paths
- Must run on clean system with clear setup instructions
