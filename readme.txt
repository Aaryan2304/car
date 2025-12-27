================================================================================
ClearQuote Vehicle Viewpoint Classifier
================================================================================

OVERVIEW
--------
This project classifies vehicle images into 7 viewpoint categories:
- Front, FrontLeft, FrontRight, Rear, RearLeft, RearRight, Background

Designed for edge/mobile deployment using TensorFlow Lite.


SETUP INSTRUCTIONS
------------------

1. Create Python environment (Python 3.9+ recommended):
   
   conda create -n vehicle_classifier python=3.9
   conda activate vehicle_classifier

2. Install dependencies:
   
   pip install -r requirements.txt

3. Verify TensorFlow installation:
   
   python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"


PROJECT STRUCTURE
-----------------

├── data_preparation.py     # Parse dataset, extract labels, create splits
├── train.py                # Model training (MobileNetV2)
├── convert_tflite.py       # SavedModel → TFLite conversion
├── test_predict.py         # Inference script (required deliverable)
├── requirements.txt        # Python dependencies
├── readme.txt              # This file
├── models/
│   ├── saved_model/        # TensorFlow SavedModel
│   │   └── labels.txt      # Class labels
│   └── model.tflite        # Quantized TFLite model
├── train.csv               # Training split
├── val.csv                 # Validation split
├── test.csv                # Test split
└── dataset/                # Input dataset with VIA annotations


USAGE
-----

Step 1: Prepare Data
   python data_preparation.py

   Creates: train.csv, val.csv, test.csv, models/saved_model/labels.txt

Step 2: Train Model
   python train.py

   Creates: models/saved_model/, models/best_model.keras

Step 3: Convert to TFLite
   python convert_tflite.py

   Creates: models/model.tflite

Step 4: Run Inference
   python test_predict.py --model models/model.tflite \
                          --labels models/saved_model/labels.txt \
                          --images <image_folder>

   Creates: predictions.csv


MODEL ARCHITECTURE
------------------

Base Model:      MobileNetV2 (ImageNet pretrained)
Input Size:      224 x 224 x 3 (RGB)
Output:          7-class softmax
Model Size:      ~4.5 MB (TFLite Float16)

Classification Head:
- GlobalAveragePooling (from backbone)
- Dropout (0.2)
- Dense (128, ReLU)
- Dropout (0.1)
- Dense (7, Softmax)


TRAINING PARAMETERS
-------------------

Preprocessing:
- Resize to 224x224
- Normalize to [-1, 1] (MobileNetV2 standard)

Training Strategy (Two-Phase):
- Phase 1: Freeze backbone, train head (20 epochs, LR=1e-3)
- Phase 2: Fine-tune all layers with BN frozen (15 epochs, LR=1e-4)

Optimization:
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Class weights: Balanced (computed from training distribution)

Callbacks:
- EarlyStopping (patience=7, monitor val_accuracy, restore_best_weights)
- ReduceLROnPlateau (factor=0.5, patience=3)
- ModelCheckpoint (save best validation accuracy)

ACHIEVED RESULTS
----------------

Test Set Performance:
- Overall Accuracy: 84.2%
- Macro F1 Score: 0.814
- Weighted F1 Score: 0.843

Per-Class F1 Scores:
- Front:      0.732
- FrontLeft:  0.848
- FrontRight: 0.874
- Rear:       0.831
- RearLeft:   0.852
- RearRight:  0.908
- Background: 0.650

TFLite Conversion:
- SavedModel ↔ TFLite Agreement: 100%
- Model Size: 4.57 MB


LABEL EXTRACTION HEURISTICS
---------------------------

Labels are inferred from VIA JSON polygon annotations using voting:

1. Extract all part identities from image annotations
2. Count votes for Front/Rear/Left/Right based on part names
3. Resolve primary axis (Front vs Rear) and secondary axis (Left vs Right)
4. Combine: e.g., "Front" + "Left" = "FrontLeft"

Part → Vote Mapping:
- FRONT: frontbumper, bonnet, frontws, frontbumpergrille, frontheadlamp, etc.
  (Note: 'logo' excluded since logos appear on both front AND rear of vehicles)
- REAR: rearbumper, tailgate, rearws, taillamp, etc.
- LEFT: leftheadlamp, leftfrontdoor, leftorvm, leftfender, etc.
- RIGHT: rightheadlamp, rightfrontdoor, rightorvm, rightfender, etc.

Background Detection:
- Empty annotations (no regions)
- Only damage annotations (scratch, dent, dirt)
- Fewer than 2 significant car parts


TFLITE CONVERSION
-----------------

Quantization: Float16 (default)
Expected Agreement: ≥95% with SavedModel predictions

Alternative: INT8 quantization available for smaller model size
(requires representative dataset for calibration)


VALIDATION CRITERIA
-------------------

- Per-class F1 ≥ 0.80 (target ≥ 0.85)
- SavedModel ↔ TFLite agreement ≥ 95%
- Test on edge cases: low-light, partial occlusion


OUTPUT FORMAT
-------------

predictions.csv columns:
- image_name: Original filename
- prediction: Predicted viewpoint class
- score: Confidence score (0.0 - 1.0)


DATASET STATISTICS
------------------

Total Images: 3,943
Total Folders: 61

Distribution:
  FrontLeft:   855 (21.7%)
  FrontRight:  764 (19.4%)
  RearRight:   613 (15.5%)
  RearLeft:    510 (12.9%)
  Background:  472 (12.0%)
  Front:       346 (8.8%)
  Rear:        336 (8.5%)


AUTHOR
------
ClearQuote Assignment Submission


================================================================================
