Vehicle Viewpoint Classifier

MobileNetV2-based classifier for 7 vehicle viewpoints (Front, FrontLeft, FrontRight, Rear, RearLeft, RearRight, Background), optimized for TFLite edge deployment.

1. Setup:

```
conda create -n vehicle_classifier python=3.9
conda activate vehicle_classifier
pip install -r requirements.txt
```

2. Usage:

a. Generate training data from VIA annotations:
```
python data_preparation.py
```

b. Train the model (two-phase: frozen backbone, then fine-tune):
```
python train.py
```

c. Convert to TFLite:
```
python convert_tflite.py
```

d. Run inference:
```
python test_predict.py --model models/model.tflite \
                       --labels models/saved_model/labels.txt \
                       --images dataset/5e9112c35026365e15eb871b
```

Optional flags:
  --visualize           Save prediction sample grids to prediction_samples/
  --ground-truth CSV    Use labels from CSV to show correct/incorrect (instead of confidence)

3. Project Structure:

```
├── data_preparation.py     # VIA JSON parsing, label extraction, splits
├── train.py                # MobileNetV2 training
├── convert_tflite.py       # SavedModel -> TFLite
├── test_predict.py         # Inference script
├── models/
│   ├── saved_model/        # TF SavedModel + labels.txt
│   └── model.tflite        # Float16 quantized (~4.5 MB)
├── train.csv, val.csv, test.csv
└── dataset/                # Raw VIA annotations
```

4. Model:

- Base: MobileNetV2 (ImageNet pretrained)
- Input: 224x224x3, normalized to [-1, 1]
- Head: GAP -> Dropout(0.2) -> Dense(128) -> Dropout(0.1) -> Softmax(7)

5: Training

Two-phase transfer learning:
1. Phase 1: Backbone frozen, train head only (20 epochs, LR=1e-3)
2. Phase 2: Full fine-tune with BatchNorm frozen (15 epochs, LR=1e-4)

Uses balanced class weights, EarlyStopping (patience=7), ReduceLROnPlateau.

Data Augmentation (training only):
- RandomBrightness(0.2): ±20% brightness variation
- RandomContrast(0.2): ±20% contrast variation  
- RandomZoom(0.05): ±5% zoom (conservative)

Note: Geometric transforms (flip, rotation) are avoided as they would change viewpoint labels (FrontLeft→FrontRight) without proper label swapping.

6: Results

| Metric | Value |
|--------|-------|
| Test Accuracy | 86.18% |
| Macro F1 | 0.832 |
| Weighted F1 | 0.861 |
| TFLite Agreement | 99.0% |
| Model Size | 4.57 MB |

Per-class F1: Front 0.74, FrontLeft 0.88, FrontRight 0.88, Rear 0.82, RearLeft 0.89, RearRight 0.91, Background 0.70

7: Label Extraction

Viewpoint labels are inferred from VIA polygon annotations via voting:
- Count parts in FRONT_PARTS, REAR_PARTS, LEFT_PARTS, RIGHT_PARTS sets
- Primary axis: Front vs Rear (whichever has more votes)
- Secondary axis: Left vs Right
- Background: empty annotations or <2 meaningful parts

Note: 'logo' excluded from voting since it appears on both front and rear bumpers.

8. Dataset

3,974 images across 61 folders (80/10/10 split).

Distribution: FrontLeft 27%, FrontRight 24%, RearRight 15%, RearLeft 13%, Front 8%, Rear 8%, Background 5%

9. Output

`predictions.csv`: image_name, prediction, score (confidence 0-1)
