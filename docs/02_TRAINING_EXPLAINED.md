# Training Script - Deep Dive

## File: `train.py`

This document explains every aspect of the training script for your interview.

---

## 1. High-Level Strategy

### Transfer Learning with MobileNetV2

**What is Transfer Learning?**
Instead of training a CNN from scratch (which requires millions of images), I start with a model pre-trained on ImageNet (1.2 million images, 1000 classes). The early layers already know how to detect edges, textures, shapes. I just need to teach the model to recognize vehicle viewpoints.

**Why MobileNetV2?**

| Consideration | MobileNetV2 | MobileNetV3-Small | EfficientNetV2-B0 | ResNet50 |
|--------------|-------------|-------------------|-------------------|----------|
| Parameters | 3.5M | 2.5M | 4.7M | 25M |
| TFLite Size | ~4.5 MB | ~2.5 MB | ~15 MB | ~100 MB |
| Inference Speed | ~15ms | ~8ms | ~25ms | ~80ms |
| Training Stability | ⭐⭐⭐ Very stable | ⭐ Unstable | ⭐⭐ Moderate | ⭐⭐⭐ Stable |
| Our Test Accuracy | **85.2%** | 11% (failed) | Not tested | Too large |

**Final Choice: MobileNetV2** - Best balance of accuracy, speed, and training stability.

**The MobileNetV3 Failure - What Happened:**

```python
# Initial attempt:
base_model = keras.applications.MobileNetV3Small(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet',
    pooling='avg'
)
# Result: Only 11% accuracy (worse than random!)
```

**Why MobileNetV3 Failed for This Task:**
1. **Hard-Swish Activation**: Sharper gradients caused instability with small dataset
2. **Squeeze-and-Excitation Blocks**: Attention mechanisms need more data to learn
3. **Smaller Bottleneck**: 576-dim output vs MobileNetV2's 1280-dim (less feature richness)
4. **Different Preprocessing**: Subtle normalization differences caused issues

**After Switching to MobileNetV2:**
- Phase 1 accuracy jumped from 11% → 61%
- Training was stable and predictable
- Final test accuracy: 85.2% (after logo removal improvement)

**Interview Q: Why not use the latest MobileNetV3?**
> I tried MobileNetV3-Small initially and got only 11% accuracy. It has architectural differences (hard-swish activation, squeeze-excite blocks) that made fine-tuning unstable with my small dataset. MobileNetV2 with its simpler architecture trained more reliably.

### Two-Phase Training - The Core Strategy

**The Problem with Single-Phase Training:**
```
Pretrained Backbone (excellent features) + Random Head (garbage weights)
                              ↓
Train everything together with high learning rate
                              ↓
Random head gradients corrupt backbone features
                              ↓
Poor final accuracy
```

**Two-Phase Solution:**
```
Phase 1: Freeze backbone, train head
- Head learns to use existing features
- Backbone stays protected
- Result: ~61% accuracy

Phase 2: Unfreeze backbone (keep BatchNorm frozen)
- Head now produces meaningful gradients
- Backbone adapts to vehicle domain
- Result: ~85% accuracy
```

**Phase 1: Feature Extraction (Frozen Backbone)**
- Freeze all MobileNetV2 layers
- Train only the classification head
- High learning rate (1e-3)
- 20 epochs

**Phase 2: Fine-Tuning (Unfrozen Backbone)**
- Unfreeze backbone layers
- **Keep BatchNorm frozen** (critical!)
- Low learning rate (1e-4)
- 15 epochs

**Training Progress (After Logo Removal):**
| Phase | Epoch | Train Acc | Val Acc | Notes |
|-------|-------|-----------|---------|-------|
| 1 | 1 | ~36% | ~47% | Random head starting to learn |
| 1 | 10 | ~73% | ~57% | Head learning |
| 1 | 18 | ~84% | ~61% | Head converged (best Phase 1) |
| 2 | 1 | ~58% | ~62% | Backbone unfrozen, adapting |
| 2 | 9 | ~93% | ~82% | Rapid improvement |
| 2 | 15 | ~99.9% | ~86% | Phase 2 complete |

**Final Test Set: 85.2% accuracy**

**Interview Q: Why two phases instead of end-to-end training?**
> 1. **Prevent destroying pretrained features**: If we train everything at once with a high learning rate, the random classification head gradients can corrupt the carefully learned ImageNet features.
> 2. **Faster convergence**: The head learns meaningful combinations first, then fine-tuning adjusts the backbone to specialize in vehicle features.
> 3. **More stable**: Less likely to diverge or oscillate.

---

## 2. Configuration Parameters

```python
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 7
```

**Interview Q: Why 224x224?**
> MobileNetV2 was trained on 224x224 ImageNet images. Using a different size would require the model to generalize to unseen resolutions, potentially hurting accuracy. The pretrained weights expect this size.

**Interview Q: Why batch size 32?**
> Trade-off between:
> - **Larger batches (64+)**: More stable gradients, but may require more memory
> - **Smaller batches (16)**: More noise in gradients, can help generalization
> 32 is a sweet spot that fits in memory and provides stable training.

```python
PHASE1_EPOCHS = 20
PHASE2_EPOCHS = 15
INITIAL_LR = 1e-3
FINE_TUNE_LR = 1e-4
```

**Interview Q: How did you choose these hyperparameters?**
> - **Epochs**: I use EarlyStopping with patience=7, so these are maximums. Training typically stops earlier.
> - **Learning rates**: 1e-3 is standard for Adam optimizer. 1e-4 (10x lower) for fine-tuning prevents destroying pretrained features.

---

## 3. Data Pipeline (tf.data)

### The `load_image()` Function

```python
def load_image(filepath, label):
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)
    return img, label
```

**Interview Q: Why use tf.io instead of PIL or OpenCV?**
> tf.data pipeline runs on GPU/CPU in parallel with training. Using TensorFlow ops keeps everything in the same graph, enabling:
> - **Parallel data loading**: `num_parallel_calls=tf.data.AUTOTUNE`
> - **Prefetching**: `dataset.prefetch(tf.data.AUTOTUNE)`
> - **No Python GIL bottleneck**

### The Preprocessing Function

```python
def preprocess_for_efficientnet(img, label):
    img = (img / 127.5) - 1.0
    return img, label
```

**Interview Q: Why normalize to [-1, 1] instead of [0, 1]?**
> MobileNetV2 was pretrained with this normalization. Using [0, 1] would shift all input values, making the pretrained weights ineffective. **The inference preprocessing must match training exactly.**

**Calculation**:
- Input: [0, 255]
- After `/ 127.5`: [0, 2]
- After `- 1.0`: [-1, 1]

### Dataset Creation

```python
def create_dataset(csv_path, classes, is_training=False):
    df = pd.read_csv(csv_path)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    filepaths = df['filepath'].tolist()
    labels = [class_to_idx[lbl] for lbl in df['label']]
    
    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
```

**Interview Q: Why create class_to_idx from the loaded labels.txt?**
> To ensure the class indices match the saved labels.txt file. If I hardcoded indices, there could be mismatches between training and inference.

```python
    if is_training:
        dataset = dataset.shuffle(buffer_size=len(filepaths), reshuffle_each_iteration=True)
```

**Interview Q: Why buffer_size=len(filepaths)?**
> For perfect shuffling, the buffer should hold all samples. With a smaller buffer, early samples might appear more often than late samples. Since our dataset fits in memory, we use the full size.

**`reshuffle_each_iteration=True`**: Different random order every epoch.

```python
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(preprocess_for_efficientnet, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
```

**Interview Q: What does AUTOTUNE do?**
> TensorFlow automatically determines the optimal number of parallel calls and prefetch buffer size based on your hardware. This maximizes CPU/GPU utilization.

---

## 4. Class Weights for Imbalanced Data

```python
def compute_class_weights(csv_path, classes):
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(len(classes)),
        y=labels
    )
    return dict(enumerate(weights))
```

**Interview Q: Why are class weights necessary?**
> The dataset is imbalanced:
> - FrontLeft: 855 samples (21.7%)
> - Rear: 336 samples (8.5%)
> 
> Without weights, the model would optimize for FrontLeft (predicting it more often) since that minimizes overall loss. Class weights penalize mistakes on minority classes more heavily.

**How 'balanced' works**:
```
weight[class] = total_samples / (num_classes × class_samples)
```
Rare classes get higher weights; common classes get lower weights.

**Interview Q: What are the actual weights?**
> Something like:
> - FrontLeft: ~0.66 (lower weight, many samples)
> - Rear: ~1.68 (higher weight, fewer samples)
> - Background: ~1.20 (moderate weight)

---

## 5. Model Architecture

```python
def build_model(num_classes):
    base_model = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    base_model.trainable = False
```

**`include_top=False`**: Remove the ImageNet classification head (1000 classes). We add our own 7-class head.

**`weights='imagenet'`**: Load pretrained weights.

**`pooling='avg'`**: Add GlobalAveragePooling2D after the last conv block. Converts spatial features (7×7×1280) to a 1280-dimensional vector.

```python
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
```

**`training=False`**: Even during training, run BatchNorm in inference mode. This is important when the backbone is frozen - we don't want to update running statistics.

```python
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
```

**Interview Q: Why such a small classification head (128 units)?**
> With only ~3000 training samples, a large head (256-512 units) would overfit. The 128-unit layer provides enough capacity to learn viewpoint combinations without memorizing the training set.

**Interview Q: Why two dropout layers?**
> - First dropout (0.2): After backbone features, before dense layer
> - Second dropout (0.1): Before final classification
> This provides regularization at two points in the network.

**Interview Q: Why not add more dense layers?**
> The backbone already produces rich 1280-dimensional features. Adding more layers would increase parameters without adding much value. Simple heads often work better for transfer learning.

---

## 6. Callbacks

### EarlyStopping

```python
EarlyStopping(
    monitor='val_accuracy',
    patience=7,
    restore_best_weights=True,
    mode='max'
)
```

**Interview Q: Why monitor val_accuracy instead of val_loss?**
> Accuracy is the metric we care about. Sometimes loss can decrease while accuracy stagnates (overfitting to confident wrong predictions). Monitoring accuracy directly aligns with our goal.

**`patience=7`**: Wait 7 epochs with no improvement before stopping.

**`restore_best_weights=True`**: After stopping, roll back to the epoch with best val_accuracy.

### ReduceLROnPlateau

```python
ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7
)
```

**Purpose**: If validation loss plateaus, reduce learning rate to fine-tune more carefully.

**Interview Q: Why monitor val_loss here but val_accuracy for EarlyStopping?**
> Loss is more sensitive to small changes. It might plateau before accuracy drops significantly. Reducing LR when loss plateaus helps escape local minima.

### ModelCheckpoint

```python
ModelCheckpoint(
    filepath='models/best_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)
```

**Purpose**: Save the model whenever val_accuracy improves.

**Interview Q: Why save during training instead of just at the end?**
> Training might overfit later. The best model might be from epoch 12, not epoch 20. Checkpointing ensures we keep the best version.

---

## 7. Phase 2: Fine-Tuning Details

```python
backbone.trainable = True

# Freeze batch normalization layers
for layer in backbone.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False
```

### The Critical BatchNorm Trick

**Interview Q: Why freeze BatchNorm during fine-tuning?**

This is crucial and often asked. BatchNorm layers have two components:
1. **Trainable weights** (gamma, beta): Scale and shift parameters
2. **Running statistics** (mean, variance): Moving averages computed during training

The running statistics were computed on ImageNet's millions of images. If we unfreeze BatchNorm:
- Statistics update based on our small batches (32 images)
- High variance in estimated mean/variance
- Model becomes unstable, accuracy drops

**By keeping BatchNorm frozen**:
- Running statistics stay fixed (ImageNet values)
- Training is stable
- We still update other backbone weights

**Interview Q: What would happen without this?**
> I'd likely see training loss oscillate wildly, validation accuracy drop, and the model might converge to a worse solution or not converge at all. This is a common pitfall in transfer learning.

### Lower Learning Rate

```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),  # 1e-4
    ...
)
```

**Interview Q: Why 10x lower learning rate for fine-tuning?**
> The backbone already has good features. We want to make small adjustments, not large changes. A high learning rate would destroy the pretrained features.

---

## 8. Evaluation

```python
def evaluate_model(model, test_ds, classes):
    y_true = []
    y_pred = []
    
    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))
    
    print(classification_report(y_true, y_pred, target_names=classes, digits=3))
    print(confusion_matrix(y_true, y_pred))
```

**Interview Q: Why collect predictions in a loop instead of model.predict(test_ds)?**
> `model.predict(test_ds)` works but returns predictions without labels. By iterating, I get both predictions and ground truth labels aligned for sklearn metrics.

**Interview Q: What do you look at in the classification report?**
> - **Per-class F1 scores**: Identify weak classes
> - **Precision vs Recall**: Understand error patterns
> - **Macro F1**: Overall performance across classes (not biased by class sizes)

---

## 8.1 Understanding the Confusion Matrix

### What Is a Confusion Matrix?

A **confusion matrix** is a table that visualizes how well a classification model performs by showing the relationship between predicted and actual labels.

```
                    PREDICTED
                 Fr  FL  FR  Re  RL  RR  Bg
           Fr   [27   3   1   0   0   1   1]   ← True Front
           FL   [ 2  93   5   0   1   3   3]   ← True FrontLeft
ACTUAL     FR   [ 1   7  85   0   0   1   0]   ← True FrontRight
           Re   [ 1   0   0  28   2   1   0]   ← True Rear
           RL   [ 0   3   0   2  44   2   0]   ← True RearLeft
           RR   [ 0   1   0   1   1  59   0]   ← True RearRight
           Bg   [ 2   3   2   1   1   0  10]   ← True Background
```

### How to Read It

- **Rows** = Actual (true) class labels
- **Columns** = Predicted class labels
- **Diagonal** = Correct predictions (true positives for each class)
- **Off-diagonal** = Errors (misclassifications)

**Example from above:**
- Row "Front", Column "Front" = 27 → Model correctly predicted 27 Front images
- Row "FrontLeft", Column "FrontRight" = 5 → Model confused 5 FrontLeft images as FrontRight
- Row "Background", Column "Front" = 2 → Model misclassified 2 Background images as Front

### Key Insights from Our Confusion Matrix

| Pattern | What It Means | Why It Happens |
|---------|---------------|----------------|
| High diagonal values | Model is accurate for that class | Good feature learning |
| FrontLeft ↔ FrontRight confusion | Left/right distinction is hard | Similar visual features |
| Background errors scattered | Background is heterogeneous | Defined by absence, not presence |
| Rear confused with RearLeft/RearRight | Subtle angle differences | Overlapping features |

### Interview Q: What does the confusion matrix tell us that accuracy doesn't?

> **Accuracy** gives one number (85.2%). The **confusion matrix** shows:
> 1. **Which classes are confused with each other** - FrontLeft/FrontRight confusion suggests we might need more discriminative features
> 2. **Error patterns** - If all errors are between adjacent viewpoints (FrontLeft ↔ Front), that's reasonable. Random errors would be concerning.
> 3. **Class-specific performance** - Background might have 70% accuracy while RearRight has 95%

### Calculating Metrics from Confusion Matrix

For any class (e.g., FrontLeft):

```
Precision = TP / (TP + FP) = Diagonal / Column Sum
         = "Of all FrontLeft predictions, how many were correct?"

Recall    = TP / (TP + FN) = Diagonal / Row Sum
         = "Of all actual FrontLeft images, how many did we find?"

F1 Score  = 2 × (Precision × Recall) / (Precision + Recall)
         = Harmonic mean of precision and recall
```

**Example for FrontLeft (from matrix above):**
- TP = 93 (diagonal)
- FP = 3+7+0+3+1+3 = 17 (column sum - diagonal)
- FN = 2+5+0+1+3+3 = 14 (row sum - diagonal)
- Precision = 93 / (93 + 17) = 0.845
- Recall = 93 / (93 + 14) = 0.869
- F1 = 2 × (0.845 × 0.869) / (0.845 + 0.869) = 0.857

### Interview Q: When is the confusion matrix more useful than metrics?

> 1. **Debugging specific failures**: "Why is Background F1 so low?" → Check which classes Background is confused with
> 2. **Business decisions**: If confusing Rear with RearLeft is acceptable but confusing Front with Rear is critical, the matrix shows this
> 3. **Data collection guidance**: High FrontLeft ↔ FrontRight confusion suggests collecting more training images with clear left/right distinctions

---

## 8.2 Which Metrics Do We Consider for 2-Phase Training?

### The Final Metrics Question

With two training phases, which metrics should we report?

**Answer: We report metrics from the FINAL model (after Phase 2), evaluated on the TEST set.**

### Why Not Phase 1 Metrics?

| Phase | Purpose | Metrics Meaning |
|-------|---------|-----------------|
| Phase 1 | Train classification head only | Head learns to use frozen ImageNet features |
| Phase 2 | Fine-tune entire model | Model specializes for vehicle viewpoints |
| **Final** | Best val_accuracy model restored | **This is what we deploy** |

Phase 1 metrics (~61% accuracy) show the head is learning, but the backbone hasn't adapted to vehicle features yet. The Phase 2 final model (~85.2% accuracy) is the actual deliverable.

### How EarlyStopping Affects Final Metrics

```python
EarlyStopping(
    monitor='val_accuracy',
    patience=7,
    restore_best_weights=True,  # ← KEY SETTING
    mode='max'
)
```

**Critical**: `restore_best_weights=True` means:
1. Training might run for 32 epochs
2. Best val_accuracy might be at epoch 28
3. After EarlyStopping triggers, model weights are **rolled back to epoch 28**
4. The saved model and reported metrics are from **epoch 28**, not epoch 32

### What We Report

```
Test Set Evaluation (Final Model):
- Overall Accuracy: 85.2%
- Macro F1 Score: 0.819 (average F1 across all classes)
- Weighted F1 Score: 0.854 (F1 weighted by class frequency)
```

### Interview Q: Why Macro F1 and Weighted F1?

> - **Macro F1 (0.819)**: Simple average of all class F1 scores. Treats all classes equally. If Background has F1=0.60, it hurts the macro average even though Background is rare.
>
> - **Weighted F1 (0.854)**: Weighted by class support (sample count). Better reflects overall performance since FrontLeft (largest class) contributes more.
>
> Both are useful:
> - **Macro F1** = "How well do we do on each class equally?"
> - **Weighted F1** = "How well do we do on average for a random image?"

### Interview Q: Why evaluate on test set, not validation set?

> The validation set was used for:
> - EarlyStopping decisions
> - Learning rate adjustments
> - Model checkpointing
>
> The test set was **never seen** during training or hyperparameter tuning. It gives an unbiased estimate of real-world performance. Reporting validation accuracy would be optimistic bias.

### Metrics Summary Table (After Logo Removal)

| Metric | Value | What It Measures | When to Use |
|--------|-------|------------------|-------------|
| Test Accuracy | 85.2% | Overall correctness | General performance |
| Macro F1 | 0.819 | Average class performance | Class-balanced evaluation |
| Weighted F1 | 0.854 | Sample-weighted performance | Real-world performance estimate |
| Per-class F1 | 0.60-0.90 | Individual class performance | Identify weak classes |
| TFLite Agreement | 99.0% | Conversion quality | Deployment validation |
| Avg Confidence | 99.81% | Model certainty | Reliability assessment |

---

## 9. Model Saving

```python
model.save(keras_path)  # .keras format
model.export(SAVED_MODEL_PATH)  # SavedModel format
```

**Interview Q: Why save in two formats?**
> - **Keras format**: For continuing training or loading in Python
> - **SavedModel**: Industry standard for deployment, required for TFLite conversion

---

## 10. What Went Wrong Initially (Debugging Story)

### The 11% Accuracy Problem

In my first attempt, I got only 11% test accuracy. Here's what went wrong and how I fixed it:

| Issue | Symptom | Fix |
|-------|---------|-----|
| MobileNetV3-Small | Training unstable, val_acc oscillating | Switched to MobileNetV2 |
| BatchNorm unfrozen | Fine-tuning caused accuracy drops | Freeze BatchNorm layers |
| High fine-tune LR | Model diverged in Phase 2 | Reduced to 1e-4 |
| Aggressive augmentation | Heavy rotations confused viewpoints | Removed augmentation (minimal in current code) |
| Label order mismatch | Predictions mapped to wrong classes | Load labels from labels.txt consistently |

**Interview Q: How did you debug this?**
> 1. Checked if training loss was decreasing (yes → model learning something)
> 2. Compared train_acc vs val_acc (close → not overfitting, but both low → underfitting or label issue)
> 3. Visualized predictions on sample images (found label mismatches)
> 4. Tried simpler model (MobileNetV2 instead of V3)
> 5. Removed augmentation to isolate issues
> 6. Added BatchNorm freezing (this was the key fix)

---

## 11. Potential Interview Questions

**Q: How would you improve accuracy further?**
> 1. More data augmentation (careful with viewpoint-changing transforms)
> 2. Ensemble multiple models
> 3. Try different backbones (EfficientNet with proper fine-tuning)
> 4. Collect more training data for weak classes (Front, Background)

**Q: Why not use validation data for training?**
> Validation data is for hyperparameter tuning and early stopping. If we train on it, we can't get an unbiased estimate of model performance. It would lead to overfitting to the validation set.

**Q: What if you had 100x more data?**
> I could:
> - Use a larger model (ResNet50, EfficientNetB3)
> - Train from scratch without pretrained weights
> - Add more aggressive augmentation
> - Use larger batch sizes for more stable training

**Q: How would you handle real-time inference?**
> The current model is designed for that:
> - MobileNetV2 is optimized for mobile inference
> - TFLite conversion with Float16 quantization
> - Could further optimize with INT8 quantization or TensorRT
