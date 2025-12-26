"""
Vehicle Viewpoint Classification Training Script
ClearQuote CV Engineer Assignment

Uses MobileNetV2 with transfer learning for 7-class viewpoint classification.
Two-phase training: frozen backbone followed by fine-tuning.
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
TRAIN_CSV = "train.csv"
VAL_CSV = "val.csv"
TEST_CSV = "test.csv"
LABELS_FILE = "models/saved_model/labels.txt"
SAVED_MODEL_PATH = "models/saved_model"

# Model hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 7

# Training phases
PHASE1_EPOCHS = 20   # Frozen backbone
PHASE2_EPOCHS = 15   # Gradual unfreezing
INITIAL_LR = 1e-3
FINE_TUNE_LR = 1e-4

# Label smoothing helps with noisy labels
LABEL_SMOOTHING = 0.1

# =============================================================================
# LOAD LABELS
# =============================================================================

def load_labels():
    """Load class labels from labels.txt"""
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    return ['Front', 'FrontLeft', 'FrontRight', 'Rear', 'RearLeft', 'RearRight', 'Background']


# =============================================================================
# DATA PIPELINE WITH KERAS PREPROCESSING
# =============================================================================

def create_data_augmentation():
    """Create data augmentation layer for training"""
    return keras.Sequential([
        layers.RandomFlip("horizontal"),  # We'll handle label swap separately
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
        layers.RandomBrightness(0.1),
        layers.RandomContrast(0.1),
    ], name="data_augmentation")


def load_image(filepath, label):
    """Load and preprocess a single image"""
    # Read file
    img = tf.io.read_file(filepath)
    
    # Decode image
    img = tf.image.decode_jpeg(img, channels=3)
    
    # Resize
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    
    # Convert to float32 [0, 255]
    img = tf.cast(img, tf.float32)
    
    return img, label


def preprocess_for_efficientnet(img, label):
    """Apply EfficientNet preprocessing: scale to [0, 255] then apply tf.keras.applications preprocessing"""
    # EfficientNet expects [0, 255] and applies its own normalization
    # But we'll use the standard [-1, 1] normalization which works well
    img = (img / 127.5) - 1.0
    return img, label


def create_dataset(csv_path, classes, is_training=False):
    """Create tf.data.Dataset from CSV"""
    df = pd.read_csv(csv_path)
    
    # Create mappings
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    filepaths = df['filepath'].tolist()
    labels = [class_to_idx[lbl] for lbl in df['label']]
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=len(filepaths), reshuffle_each_iteration=True)
    
    # Load and preprocess images
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(preprocess_for_efficientnet, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch
    dataset = dataset.batch(BATCH_SIZE)
    
    # Prefetch
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset, len(filepaths)


def compute_class_weights(csv_path, classes):
    """Compute class weights for imbalanced dataset"""
    df = pd.read_csv(csv_path)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    labels = [class_to_idx[lbl] for lbl in df['label']]
    
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(len(classes)),
        y=labels
    )
    
    return dict(enumerate(weights))


# =============================================================================
# MODEL BUILDING
# =============================================================================

def build_model(num_classes):
    """
    Build MobileNetV2-based classifier for viewpoint detection.
    """
    base_model = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    base_model.trainable = False
    
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs, training=False)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    return model, base_model


# =============================================================================
# TRAINING WITH GRADUAL UNFREEZING
# =============================================================================

def unfreeze_layers(backbone, num_layers):
    """Unfreeze the last num_layers of the backbone"""
    # First, freeze all layers
    backbone.trainable = True
    
    # Then unfreeze only the last num_layers
    for layer in backbone.layers[:-num_layers]:
        layer.trainable = False
    
    return backbone


def evaluate_model(model, test_ds, classes):
    """Evaluate model and print classification report"""
    # Get predictions
    y_true = []
    y_pred = []
    
    for images, labels in test_ds:
        predictions = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(predictions, axis=1))
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=classes, digits=3))
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    return y_true, y_pred


# =============================================================================
# MAIN TRAINING
# =============================================================================

def train():
    """Main training function"""
    
    print("=" * 70)
    print("ClearQuote Vehicle Viewpoint Classifier - Training")
    print("=" * 70)
    
    # Check for GPU
    gpus = tf.config.list_physical_devices('GPU')
    print(f"\nGPUs available: {len(gpus)}")
    
    # Load labels
    classes = load_labels()
    print(f"\nClasses ({len(classes)}): {classes}")
    
    # Create datasets
    print("\nLoading datasets...")
    train_ds, train_size = create_dataset(TRAIN_CSV, classes, is_training=True)
    val_ds, val_size = create_dataset(VAL_CSV, classes, is_training=False)
    test_ds, test_size = create_dataset(TEST_CSV, classes, is_training=False)
    
    print(f"  Training samples: {train_size}")
    print(f"  Validation samples: {val_size}")
    print(f"  Test samples: {test_size}")
    
    # Compute class weights
    class_weights = compute_class_weights(TRAIN_CSV, classes)
    print(f"\nClass weights:")
    for i, w in class_weights.items():
        print(f"  {classes[i]}: {w:.3f}")
    
    # Build model
    print("\nBuilding model (MobileNetV2)...")
    model, backbone = build_model(len(classes))
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=7,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        ),
        ModelCheckpoint(
            filepath='models/best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            mode='max'
        )
    ]
    
    # =========================================================================
    # PHASE 1: Train classification head only
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 1: Training classification head (backbone frozen)")
    print("=" * 70)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=INITIAL_LR),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE1_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate after phase 1
    print("\n--- Phase 1 Evaluation ---")
    evaluate_model(model, val_ds, classes)
    
    # =========================================================================
    # PHASE 2: Fine-tune with gradual unfreezing
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: Fine-tuning (gradual unfreezing)")
    print("=" * 70)
    
    # Unfreeze top layers of backbone
    backbone.trainable = True
    
    # Freeze batch normalization layers (important for fine-tuning)
    for layer in backbone.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )
    
    # Continue training
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=PHASE2_EPOCHS,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    # =========================================================================
    # FINAL EVALUATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 70)
    
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    evaluate_model(model, test_ds, classes)
    
    # =========================================================================
    # SAVE MODEL
    # =========================================================================
    print("\n" + "=" * 70)
    print("SAVING MODEL")
    print("=" * 70)
    
    Path(SAVED_MODEL_PATH).mkdir(parents=True, exist_ok=True)
    
    # Save in Keras format
    keras_path = SAVED_MODEL_PATH + '.keras'
    model.save(keras_path)
    print(f"\nKeras model saved to: {keras_path}")
    
    # Export as SavedModel
    model.export(SAVED_MODEL_PATH)
    print(f"SavedModel exported to: {SAVED_MODEL_PATH}")
    
    # Save labels
    labels_path = Path(SAVED_MODEL_PATH) / 'labels.txt'
    with open(labels_path, 'w') as f:
        for cls in classes:
            f.write(f"{cls}\n")
    print(f"Labels saved to: {labels_path}")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    train()
