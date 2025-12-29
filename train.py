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
import matplotlib.pyplot as plt
import seaborn as sns

### Config
PATHS = {
    'train': "train.csv",
    'val': "val.csv",
    'test': "test.csv",
    'labels': "models/saved_model/labels.txt",
    'saved_model': "models/saved_model"
}

# Model params
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 7

# Two-phase training schedule
PHASE1_EPOCHS = 20
PHASE2_EPOCHS = 15
INITIAL_LR = 1e-3
FINE_TUNE_LR = 1e-4  # 10x lower for fine-tuning

LABEL_SMOOTHING = 0.1  # helps with noisy VIA annotations

def load_labels():
    if os.path.exists(PATHS['labels']):
        with open(PATHS['labels'], 'r') as f:
            return [line.strip() for line in f if line.strip()]
    return ['Front', 'FrontLeft', 'FrontRight', 'Rear', 'RearLeft', 'RearRight', 'Background']


# DATA PIPELINE WITH KERAS PREPROCESSING

def create_data_augmentation():
    """
    Create data augmentation layer for training.
    """
    return keras.Sequential([
        layers.RandomBrightness(0.2),    # ±20% brightness variation
        layers.RandomContrast(0.2),      # ±20% contrast variation
        layers.RandomZoom(0.05),         # ±5% zoom
    ], name="data_augmentation")


# Global augmentation layer
augmentation_layer = None


def load_image(filepath, label):
    """Load and resize to 224x224 for MobileNet."""
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)
    return img, label


def augment_image(img, label):
    """Apply augmentation to a single image."""
    global augmentation_layer
    if augmentation_layer is None:
        augmentation_layer = create_data_augmentation()
    img = augmentation_layer(img, training=True)
    return img, label


def preprocess_for_mobilenet(img, label):
    """Normalize to [-1, 1] range expected by MobileNetV2."""
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
    
    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Apply augmentation only during training
    if is_training:
        dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.map(preprocess_for_mobilenet, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    return dataset, len(filepaths)


def compute_class_weights(csv_path, classes):
    """Sklearn balanced weights - penalizes mistakes on rare classes more."""
    df = pd.read_csv(csv_path)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    labels = [class_to_idx[lbl] for lbl in df['label']]
    
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(len(classes)),
        y=labels
    )
    return dict(enumerate(weights))


# MODEL BUILDING

def build_model(num_classes):
    # Tried MobileNetV3-Small first but got only 11% acc - V2 is more stable
    # TODO: revisit V3 with different LR schedule?
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


# TRAINING WITH GRADUAL UNFREEZING

def unfreeze_layers(backbone, num_layers):
    """Unfreeze the last N layers for fine-tuning."""
    backbone.trainable = True
    for layer in backbone.layers[:-num_layers]:
        layer.trainable = False
    return backbone


def plot_training_history(history1, history2, save_path='training_curves.png'):
    """Plot accuracy and loss curves for both training phases."""
    # Combine histories from both phases
    acc = history1.history['accuracy'] + history2.history['accuracy']
    val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    loss = history1.history['loss'] + history2.history['loss']
    val_loss = history1.history['val_loss'] + history2.history['val_loss']
    epochs = range(1, len(acc) + 1)
    phase1_end = len(history1.history['accuracy'])
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy plot
    axes[0].plot(epochs, acc, 'b-', label='Train')
    axes[0].plot(epochs, val_acc, 'r-', label='Validation')
    axes[0].axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.7)
    axes[0].text(phase1_end + 0.5, max(val_acc) * 0.5, 'Phase 2', fontsize=9, color='gray')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Training vs Validation Accuracy')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(epochs, loss, 'b-', label='Train')
    axes[1].plot(epochs, val_loss, 'r-', label='Validation')
    axes[1].axvline(x=phase1_end, color='gray', linestyle='--', alpha=0.7)
    axes[1].text(phase1_end + 0.5, max(val_loss) * 0.7, 'Phase 2', fontsize=9, color='gray')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Training vs Validation Loss')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved to: {save_path}")


def plot_confusion_matrix(y_true, y_pred, classes, save_path='confusion_matrix.png'):
    """Save confusion matrix as a heatmap image."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - Test Set')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to: {save_path}")


def evaluate_model(model, test_ds, classes):
    y_true, y_pred = [], []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=classes, digits=3))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    return y_true, y_pred


# MAIN TRAINING

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
    train_ds, train_size = create_dataset(PATHS['train'], classes, is_training=True)
    val_ds, val_size = create_dataset(PATHS['val'], classes, is_training=False)
    test_ds, test_size = create_dataset(PATHS['test'], classes, is_training=False)
    
    print(f"  Training samples: {train_size}")
    print(f"  Validation samples: {val_size}")
    print(f"  Test samples: {test_size}")
    
    # Compute class weights
    class_weights = compute_class_weights(PATHS['train'], classes)
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
    
    # PHASE 1: Train classification head only

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
    
    # FINAL EVALUATION

    print("\n" + "=" * 70)
    print("FINAL EVALUATION ON TEST SET")
    print("=" * 70)
    
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    y_true, y_pred = evaluate_model(model, test_ds, classes)
    
    # Save visualizations
    plot_training_history(history1, history2, 'training_curves.png')
    plot_confusion_matrix(y_true, y_pred, classes, 'confusion_matrix.png')
    
    # SAVE MODEL

    print("\n" + "=" * 70)
    print("SAVING MODEL")
    print("=" * 70)
    
    Path(PATHS['saved_model']).mkdir(parents=True, exist_ok=True)
    
    # Save in Keras format
    keras_path = PATHS['saved_model'] + '.keras'
    model.save(keras_path)
    print(f"\nKeras model saved to: {keras_path}")
    
    # Export as SavedModel
    model.export(PATHS['saved_model'])
    print(f"SavedModel exported to: {PATHS['saved_model']}")
    
    # Save labels
    labels_path = Path(PATHS['saved_model']) / 'labels.txt'
    with open(labels_path, 'w') as f:
        for cls in classes:
            f.write(f"{cls}\n")
    print(f"Labels saved to: {labels_path}")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)


if __name__ == '__main__':
    train()
