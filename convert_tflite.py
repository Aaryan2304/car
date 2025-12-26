"""
TFLite Conversion Script for ClearQuote Vehicle Viewpoint Classifier

Converts TensorFlow SavedModel to TFLite format with:
- Float16 quantization (default)
- Optional INT8 quantization with representative dataset

Validates conversion by comparing predictions.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

SAVED_MODEL_PATH = "models/saved_model"
TFLITE_MODEL_PATH = "models/model.tflite"
VAL_CSV = "val.csv"
IMG_SIZE = 224


# =============================================================================
# CONVERSION FUNCTIONS
# =============================================================================

def load_and_preprocess_image(filepath):
    """Load and preprocess image for inference"""
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)
    img = (img / 127.5) - 1.0
    return img


def representative_dataset_gen():
    """Generator for INT8 quantization representative dataset"""
    df = pd.read_csv(VAL_CSV)
    filepaths = df['filepath'].tolist()[:100]  # Use subset for calibration
    
    for filepath in filepaths:
        img = load_and_preprocess_image(filepath)
        img = tf.expand_dims(img, 0)
        yield [img]


def convert_to_tflite(quantization='float16'):
    """
    Convert SavedModel to TFLite format.
    
    Args:
        quantization: 'float16' (default), 'int8', or 'none'
    
    Returns:
        Path to converted TFLite model
    """
    print(f"\nConverting model to TFLite ({quantization} quantization)...")
    
    # Try SavedModel first, then Keras format
    if os.path.isdir(SAVED_MODEL_PATH):
        print(f"Loading from SavedModel: {SAVED_MODEL_PATH}")
        converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_PATH)
    elif os.path.exists(SAVED_MODEL_PATH + '.keras'):
        print(f"Loading from Keras model: {SAVED_MODEL_PATH}.keras")
        model = tf.keras.models.load_model(SAVED_MODEL_PATH + '.keras')
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    else:
        raise FileNotFoundError(f"No model found at {SAVED_MODEL_PATH}")
    
    # Apply quantization
    if quantization == 'float16':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantization == 'int8':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    # else: no quantization
    
    # Convert
    tflite_model = converter.convert()
    
    # Save
    Path(TFLITE_MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(TFLITE_MODEL_PATH, 'wb') as f:
        f.write(tflite_model)
    
    model_size_mb = os.path.getsize(TFLITE_MODEL_PATH) / (1024 * 1024)
    print(f"TFLite model saved to: {TFLITE_MODEL_PATH}")
    print(f"Model size: {model_size_mb:.2f} MB")
    
    return TFLITE_MODEL_PATH


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def load_tflite_model(model_path):
    """Load TFLite model and return interpreter"""
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def predict_tflite(interpreter, image):
    """Run inference with TFLite interpreter"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Get input dtype and adjust if needed
    input_dtype = input_details[0]['dtype']
    
    if input_dtype == np.uint8:
        # Quantized model expects uint8 input
        image = (image + 1.0) * 127.5  # Denormalize from [-1, 1] to [0, 255]
        image = np.clip(image, 0, 255).astype(np.uint8)
    else:
        image = image.astype(np.float32)
    
    # Ensure correct shape
    if len(image.shape) == 3:
        image = np.expand_dims(image, 0)
    
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    
    output = interpreter.get_tensor(output_details[0]['index'])
    
    # Dequantize if needed
    output_dtype = output_details[0]['dtype']
    if output_dtype == np.uint8:
        scale, zero_point = output_details[0]['quantization']
        output = (output.astype(np.float32) - zero_point) * scale
    
    return output[0]


def validate_conversion(saved_model_path, tflite_model_path, num_samples=100):
    """
    Validate TFLite conversion by comparing predictions with SavedModel.
    
    Returns:
        agreement_rate: Percentage of matching top-1 predictions
    """
    print(f"\nValidating TFLite conversion ({num_samples} samples)...")
    
    # Load models - try Keras format first
    if os.path.exists(saved_model_path + '.keras'):
        saved_model = tf.keras.models.load_model(saved_model_path + '.keras')
    else:
        saved_model = tf.keras.models.load_model(saved_model_path)
    
    interpreter = load_tflite_model(tflite_model_path)
    
    # Load validation data
    df = pd.read_csv(VAL_CSV)
    filepaths = df['filepath'].tolist()[:num_samples]
    
    matches = 0
    total = 0
    
    for filepath in tqdm(filepaths, desc="Validating"):
        try:
            # Load image
            img = load_and_preprocess_image(filepath)
            img_array = img.numpy()
            
            # SavedModel prediction
            sm_pred = saved_model.predict(np.expand_dims(img_array, 0), verbose=0)
            sm_class = np.argmax(sm_pred[0])
            
            # TFLite prediction
            tflite_pred = predict_tflite(interpreter, img_array)
            tflite_class = np.argmax(tflite_pred)
            
            if sm_class == tflite_class:
                matches += 1
            total += 1
            
        except Exception as e:
            print(f"Warning: Error processing {filepath}: {e}")
            continue
    
    agreement_rate = matches / total * 100 if total > 0 else 0
    print(f"\nPrediction Agreement: {matches}/{total} ({agreement_rate:.1f}%)")
    
    return agreement_rate


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("ClearQuote Vehicle Viewpoint Classifier - TFLite Conversion")
    print("=" * 60)
    
    # Check if SavedModel exists
    if not os.path.exists(SAVED_MODEL_PATH):
        print(f"Error: SavedModel not found at {SAVED_MODEL_PATH}")
        print("Please run train.py first.")
        return
    
    # Convert to TFLite
    tflite_path = convert_to_tflite(quantization='float16')
    
    # Validate conversion
    agreement = validate_conversion(SAVED_MODEL_PATH, tflite_path)
    
    if agreement >= 95:
        print(f"\n✓ Validation PASSED: {agreement:.1f}% agreement (threshold: 95%)")
    else:
        print(f"\n✗ Validation WARNING: {agreement:.1f}% agreement (below 95% threshold)")
        print("  Consider using different quantization settings.")
    
    # Print summary
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  - {TFLITE_MODEL_PATH}")
    print(f"  - {SAVED_MODEL_PATH}/labels.txt")


if __name__ == '__main__':
    main()
