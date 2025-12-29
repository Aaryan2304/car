"""
SavedModel -> TFLite conversion with Float16 quantization.
Validates by comparing predictions against the original model.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm

# CONFIGURATION

saved_model = "models/saved_model"
tflite_model = "models/model.tflite"
val = "val.csv"
IMG_SIZE = 224

# CONVERSION FUNCTIONS

def load_and_preprocess_image(filepath):
    """Same preprocessing as training: resize + normalize to [-1,1]."""
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)
    img = (img / 127.5) - 1.0
    return img


def representative_dataset_gen():
    """For INT8 quantization calibration (not used by default)."""
    df = pd.read_csv(val)
    filepaths = df['filepath'].tolist()[:100]  # Use subset for calibration
    
    for filepath in filepaths:
        img = load_and_preprocess_image(filepath)
        img = tf.expand_dims(img, 0)
        yield [img]


def convert_to_tflite(quantization='float16'):
    """Convert to TFLite with optional quantization."""
    print(f"\nConverting to TFLite ({quantization})...")
    
    # Try SavedModel first, then Keras format
    if os.path.isdir(saved_model):
        print(f"Loading from SavedModel: {saved_model}")
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model)
    elif os.path.exists(saved_model + '.keras'):
        print(f"Loading from Keras model: {saved_model}.keras")
        model = tf.keras.models.load_model(saved_model + '.keras')
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
    else:
        raise FileNotFoundError(f"No model found at {saved_model}")
    
    # Apply quantization
    if quantization == 'float16':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantization == 'int8':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8
    
    tflite_bytes = converter.convert()
    
    Path(tflite_model).parent.mkdir(parents=True, exist_ok=True)
    with open(tflite_model, 'wb') as f:
        f.write(tflite_bytes)
    
    model_size_mb = os.path.getsize(tflite_model) / (1024 * 1024)
    print(f"TFLite model saved to: {tflite_model}")
    print(f"Model size: {model_size_mb:.2f} MB")
    
    return tflite_model

# VALIDATION FUNCTIONS

def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def predict_tflite(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_dtype = input_details[0]['dtype']
    
    if input_dtype == np.uint8:
        image = (image + 1.0) * 127.5
        image = np.clip(image, 0, 255).astype(np.uint8)
    else:
        image = image.astype(np.float32)
    
    if len(image.shape) == 3:
        image = np.expand_dims(image, 0)
    
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    
    output = interpreter.get_tensor(output_details[0]['index'])
    output_dtype = output_details[0]['dtype']
    if output_dtype == np.uint8:
        scale, zero_point = output_details[0]['quantization']
        output = (output.astype(np.float32) - zero_point) * scale
    
    return output[0]


def validate_conversion(saved_model_path, tflite_model_path, num_samples=100):
    """Check that TFLite and SavedModel agree on predictions."""
    print(f"\nValidating ({num_samples} samples)...")
    
    if os.path.exists(saved_model_path + '.keras'):
        saved_model = tf.keras.models.load_model(saved_model_path + '.keras')
    else:
        saved_model = tf.keras.models.load_model(saved_model_path)
    
    interpreter = load_tflite_model(tflite_model_path)
    df = pd.read_csv(val)
    filepaths = df['filepath'].tolist()[:num_samples]
    
    matches = 0
    for filepath in tqdm(filepaths, desc="Comparing"):
        try:
            img = load_and_preprocess_image(filepath).numpy()
            sm_class = np.argmax(saved_model.predict(np.expand_dims(img, 0), verbose=0)[0])
            tflite_class = np.argmax(predict_tflite(interpreter, img))
            if sm_class == tflite_class:
                matches += 1
        except Exception as e:
            print(f"Warning: {filepath}: {e}")
    
    agreement = matches / len(filepaths) * 100
    print(f"Agreement: {matches}/{len(filepaths)} ({agreement:.1f}%)")
    return agreement

# MAIN

def main():
    print("TFLite Conversion")
    print("=" * 40)
    
    if not os.path.exists(saved_model):
        print(f"Error: No model at {saved_model}. Run train.py first.")
        return
    
    tflite_path = convert_to_tflite(quantization='float16')
    agreement = validate_conversion(saved_model, tflite_path)
    
    if agreement >= 95:
        print(f"\n PASSED ({agreement:.1f}% agreement)")
    else:
        print(f"\n WARNING: only {agreement:.1f}% agreement")
    
    print(f"\nOutputs: {tflite_model}, {saved_model}/labels.txt")


if __name__ == '__main__':
    main()
