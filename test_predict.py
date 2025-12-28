"""
Inference script for vehicle viewpoint classifier.

Usage:
    python test_predict.py --model models/model.tflite --labels models/saved_model/labels.txt --images <folder>
"""

import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm

# CONFIGURATION

IMG_SIZE = 224
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}

# MODEL LOADING

def load_labels(labels_path):
    with open(labels_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def load_model(model_path):
    """Load TFLite or SavedModel, return predict function and model type."""
    model_path = str(model_path)
    
    if model_path.endswith('.tflite'):
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        input_dtype = input_details[0]['dtype']
        output_dtype = output_details[0]['dtype']
        input_scale = output_scale = None
        input_zp = output_zp = None
        
        # Get quantization params if available
        if 'quantization' in input_details[0]:
            quant = input_details[0]['quantization']
            if len(quant) >= 2:
                input_scale, input_zp = quant[0], quant[1]
        
        if 'quantization' in output_details[0]:
            quant = output_details[0]['quantization']
            if len(quant) >= 2:
                output_scale, output_zp = quant[0], quant[1]
        
        def predict_tflite(image):
            # Handle quantized input
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
            
            if output_dtype == np.uint8 and output_scale is not None:
                output = (output.astype(np.float32) - output_zp) * output_scale
            
            return output[0]
        
        return predict_tflite, 'tflite'
    
    else:
        # Load SavedModel
        model = tf.keras.models.load_model(model_path)
        
        def predict_savedmodel(image):
            image = image.astype(np.float32)
            if len(image.shape) == 3:
                image = np.expand_dims(image, 0)
            return model.predict(image, verbose=0)[0]
        
        return predict_savedmodel, 'savedmodel'


# IMAGE PROCESSING

def preprocess_image(image_path):
    """Same preprocessing as training."""
    img = tf.io.read_file(str(image_path))
    ext = Path(image_path).suffix.lower()
    img = tf.image.decode_png(img, channels=3) if ext == '.png' else tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)
    img = (img / 127.5) - 1.0
    return img.numpy()


def get_image_files(folder_path):
    folder = Path(folder_path)
    files = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(folder.rglob(f'*{ext}'))
        files.extend(folder.rglob(f'*{ext.upper()}'))
    return sorted(set(files))


# MAIN PREDICTION

def predict_images(model_path, labels_path, images_path, output_path='predictions.csv'):
    """Run inference on all images in a folder."""
    print("Viewpoint Classifier - Inference")
    print("=" * 40)
    
    print(f"\nModel: {model_path}")
    predict_fn, model_type = load_model(model_path)
    
    labels = load_labels(labels_path)
    print(f"Classes: {labels}")
    
    image_files = get_image_files(images_path)
    print(f"Found {len(image_files)} images in {images_path}")
    
    if not image_files:
        print("No images found!")
        return None
    
    results = []
    for image_path in tqdm(image_files):
        try:
            image = preprocess_image(image_path)
            probs = predict_fn(image)
            pred_idx = np.argmax(probs)
            results.append({
                'image_name': image_path.name,
                'prediction': labels[pred_idx],
                'score': round(float(probs[pred_idx]), 4)
            })
        except Exception as e:
            print(f"\nError: {image_path.name}: {e}")
            results.append({'image_name': image_path.name, 'prediction': 'ERROR', 'score': 0.0})
    
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    
    print(f"\nSaved to: {output_path}")
    print(f"Distribution:")
    for label, count in df['prediction'].value_counts().items():
        print(f"  {label}: {count} ({100*count/len(df):.1f}%)")
    
    avg_conf = df[df['prediction'] != 'ERROR']['score'].mean()
    print(f"Avg confidence: {avg_conf:.4f}")
    return df


# CLI

def main():
    parser = argparse.ArgumentParser(description='Vehicle viewpoint inference')
    parser.add_argument('--model', '-m', required=True, help='Path to model')
    parser.add_argument('--labels', '-l', required=True, help='Path to labels.txt')
    parser.add_argument('--images', '-i', required=True, help='Folder with images')
    parser.add_argument('--output', '-o', default='predictions.csv', help='Output CSV')
    args = parser.parse_args()
    
    for path, name in [(args.model, 'Model'), (args.labels, 'Labels'), (args.images, 'Images')]:
        if not os.path.exists(path):
            print(f"Error: {name} not found: {path}")
            return 1
    
    df = predict_images(args.model, args.labels, args.images, args.output)
    return 0 if df is not None else 1


if __name__ == '__main__':
    exit(main())
