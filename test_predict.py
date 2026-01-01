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
import matplotlib.pyplot as plt
from PIL import Image

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


def visualize_predictions(df, images_path, ground_truth_df=None, output_dir='prediction_samples'):
    """
    Save prediction grids for review.
    
    If ground_truth_df is provided: shows correct vs incorrect predictions
    Otherwise: shows high vs low confidence predictions
    """
    Path(output_dir).mkdir(exist_ok=True)
    
    valid_df = df[df['prediction'] != 'ERROR'].copy()
    if len(valid_df) < 5:
        print("Not enough valid predictions to visualize.")
        return
    
    def make_grid(subset, title, filename, show_actual=False):
        n = len(subset)
        if n == 0:
            print(f"No samples for: {title}")
            return
        cols = min(5, n)
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.5 * rows))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, (_, row) in enumerate(subset.iterrows()):
            r, c = idx // cols, idx % cols
            ax = axes[r, c]
            
            # Find the actual image file
            matches = list(Path(images_path).rglob(row['image_name']))
            if not matches:
                ax.axis('off')
                continue
            
            img = Image.open(matches[0]).convert('RGB')
            ax.imshow(img)
            
            if show_actual and 'actual' in row:
                # Show predicted vs actual for incorrect predictions
                ax.set_title(f"Pred: {row['prediction']}\nActual: {row['actual']}", 
                           fontsize=9, color='red')
            else:
                color = 'green' if row['score'] > 0.8 else 'orange' if row['score'] > 0.5 else 'red'
                ax.set_title(f"{row['prediction']}\n{row['score']:.2%}", fontsize=10, color=color)
            ax.axis('off')
        
        # Hide empty subplots
        for idx in range(n, rows * cols):
            axes[idx // cols, idx % cols].axis('off')
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        save_path = Path(output_dir) / filename
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
    
    # If ground truth available, show correct vs incorrect
    if ground_truth_df is not None:
        # Build filename -> label lookup from ground truth
        gt_lookup = {}
        for _, row in ground_truth_df.iterrows():
            fname = Path(row['filepath']).name if 'filepath' in row else row.get('filename', '')
            gt_lookup[fname] = row['label']
        
        # Add actual labels and correctness
        valid_df['actual'] = valid_df['image_name'].map(gt_lookup)
        valid_df['correct'] = valid_df['prediction'] == valid_df['actual']
        
        # Filter to only images we have ground truth for
        labeled = valid_df[valid_df['actual'].notna()]
        
        if len(labeled) > 0:
            correct = labeled[labeled['correct']]
            incorrect = labeled[~labeled['correct']]
            
            # Sample up to 10 of each
            correct_sample = correct.nlargest(min(10, len(correct)), 'score')
            incorrect_sample = incorrect.nsmallest(min(10, len(incorrect)), 'score')
            
            make_grid(correct_sample, f'Correct Predictions ({len(correct)}/{len(labeled)} total)', 
                     'correct_predictions.png')
            make_grid(incorrect_sample, f'Incorrect Predictions ({len(incorrect)}/{len(labeled)} total)', 
                     'incorrect_predictions.png', show_actual=True)
            
            acc = len(correct) / len(labeled) * 100
            print(f"Visualized: {len(correct)} correct, {len(incorrect)} incorrect ({acc:.1f}% accuracy)")
            return
        else:
            print("Warning: No matching filenames in ground truth. Falling back to confidence mode.")
    
    # Fallback: high/low confidence (no ground truth)
    top_conf = valid_df.nlargest(min(10, len(valid_df)), 'score')
    low_conf = valid_df.nsmallest(min(10, len(valid_df)), 'score')
    
    make_grid(top_conf, 'High Confidence Predictions', 'high_confidence.png')
    make_grid(low_conf, 'Low Confidence Predictions (Review These)', 'low_confidence.png')


# MAIN PREDICTION

def predict_images(model_path, labels_path, images_path, output_path='predictions.csv', 
                   visualize=False, ground_truth_path=None):
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
    
    if visualize:
        # Load ground truth if provided
        gt_df = None
        if ground_truth_path and os.path.exists(ground_truth_path):
            gt_df = pd.read_csv(ground_truth_path)
            print(f"Using ground truth from: {ground_truth_path}")
        visualize_predictions(df, images_path, gt_df)
    
    return df


# CLI

def main():
    parser = argparse.ArgumentParser(description='Vehicle viewpoint inference')
    parser.add_argument('--model', '-m', required=True, help='Path to model')
    parser.add_argument('--labels', '-l', required=True, help='Path to labels.txt')
    parser.add_argument('--images', '-i', required=True, help='Folder with images')
    parser.add_argument('--output', '-o', default='predictions.csv', help='Output CSV')
    parser.add_argument('--visualize', '-v', action='store_true', 
                        help='Save sample prediction images for review')
    parser.add_argument('--ground-truth', '-g', default=None,
                        help='CSV with ground truth labels (filepath,label) for correct/incorrect viz')
    args = parser.parse_args()
    
    for path, name in [(args.model, 'Model'), (args.labels, 'Labels'), (args.images, 'Images')]:
        if not os.path.exists(path):
            print(f"Error: {name} not found: {path}")
            return 1
    
    df = predict_images(args.model, args.labels, args.images, args.output, 
                        args.visualize, args.ground_truth)
    return 0 if df is not None else 1


if __name__ == '__main__':
    exit(main())
