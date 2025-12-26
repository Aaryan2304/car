"""
Test Prediction Script for ClearQuote Vehicle Viewpoint Classifier

Required deliverable for the assignment.

Usage:
    python test_predict.py --model models/model.tflite --labels models/saved_model/labels.txt --images <folder>
    python test_predict.py --model models/saved_model --labels models/saved_model/labels.txt --images <folder>

Output:
    predictions.csv with columns: image_name, prediction, score
"""

import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

IMG_SIZE = 224
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_labels(labels_path):
    """Load class labels from labels.txt"""
    with open(labels_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def load_model(model_path):
    """
    Load model (TFLite or SavedModel) and return prediction function.
    
    Returns:
        (predict_fn, model_type)
    """
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
            """Run TFLite inference"""
            # Handle quantized input
            if input_dtype == np.uint8:
                # Denormalize from [-1, 1] to [0, 255]
                image = (image + 1.0) * 127.5
                image = np.clip(image, 0, 255).astype(np.uint8)
            else:
                image = image.astype(np.float32)
            
            # Add batch dimension
            if len(image.shape) == 3:
                image = np.expand_dims(image, 0)
            
            interpreter.set_tensor(input_details[0]['index'], image)
            interpreter.invoke()
            
            output = interpreter.get_tensor(output_details[0]['index'])
            
            # Dequantize output if needed
            if output_dtype == np.uint8 and output_scale is not None:
                output = (output.astype(np.float32) - output_zp) * output_scale
            
            return output[0]
        
        return predict_tflite, 'tflite'
    
    else:
        # Load SavedModel
        model = tf.keras.models.load_model(model_path)
        
        def predict_savedmodel(image):
            """Run SavedModel inference"""
            image = image.astype(np.float32)
            if len(image.shape) == 3:
                image = np.expand_dims(image, 0)
            return model.predict(image, verbose=0)[0]
        
        return predict_savedmodel, 'savedmodel'


# =============================================================================
# IMAGE PROCESSING
# =============================================================================

def preprocess_image(image_path):
    """Load and preprocess image for inference"""
    img = tf.io.read_file(str(image_path))
    
    # Try to decode based on extension
    ext = Path(image_path).suffix.lower()
    if ext == '.png':
        img = tf.image.decode_png(img, channels=3)
    else:
        img = tf.image.decode_jpeg(img, channels=3)
    
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)
    img = (img / 127.5) - 1.0  # Normalize to [-1, 1]
    
    return img.numpy()


def get_image_files(folder_path):
    """Get all image files from folder (recursively)"""
    folder = Path(folder_path)
    image_files = []
    
    for ext in SUPPORTED_EXTENSIONS:
        image_files.extend(folder.rglob(f'*{ext}'))
        image_files.extend(folder.rglob(f'*{ext.upper()}'))
    
    return sorted(set(image_files))


# =============================================================================
# MAIN PREDICTION
# =============================================================================

def predict_images(model_path, labels_path, images_path, output_path='predictions.csv'):
    """
    Run predictions on all images in folder.
    
    Args:
        model_path: Path to TFLite or SavedModel
        labels_path: Path to labels.txt
        images_path: Path to folder with images
        output_path: Path for output CSV
    
    Returns:
        DataFrame with predictions
    """
    print("=" * 60)
    print("ClearQuote Vehicle Viewpoint Classifier - Inference")
    print("=" * 60)
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    predict_fn, model_type = load_model(model_path)
    print(f"  Model type: {model_type}")
    
    # Load labels
    print(f"Loading labels from: {labels_path}")
    labels = load_labels(labels_path)
    print(f"  Classes ({len(labels)}): {labels}")
    
    # Get image files
    print(f"\nScanning for images in: {images_path}")
    image_files = get_image_files(images_path)
    print(f"  Found {len(image_files)} images")
    
    if len(image_files) == 0:
        print("Error: No images found!")
        return None
    
    # Run predictions
    print("\nRunning predictions...")
    results = []
    
    for image_path in tqdm(image_files, desc="Predicting"):
        try:
            # Preprocess
            image = preprocess_image(image_path)
            
            # Predict
            probs = predict_fn(image)
            
            # Get top prediction
            pred_idx = np.argmax(probs)
            pred_label = labels[pred_idx]
            pred_score = float(probs[pred_idx])
            
            results.append({
                'image_name': image_path.name,
                'prediction': pred_label,
                'score': round(pred_score, 4)
            })
            
        except Exception as e:
            print(f"\nWarning: Error processing {image_path.name}: {e}")
            results.append({
                'image_name': image_path.name,
                'prediction': 'ERROR',
                'score': 0.0
            })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)
    print(f"\nTotal images processed: {len(df)}")
    print(f"\nPrediction distribution:")
    for label, count in df['prediction'].value_counts().items():
        pct = 100 * count / len(df)
        print(f"  {label:12s}: {count:4d} ({pct:5.1f}%)")
    
    avg_score = df[df['prediction'] != 'ERROR']['score'].mean()
    print(f"\nAverage confidence score: {avg_score:.4f}")
    
    return df


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='ClearQuote Vehicle Viewpoint Classifier - Inference Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python test_predict.py --model models/model.tflite --labels models/saved_model/labels.txt --images dataset/
    python test_predict.py --model models/saved_model --labels models/saved_model/labels.txt --images test_images/
        """
    )
    
    parser.add_argument(
        '--model', '-m',
        required=True,
        help='Path to TFLite model (.tflite) or SavedModel directory'
    )
    
    parser.add_argument(
        '--labels', '-l',
        required=True,
        help='Path to labels.txt file'
    )
    
    parser.add_argument(
        '--images', '-i',
        required=True,
        help='Path to folder containing images for prediction'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='predictions.csv',
        help='Output CSV file path (default: predictions.csv)'
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        return 1
    
    if not os.path.exists(args.labels):
        print(f"Error: Labels file not found: {args.labels}")
        return 1
    
    if not os.path.exists(args.images):
        print(f"Error: Images folder not found: {args.images}")
        return 1
    
    # Run predictions
    df = predict_images(
        model_path=args.model,
        labels_path=args.labels,
        images_path=args.images,
        output_path=args.output
    )
    
    if df is None:
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
