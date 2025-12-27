# Test Prediction Script - Deep Dive

## File: `test_predict.py`

This document explains every aspect of the inference script for your interview.

---

## 1. Purpose and Requirements

### This is a Required Deliverable

The assignment explicitly asks for an inference script that:
- Accepts command-line arguments for model, labels, and image folder
- Works with both TFLite and SavedModel
- Outputs predictions to `predictions.csv`

### Output Format

```csv
image_name,prediction,score
image001.jpg,FrontLeft,0.9823
image002.jpg,RearRight,0.8567
...
```

---

## 2. Command-Line Interface

```python
parser = argparse.ArgumentParser(
    description='ClearQuote Vehicle Viewpoint Classifier - Inference Script',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
    python test_predict.py --model models/model.tflite --labels models/saved_model/labels.txt --images dataset/
    """
)
```

**Interview Q: Why use argparse instead of sys.argv directly?**
> - **Automatic help**: `python test_predict.py --help` works
> - **Validation**: Checks required arguments
> - **Cleaner code**: Named arguments instead of positional indices
> - **Default values**: Easy to specify

### Arguments

```python
parser.add_argument('--model', '-m', required=True,
    help='Path to TFLite model (.tflite) or SavedModel directory')

parser.add_argument('--labels', '-l', required=True,
    help='Path to labels.txt file')

parser.add_argument('--images', '-i', required=True,
    help='Path to folder containing images for prediction')

parser.add_argument('--output', '-o', default='predictions.csv',
    help='Output CSV file path (default: predictions.csv)')
```

**Interview Q: Why are model, labels, and images required but output is optional?**
> The script can't do anything without a model, labels, and images. But `predictions.csv` is a sensible default output location. Users can override it if needed.

### Input Validation

```python
if not os.path.exists(args.model):
    print(f"Error: Model not found: {args.model}")
    return 1
```

**Interview Q: Why return 1 instead of raising an exception?**
> CLI scripts conventionally return exit codes:
> - **0**: Success
> - **1**: Error
> This allows shell scripts to check `$?` and handle errors appropriately.

---

## 3. Dual Model Support

### The Challenge

The script must work with two different model formats:
1. **TFLite** (`.tflite`): For edge deployment
2. **SavedModel**: For testing with the original TensorFlow model

Each format has a different inference API.

### The Solution: Unified Prediction Function

```python
def load_model(model_path):
    """
    Load model (TFLite or SavedModel) and return prediction function.
    
    Returns:
        (predict_fn, model_type)
    """
```

**Key insight**: Return a **function** that encapsulates the inference logic. The rest of the code doesn't need to know which model type it's using.

### TFLite Loading

```python
if model_path.endswith('.tflite'):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
```

**Interview Q: Why call `allocate_tensors()` here?**
> It's called once during loading, not for every prediction. This is more efficient than re-allocating for each image.

### Handling Quantized Models

```python
input_dtype = input_details[0]['dtype']
output_dtype = output_details[0]['dtype']
input_scale = output_scale = None
input_zp = output_zp = None

if 'quantization' in input_details[0]:
    quant = input_details[0]['quantization']
    if len(quant) >= 2:
        input_scale, input_zp = quant[0], quant[1]
```

**Interview Q: What are scale and zero_point?**
> For quantized models, float values are mapped to integers:
> ```
> quantized_value = round(float_value / scale) + zero_point
> float_value = (quantized_value - zero_point) × scale
> ```
> We extract these parameters at load time to handle dequantization.

### The TFLite Prediction Function

```python
def predict_tflite(image):
    # Handle quantized input
    if input_dtype == np.uint8:
        image = (image + 1.0) * 127.5  # [-1,1] → [0,255]
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
```

**Interview Q: Why is this a closure?**
> The inner function `predict_tflite` captures variables from the outer scope: `interpreter`, `input_details`, `output_details`, `input_dtype`, etc. This avoids passing these as arguments for every prediction.

### SavedModel Loading

```python
else:
    model = tf.keras.models.load_model(model_path)
    
    def predict_savedmodel(image):
        image = image.astype(np.float32)
        if len(image.shape) == 3:
            image = np.expand_dims(image, 0)
        return model.predict(image, verbose=0)[0]
    
    return predict_savedmodel, 'savedmodel'
```

**Interview Q: Why `verbose=0`?**
> By default, `model.predict()` shows a progress bar. For single-image predictions, this clutters the output. `verbose=0` silences it.

---

## 4. Image Preprocessing

### The Preprocessing Function

```python
def preprocess_image(image_path):
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
```

**Critical**: This preprocessing MUST match training exactly.

**Interview Q: Why check the extension for decoding?**
> `decode_jpeg` can't handle PNG files (different compression). By checking the extension, we use the correct decoder. This prevents errors on mixed-format datasets.

**Interview Q: What if preprocessing differs between training and inference?**
> Predictions will be wrong! Common mistakes:
> - Training: [0, 1] normalization, Inference: [-1, 1] (or vice versa)
> - Training: RGB, Inference: BGR (OpenCV default)
> - Training: resize then crop, Inference: just resize
> 
> Always verify preprocessing matches by checking a few predictions manually.

### Finding Image Files

```python
def get_image_files(folder_path):
    folder = Path(folder_path)
    image_files = []
    
    for ext in SUPPORTED_EXTENSIONS:
        image_files.extend(folder.rglob(f'*{ext}'))
        image_files.extend(folder.rglob(f'*{ext.upper()}'))
    
    return sorted(set(image_files))
```

**`rglob`**: Recursive glob - finds files in all subdirectories.

**Interview Q: Why use both lowercase and uppercase extensions?**
> File systems vary:
> - Linux: case-sensitive (`.jpg` ≠ `.JPG`)
> - Windows/macOS: case-insensitive
> 
> By searching for both, the script works on all systems. Camera photos often have `.JPG` (uppercase).

**`sorted(set(...))`**: Remove duplicates (in case a file matches multiple patterns) and sort for reproducible output.

---

## 5. Main Prediction Loop

```python
def predict_images(model_path, labels_path, images_path, output_path='predictions.csv'):
    # Load model
    predict_fn, model_type = load_model(model_path)
    
    # Load labels
    labels = load_labels(labels_path)
    
    # Get image files
    image_files = get_image_files(images_path)
    
    # Run predictions
    results = []
    
    for image_path in tqdm(image_files, desc="Predicting"):
        try:
            image = preprocess_image(image_path)
            probs = predict_fn(image)
            
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
```

### Error Handling

**Interview Q: Why catch exceptions per-image instead of failing the whole script?**
> Production resilience. If one image is corrupted, we still want predictions for the other 999 images. The ERROR marker in the output makes it easy to identify and investigate failed images.

### Progress Tracking

```python
for image_path in tqdm(image_files, desc="Predicting"):
```

**Interview Q: Why use tqdm?**
> For large datasets (1000+ images), users need feedback. A progress bar shows:
> - How many images are processed
> - Estimated time remaining
> - Processing speed (images/second)

Without it, the script appears frozen.

---

## 6. Output Generation

### Creating the DataFrame

```python
df = pd.DataFrame(results)
df.to_csv(output_path, index=False)
```

**`index=False`**: Don't write row numbers to CSV. The output is cleaner:
```csv
image_name,prediction,score
image1.jpg,FrontLeft,0.9823
```

Instead of:
```csv
,image_name,prediction,score
0,image1.jpg,FrontLeft,0.9823
```

### Summary Statistics

```python
print(f"\nTotal images processed: {len(df)}")
print(f"\nPrediction distribution:")
for label, count in df['prediction'].value_counts().items():
    pct = 100 * count / len(df)
    print(f"  {label:12s}: {count:4d} ({pct:5.1f}%)")

avg_score = df[df['prediction'] != 'ERROR']['score'].mean()
print(f"\nAverage confidence score: {avg_score:.4f}")
```

**Interview Q: Why show prediction distribution?**
> Quick sanity check. If 95% of predictions are "Background," something might be wrong. The distribution should roughly match expected viewpoint frequencies.

**Interview Q: Why exclude ERROR from average score?**
> ERROR entries have score 0.0, which would artificially lower the average. We want the average confidence of successful predictions.

---

## 7. Memory Management

**Interview Q: How does this handle large image folders (100,000+ images)?**

The current implementation:
1. Loads image file paths (small, just strings)
2. Processes one image at a time
3. Stores results as dictionaries (small)

**Memory-efficient**: Only one image is in memory at a time.

For extreme scale, potential improvements:
```python
# Process in chunks
for i in range(0, len(image_files), CHUNK_SIZE):
    chunk = image_files[i:i+CHUNK_SIZE]
    results = process_chunk(chunk)
    append_to_csv(results)  # Append instead of holding all in memory
```

---

## 8. Robustness Features

### Path Handling

```python
img = tf.io.read_file(str(image_path))
```

**`str(image_path)`**: Convert Path object to string. TensorFlow expects string paths.

### Extension Flexibility

```python
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
```

**Interview Q: Why support multiple formats?**
> Real-world datasets are messy. Different cameras, screenshots, converted images. Supporting common formats reduces friction.

### Graceful Degradation

If the model isn't found:
```python
if not os.path.exists(args.model):
    print(f"Error: Model not found: {args.model}")
    return 1
```

Clear error message, non-zero exit code. Doesn't crash with cryptic traceback.

---

## 9. Testing the Script

### Manual Testing

```bash
# Test with TFLite
python test_predict.py --model models/model.tflite --labels models/saved_model/labels.txt --images dataset/5e9112c35026365e15eb871b/

# Test with SavedModel
python test_predict.py --model models/saved_model --labels models/saved_model/labels.txt --images dataset/5e9112c35026365e15eb871b/

# Verify output
head predictions.csv
```

### Validation Against Ground Truth

```python
# If you have ground truth labels:
import pandas as pd
from sklearn.metrics import accuracy_score

pred_df = pd.read_csv('predictions.csv')
truth_df = pd.read_csv('test.csv')

# Merge on filename
merged = pred_df.merge(truth_df, left_on='image_name', right_on='filename')
accuracy = accuracy_score(merged['label'], merged['prediction'])
print(f"Accuracy: {accuracy:.2%}")
```

---

## 10. Potential Interview Questions

**Q: How would you speed up inference?**
> 1. **Batch processing**: Predict multiple images at once instead of one-by-one
> 2. **GPU acceleration**: Use TFLite GPU delegate
> 3. **Multi-threading**: Parallelize image loading and preprocessing
> 4. **Model optimization**: INT8 quantization, pruning

**Q: How would you add batch inference?**
```python
# Pseudocode
batch_size = 32
for i in range(0, len(image_files), batch_size):
    batch_files = image_files[i:i+batch_size]
    batch_images = [preprocess_image(f) for f in batch_files]
    batch_array = np.stack(batch_images)
    batch_probs = model.predict(batch_array)
    # Process results
```

**Q: What if you need to run on a video stream?**
> 1. Use OpenCV to capture frames
> 2. Keep model loaded (don't reload per frame)
> 3. Process every Nth frame for speed
> 4. Smooth predictions over time (temporal consistency)

```python
import cv2

cap = cv2.VideoCapture('video.mp4')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess frame (BGR → RGB, resize, normalize)
    image = preprocess_frame(frame)
    probs = predict_fn(image)
    # Display or save result
```

**Q: How would you add confidence thresholding?**
```python
CONFIDENCE_THRESHOLD = 0.6

pred_score = float(probs[pred_idx])
if pred_score < CONFIDENCE_THRESHOLD:
    pred_label = 'Unknown'
```

This filters out low-confidence predictions that might be wrong.

**Q: How would you deploy this as a web API?**
```python
from flask import Flask, request, jsonify

app = Flask(__name__)
predict_fn, _ = load_model('models/model.tflite')
labels = load_labels('models/saved_model/labels.txt')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    image = preprocess_uploaded_file(file)
    probs = predict_fn(image)
    pred_idx = np.argmax(probs)
    return jsonify({
        'prediction': labels[pred_idx],
        'score': float(probs[pred_idx])
    })
```

---

## 11. Summary

The inference script is designed for:
- **Flexibility**: Works with TFLite and SavedModel
- **Robustness**: Handles errors gracefully
- **Usability**: Clear CLI, progress feedback, summary stats
- **Correctness**: Matches training preprocessing exactly

This is the script that would actually be used in production to classify new vehicle images.
