# TFLite Conversion Script - Deep Dive

## File: `convert_tflite.py`

This document explains every aspect of the TFLite conversion script for your interview.

---

## 1. Why TFLite?

### The Deployment Goal

The assignment requires **edge/mobile deployment**. TensorFlow models (SavedModel format) are:
- Large (20+ MB for MobileNetV2)
- Require TensorFlow runtime (heavy dependency)
- Slow on mobile devices without optimization

TFLite (TensorFlow Lite) is designed for:
- **Mobile phones** (Android/iOS)
- **Edge devices** (Raspberry Pi, embedded systems)
- **Browsers** (via TensorFlow.js)

### TFLite Benefits

| Aspect | SavedModel | TFLite |
|--------|------------|--------|
| Size | ~21 MB | ~4.5 MB (Float16) |
| Runtime | Full TensorFlow | TFLite Interpreter (lightweight) |
| Inference Speed | Baseline | 2-4x faster on mobile |
| Dependencies | tensorflow package | tflite-runtime (tiny) |

---

## 2. Quantization Options

### Float16 Quantization (Our Choice)

```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
```

**What happens**:
- Weights are stored as 16-bit floats instead of 32-bit
- Model size reduced by ~50%
- Accuracy loss: negligible (usually <0.1%)
- Inference: Hardware accelerated on GPUs with FP16 support

**Interview Q: Why Float16 instead of INT8?**
> - **Simplicity**: No representative dataset needed
> - **Zero accuracy loss**: In my testing, 100% prediction agreement with SavedModel
> - **Good compression**: 21 MB → 4.5 MB
> - **Compatibility**: Works on all devices

### INT8 Quantization (Alternative)

```python
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
```

**What happens**:
- Weights AND activations quantized to 8-bit integers
- Model size: ~2 MB (4x smaller than Float16)
- Requires calibration data (representative dataset)
- Potential accuracy loss (needs careful validation)

**Interview Q: What is the representative dataset for?**
> INT8 quantization needs to know the typical range of activation values. The representative dataset (100 sample images) is fed through the model to compute min/max ranges for each layer. These ranges are used to scale float values to int8.

```python
def representative_dataset_gen():
    df = pd.read_csv(VAL_CSV)
    filepaths = df['filepath'].tolist()[:100]
    
    for filepath in filepaths:
        img = load_and_preprocess_image(filepath)
        img = tf.expand_dims(img, 0)
        yield [img]
```

**Interview Q: Why only 100 samples?**
> More samples would give more accurate range estimates but take longer. 100 is usually sufficient if the samples are representative of real inputs. Using validation data (not training data) ensures we don't overfit the quantization.

---

## 3. Conversion Process

### Loading the Model

```python
if os.path.isdir(SAVED_MODEL_PATH):
    converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_PATH)
elif os.path.exists(SAVED_MODEL_PATH + '.keras'):
    model = tf.keras.models.load_model(SAVED_MODEL_PATH + '.keras')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
```

**Interview Q: Why support both formats?**
> Flexibility. The training script saves both `.keras` and SavedModel formats. Converting from SavedModel is preferred (official deployment format), but the Keras fallback handles edge cases.

### The Conversion

```python
tflite_model = converter.convert()

with open(TFLITE_MODEL_PATH, 'wb') as f:
    f.write(tflite_model)
```

**Interview Q: What does `converter.convert()` return?**
> A bytes object containing the serialized FlatBuffer representation of the TFLite model. FlatBuffer is a memory-efficient serialization format that allows zero-copy access to data.

---

## 4. Validation: Why It Matters

### The Agreement Test

```python
def validate_conversion(saved_model_path, tflite_model_path, num_samples=100):
```

**Purpose**: Ensure the TFLite model produces the same predictions as the original.

**Interview Q: What could cause disagreement?**
> 1. **Quantization error**: INT8 can shift predictions slightly
> 2. **Operator differences**: Some TensorFlow ops have different TFLite implementations
> 3. **Floating-point precision**: Slight differences in computation order
> 4. **Preprocessing mismatch**: Input normalization done differently

### Loading TFLite Model

```python
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter
```

**Interview Q: What is `allocate_tensors()`?**
> It allocates memory for the model's input and output tensors. Must be called before inference. TFLite uses static memory allocation for efficiency.

### Running TFLite Inference

```python
def predict_tflite(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
```

**`input_details`** contains:
- `index`: Tensor index for `set_tensor()`
- `dtype`: Expected data type (float32, uint8, etc.)
- `shape`: Expected input shape [1, 224, 224, 3]
- `quantization`: Scale and zero_point for quantized models

```python
    input_dtype = input_details[0]['dtype']
    
    if input_dtype == np.uint8:
        # Quantized model expects uint8 input
        image = (image + 1.0) * 127.5  # Denormalize from [-1, 1] to [0, 255]
        image = np.clip(image, 0, 255).astype(np.uint8)
    else:
        image = image.astype(np.float32)
```

**Interview Q: Why denormalize for uint8 input?**
> INT8 quantized models expect uint8 input in range [0, 255]. Our preprocessing normalizes to [-1, 1]. We reverse that transformation for quantized models.

**The math**:
- Original: `[-1, 1]`
- After `+ 1.0`: `[0, 2]`
- After `* 127.5`: `[0, 255]`
- Clip and cast to uint8

```python
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
```

**Interview Q: What is `invoke()`?**
> It runs the forward pass through the model. Unlike Keras `model.predict()`, TFLite uses an explicit invoke step after setting input tensors.

### Dequantizing Output

```python
    if output_dtype == np.uint8:
        scale, zero_point = output_details[0]['quantization']
        output = (output.astype(np.float32) - zero_point) * scale
```

**Interview Q: What is dequantization?**
> INT8 quantized outputs are integers in range [0, 255]. To get actual probability values [0, 1], we apply:
```
real_value = (quantized_value - zero_point) × scale
```
Where `scale` and `zero_point` are stored in the model during quantization.

---

## 5. Comparing Predictions

```python
for filepath in tqdm(filepaths, desc="Validating"):
    # Load and preprocess image
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
```

**Interview Q: Why compare top-1 classes instead of probability values?**
> Probability values might differ slightly due to floating-point precision. What matters for deployment is whether the predicted class is the same. 100% top-1 agreement means the TFLite model makes identical decisions.

**Interview Q: What agreement threshold is acceptable?**
> - **≥95%**: Good for deployment
> - **<95%**: Indicates potential issues with quantization
> 
> We achieved **100% agreement**, which is ideal.

---

## 6. The Main Flow

```python
def main():
    # Check if SavedModel exists
    if not os.path.exists(SAVED_MODEL_PATH):
        print("Error: SavedModel not found. Run train.py first.")
        return
    
    # Convert
    tflite_path = convert_to_tflite(quantization='float16')
    
    # Validate
    agreement = validate_conversion(SAVED_MODEL_PATH, tflite_path)
    
    # Report results
    if agreement >= 95:
        print(f"✓ Validation PASSED: {agreement:.1f}%")
    else:
        print(f"✗ Validation WARNING: {agreement:.1f}%")
```

**Interview Q: What would you do if validation fails?**
> 1. Try different quantization (Float16 instead of INT8, or no quantization)
> 2. Check for unsupported operators
> 3. Verify preprocessing is identical
> 4. Inspect which samples disagree and look for patterns

---

## 7. TFLite Internals (Advanced)

### FlatBuffer Format

TFLite uses FlatBuffers for serialization:
- **Zero-copy access**: No parsing needed, read directly from memory
- **Cross-platform**: Same binary works on Android, iOS, embedded
- **Compact**: Efficient binary encoding

### Op Fusion

During conversion, TFLite may fuse multiple operations:
- Conv2D + BatchNorm + ReLU → Single fused operation
- This reduces memory bandwidth and improves speed

**Interview Q: Does fusion affect accuracy?**
> No, fusion is mathematically equivalent. It just changes how operations are executed, not the results.

### Delegate Support

TFLite can use hardware accelerators:
- **GPU Delegate**: Runs on mobile GPU (faster for large models)
- **NNAPI Delegate**: Android Neural Networks API
- **Core ML Delegate**: iOS Neural Engine
- **Edge TPU Delegate**: Google Coral accelerator

**Interview Q: How would you use GPU acceleration?**
```python
# At inference time
interpreter = tf.lite.Interpreter(
    model_path='model.tflite',
    experimental_delegates=[tf.lite.experimental.load_delegate('libdelegate.so')]
)
```

---

## 8. Potential Interview Questions

**Q: How would you further reduce model size?**
> 1. INT8 quantization (requires representative dataset)
> 2. Pruning (remove unimportant weights, then fine-tune)
> 3. Knowledge distillation (train smaller model to mimic larger one)
> 4. Use MobileNetV3-Small as backbone (after fixing training issues)

**Q: What's the difference between post-training and quantization-aware training?**
> - **Post-training quantization** (what we did): Quantize after training is complete
> - **Quantization-aware training (QAT)**: Simulate quantization during training, model learns to be robust to quantization errors
> QAT usually gives better accuracy but requires modifying the training process.

**Q: How would you deploy this on Android?**
> 1. Copy `model.tflite` and `labels.txt` to app's assets folder
> 2. Use TensorFlow Lite Android library
> 3. Load interpreter, allocate tensors
> 4. Preprocess camera image to match training (224×224, normalize to [-1, 1])
> 5. Run inference, get predicted class

**Q: What if a TensorFlow operation isn't supported in TFLite?**
> Options:
> 1. Use `tf.lite.OpsSet.SELECT_TF_OPS` to include TensorFlow ops (increases model size)
> 2. Replace unsupported op with TFLite-compatible alternative in the original model
> 3. Implement custom TFLite operator (advanced)

**Q: How would you benchmark inference speed?**
```python
import time

# Warmup
for _ in range(10):
    interpreter.invoke()

# Benchmark
start = time.time()
for _ in range(100):
    interpreter.invoke()
elapsed = (time.time() - start) / 100
print(f"Average inference time: {elapsed*1000:.2f} ms")
```

---

## 9. Summary of Our Conversion

| Metric | Value |
|--------|-------|
| Quantization | Float16 |
| Original Size | ~21 MB |
| TFLite Size | 4.57 MB |
| Size Reduction | ~78% |
| Prediction Agreement | 100% |
| Accuracy Impact | None |

The conversion was successful with zero accuracy loss, making the model ready for edge deployment.
