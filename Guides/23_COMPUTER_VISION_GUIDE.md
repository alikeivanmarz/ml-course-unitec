# Computer Vision Guide

Computer vision enables machines to interpret and understand images. This guide covers image handling, convolutional neural networks, transfer learning, and object detection as used in Weeks 3 and 4 of the course.

**Table of Contents**

1. [Image Basics](#1-image-basics)
2. [Loading and Displaying Images](#2-loading-and-displaying-images)
3. [Image Preprocessing](#3-image-preprocessing)
4. [CNN Architecture Explained](#4-cnn-architecture-explained)
5. [Building CNNs in Keras](#5-building-cnns-in-keras)
6. [Transfer Learning](#6-transfer-learning)
7. [Using Pre-trained Models](#7-using-pre-trained-models)
8. [Fine-Tuning Strategies](#8-fine-tuning-strategies)
9. [Object Detection with YOLO](#9-object-detection-with-yolo)
10. [Data Augmentation](#10-data-augmentation)
11. [Quick Reference Tables](#11-quick-reference-tables)
12. [Resources](#12-resources)

---

## 1. Image Basics

### 1.1 Pixels, Channels, and Color Spaces

An image is a **NumPy array** of pixel values.

- **Grayscale:** 2D array `(height, width)` -- values 0 (black) to 255 (white)
- **Color (RGB):** 3D array `(height, width, 3)` -- three channels: Red, Green, Blue
- **Color (BGR):** OpenCV loads images in BGR order, not RGB

```python
import numpy as np

# Grayscale image: shape (100, 100)
gray_image = np.zeros((100, 100), dtype=np.uint8)  # Black image

# Color image: shape (100, 100, 3)
color_image = np.zeros((100, 100, 3), dtype=np.uint8)  # Black RGB image
```

### 1.2 Image as a NumPy Array

```python
print(f"Shape:    {image.shape}")      # (height, width, channels) e.g. (224, 224, 3)
print(f"Dtype:    {image.dtype}")      # uint8 (0-255) or float32 (0.0-1.0)
print(f"Min/Max:  {image.min()}, {image.max()}")
print(f"Channels: {image.shape[2] if image.ndim == 3 else 1}")
```

---

## 2. Loading and Displaying Images

### 2.1 Using OpenCV

```python
import cv2
import matplotlib.pyplot as plt

# Load image (OpenCV loads as BGR, not RGB!)
img_bgr = cv2.imread('path/to/image.jpg')

# Convert BGR to RGB for display
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

print(f"Shape: {img_rgb.shape}")  # (height, width, 3)

# Display
plt.figure(figsize=(6, 6))
plt.imshow(img_rgb)
plt.title('Sample Image')
plt.axis('off')
plt.show()
```

> **Note:** Always convert BGR to RGB before displaying with matplotlib. If colors look wrong (blue faces, orange sky), you forgot this step.

### 2.2 Using PIL/Pillow

```python
from PIL import Image
import numpy as np

# Load image (already in RGB)
img = Image.open('path/to/image.jpg')

# Convert to NumPy array
img_array = np.array(img)
print(f"Shape: {img_array.shape}")

# Display
plt.imshow(img_array)
plt.axis('off')
plt.show()
```

### 2.3 Using Keras Utilities

```python
from tensorflow.keras.utils import load_img, img_to_array

# Load and resize
img = load_img('path/to/image.jpg', target_size=(224, 224))

# Convert to array
img_array = img_to_array(img)  # shape: (224, 224, 3), values 0-255
print(f"Shape: {img_array.shape}")
```

---

## 3. Image Preprocessing

### 3.1 Resizing

Most neural networks expect a fixed input size.

```python
# OpenCV
img_resized = cv2.resize(img_rgb, (224, 224))  # (width, height) order!

# PIL
img_resized = img.resize((224, 224))

# TensorFlow
img_resized = tf.image.resize(img_array, [224, 224])
```

### 3.2 Normalization

Neural networks work best with values in the range [0, 1] or [-1, 1].

```python
# Simple normalization: divide by 255
img_normalized = img_array / 255.0  # Scales from [0, 255] to [0.0, 1.0]

# ImageNet normalization (for pre-trained models)
from tensorflow.keras.applications.resnet50 import preprocess_input
img_preprocessed = preprocess_input(img_array)  # Handles mean subtraction etc.
```

### 3.3 Preparing a Batch

Neural networks expect a **batch dimension**: `(batch_size, height, width, channels)`.

```python
# Single image: add batch dimension
img_batch = np.expand_dims(img_normalized, axis=0)  # (1, 224, 224, 3)

# Or use reshape
img_batch = img_normalized.reshape(1, 224, 224, 3)
```

---

## 4. CNN Architecture Explained

### 4.1 Convolution Layer

A convolution layer slides small **filters** (kernels) across the image to detect features like edges, textures, and shapes.

- **Filters:** learnable weight matrices (e.g., 3x3)
- **Feature maps:** output of applying filters to the input
- **Early layers:** detect simple features (edges, colors)
- **Deeper layers:** detect complex features (eyes, wheels, faces)

### 4.2 Pooling Layer

Reduces the spatial size of feature maps, keeping the most important information.

- **MaxPooling:** takes the maximum value in each window (most common)
- **AveragePooling:** takes the average value

### 4.3 Complete CNN Flow

```
Input Image (224 x 224 x 3)
    |
    v
[Conv2D] --> Feature extraction (filters detect patterns)
    |
    v
[MaxPooling2D] --> Reduce spatial size
    |
    v
[Conv2D] --> Extract higher-level features
    |
    v
[MaxPooling2D] --> Further reduce size
    |
    v
[Flatten] --> Convert 2D feature maps to 1D vector
    |
    v
[Dense] --> Learn combinations of features
    |
    v
[Dense (softmax)] --> Output class probabilities
```

---

## 5. Building CNNs in Keras

### 5.1 Simple CNN for Image Classification

```python
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    # First convolutional block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    # Second convolutional block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Third convolutional block
    layers.Conv2D(64, (3, 3), activation='relu'),

    # Classification head
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')  # 10 classes
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
```

### 5.2 Understanding model.summary() Output

```
Layer (type)                 Output Shape              Param #
=================================================================
conv2d (Conv2D)              (None, 30, 30, 32)        896
max_pooling2d (MaxPool2D)    (None, 15, 15, 32)        0
conv2d_1 (Conv2D)            (None, 13, 13, 64)        18496
...
```

- **Output Shape:** `None` = batch size, then spatial dimensions, then filters
- **Param #:** number of trainable weights in that layer

### 5.3 Training a CNN

```python
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=64,
    validation_split=0.2,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ]
)
```

---

## 6. Transfer Learning

### 6.1 What is Transfer Learning?

Instead of training a CNN from scratch (which requires millions of images), you start with a model that has already learned from a large dataset (ImageNet: 1.2 million images, 1000 classes).

**Two approaches:**

| Approach | What You Do | When to Use |
|----------|-------------|-------------|
| **Feature extraction** | Freeze base model, train only new head | Small dataset, similar domain |
| **Fine-tuning** | Unfreeze some base layers, retrain | Medium dataset, somewhat different domain |

### 6.2 Why Transfer Learning Works

- Early layers learn **universal features** (edges, textures, colors) -- useful for any image task
- Later layers learn **task-specific features** -- these may need retraining
- Much faster than training from scratch
- Works with much smaller datasets (hundreds instead of millions)

---

## 7. Using Pre-trained Models

### 7.1 Feature Extraction with ResNet50

```python
from tensorflow import keras
from tensorflow.keras import layers

# Load pre-trained base model (without the classification head)
base_model = keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze all base model layers
base_model.trainable = False

# Add new classification head
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
```

### 7.2 VGG16

```python
base_model = keras.applications.VGG16(
    weights='imagenet', include_top=False, input_shape=(224, 224, 3)
)
base_model.trainable = False
```

### 7.3 MobileNetV2

Lightweight model, good for deployment on mobile or edge devices.

```python
base_model = keras.applications.MobileNetV2(
    weights='imagenet', include_top=False, input_shape=(224, 224, 3)
)
base_model.trainable = False
```

### 7.4 EfficientNetB0

Good accuracy-to-size ratio.

```python
base_model = keras.applications.EfficientNetB0(
    weights='imagenet', include_top=False, input_shape=(224, 224, 3)
)
base_model.trainable = False
```

### 7.5 Model Comparison

| Model | Parameters | Top-1 Accuracy | Speed | Best For |
|-------|-----------|---------------|-------|----------|
| MobileNetV2 | 3.4M | 71.3% | Fast | Mobile/edge deployment |
| EfficientNetB0 | 5.3M | 77.1% | Medium | Best accuracy-to-size ratio |
| ResNet50 | 25.6M | 76.0% | Medium | General purpose |
| VGG16 | 138M | 71.3% | Slow | Feature extraction |

### 7.6 Preprocessing for Pre-trained Models

Each model expects input preprocessed in a specific way.

```python
# ResNet50 preprocessing
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

# MobileNetV2 preprocessing
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobile_preprocess

# Apply preprocessing
img = keras.utils.load_img('image.jpg', target_size=(224, 224))
img_array = keras.utils.img_to_array(img)
img_batch = np.expand_dims(img_array, axis=0)
img_preprocessed = resnet_preprocess(img_batch)
```

---

## 8. Fine-Tuning Strategies

### 8.1 Phase 1: Train the Head

First train only the new classification head with the base frozen.

```python
base_model.trainable = False

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_split=0.2)
```

### 8.2 Phase 2: Unfreeze and Fine-Tune

Then unfreeze the last few layers of the base model and train with a **lower learning rate**.

```python
# Unfreeze the base model
base_model.trainable = True

# Freeze all layers except the last 20
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Recompile with a LOWER learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),  # Much lower than default
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=10, validation_split=0.2)
```

> **Note:** Use a learning rate 10-100x smaller for fine-tuning (e.g., 1e-5 instead of 1e-3) to avoid destroying the pre-learned weights.

---

## 9. Object Detection with YOLO

### 9.1 What is YOLO?

**YOLO** (You Only Look Once) detects objects and their locations in images in real time. Unlike classification (which says "this is a dog"), detection says "there is a dog at coordinates (x1, y1, x2, y2)."

### 9.2 Using Ultralytics YOLOv8

```python
from ultralytics import YOLO

# Load a pre-trained model
model = YOLO('yolov8n.pt')  # n=nano (fastest), s=small, m=medium, l=large, x=xlarge

# Run inference on an image
results = model('path/to/image.jpg')

# Show results
results[0].show()  # Display image with bounding boxes
```

### 9.3 Accessing Detection Results

```python
# Get detection details
for result in results:
    boxes = result.boxes

    for box in boxes:
        # Bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        # Confidence score
        confidence = box.conf[0].item()

        # Class name
        class_id = int(box.cls[0].item())
        class_name = model.names[class_id]

        print(f"{class_name}: {confidence:.2f} at ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
```

### 9.4 YOLO on Video

```python
# Process a video
results = model('path/to/video.mp4', show=True)

# Use webcam (real-time)
results = model(source=0, show=True)  # 0 = default webcam
```

### 9.5 YOLO Model Sizes

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| YOLOv8n | 6.3 MB | Fastest | Lower | Mobile, edge, real-time |
| YOLOv8s | 22.4 MB | Fast | Good | General purpose |
| YOLOv8m | 52.0 MB | Medium | Better | Balanced |
| YOLOv8l | 87.7 MB | Slower | High | When accuracy matters |
| YOLOv8x | 136.7 MB | Slowest | Highest | Maximum accuracy |

---

## 10. Data Augmentation

Data augmentation creates **variations of existing images** to effectively increase your dataset size and reduce overfitting.

### 10.1 Keras Preprocessing Layers

Applied as part of the model. Augmentation only happens during training.

```python
data_augmentation = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.1),        # Rotate up to 10%
    layers.RandomZoom(0.1),            # Zoom up to 10%
    layers.RandomContrast(0.1),        # Adjust contrast up to 10%
])

# Add to model
model = keras.Sequential([
    data_augmentation,                 # Augmentation as first layers
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])
```

### 10.2 ImageDataGenerator

Generates augmented batches on-the-fly during training.

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,              # Normalize to [0, 1]
    rotation_range=20,           # Random rotation up to 20 degrees
    width_shift_range=0.2,       # Random horizontal shift
    height_shift_range=0.2,      # Random vertical shift
    horizontal_flip=True,        # Random horizontal flip
    zoom_range=0.2,              # Random zoom
    fill_mode='nearest'          # Fill new pixels
)

# For validation/test: only rescale, no augmentation
val_datagen = ImageDataGenerator(rescale=1./255)

# Load from directory
train_generator = train_datagen.flow_from_directory(
    'data/train/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    'data/val/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Train with generators
model.fit(train_generator, validation_data=val_generator, epochs=50)
```

### 10.3 Visualizing Augmentations

```python
# Show augmented versions of a single image
img = load_img('path/to/image.jpg', target_size=(224, 224))
img_array = img_to_array(img).reshape(1, 224, 224, 3)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax in axes.flatten():
    augmented = data_augmentation(img_array)
    ax.imshow(augmented[0].numpy().astype('uint8'))
    ax.axis('off')
plt.suptitle('Augmented Versions', fontsize=14)
plt.tight_layout()
plt.show()
```

---

## 11. Quick Reference Tables

### 11.1 CNN Layer Summary

| Layer | Input Shape | Output Shape | Purpose |
|-------|-----------|-------------|---------|
| Conv2D(32, 3x3) | (H, W, C) | (H-2, W-2, 32) | Extract features |
| MaxPooling2D(2x2) | (H, W, C) | (H/2, W/2, C) | Reduce size |
| Flatten | (H, W, C) | (H*W*C,) | Reshape for Dense |
| GlobalAveragePooling2D | (H, W, C) | (C,) | Better than Flatten for transfer |
| Dense(64) | (N,) | (64,) | Classification |

### 11.2 Pre-trained Model Quick Setup

| Step | Code |
|------|------|
| Load base | `keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))` |
| Freeze | `base_model.trainable = False` |
| Add head | `GlobalAveragePooling2D() → Dense(256) → Dropout(0.5) → Dense(n_classes, softmax)` |
| Compile | `optimizer='adam', loss='sparse_categorical_crossentropy'` |
| Train head | `model.fit(X, y, epochs=10)` |
| Unfreeze | `base_model.trainable = True` (last N layers) |
| Fine-tune | `optimizer=Adam(1e-5)`, train again |

### 11.3 Common Image Sizes

| Model | Expected Size | Preprocessing |
|-------|--------------|---------------|
| ResNet50 | 224 x 224 | `resnet50.preprocess_input` |
| VGG16 | 224 x 224 | `vgg16.preprocess_input` |
| MobileNetV2 | 224 x 224 | `mobilenet_v2.preprocess_input` |
| EfficientNetB0 | 224 x 224 | `efficientnet.preprocess_input` |
| YOLOv8 | 640 x 640 | Handled automatically |

---

## 12. Resources

- [Keras Applications (Pre-trained Models)](https://keras.io/api/applications/)
- [TensorFlow Image Classification Tutorial](https://www.tensorflow.org/tutorials/images/classification)
- [TensorFlow Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [OpenCV Python Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)

---

**Start with a pre-trained model and transfer learning -- you'll get better results faster than training from scratch!**
