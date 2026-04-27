# Deep Learning with TensorFlow and Keras

TensorFlow is the deep learning framework used in this course, and Keras is its high-level API for building and training neural networks. This guide covers everything from building your first neural network to saving trained models.

**Table of Contents**

1. [Getting Started](#1-getting-started)
2. [Building Models](#2-building-models)
3. [Layer Types Reference](#3-layer-types-reference)
4. [Activation Functions](#4-activation-functions)
5. [Loss Functions and Optimizers](#5-loss-functions-and-optimizers)
6. [Training a Model](#6-training-a-model)
7. [Callbacks](#7-callbacks)
8. [Plotting Training History](#8-plotting-training-history)
9. [Saving and Loading Models](#9-saving-and-loading-models)
10. [GPU Usage](#10-gpu-usage)
11. [Common Architectures Overview](#11-common-architectures-overview)
12. [Quick Reference Tables](#12-quick-reference-tables)
13. [Resources](#13-resources)

---

## 1. Getting Started

### 1.1 Imports and Version Check

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(f"TensorFlow version: {tf.__version__}")
```

### 1.2 Check GPU Availability

```python
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs available: {len(gpus)}")
    for gpu in gpus:
        print(f"  {gpu}")
else:
    print("No GPU found. Using CPU.")
```

### 1.3 TensorFlow vs PyTorch

| Feature | TensorFlow/Keras | PyTorch |
|---------|-----------------|---------|
| API style | High-level (Sequential/Functional) | More Pythonic, explicit |
| Used in course | Weeks 2-4 (primary) | Week 3-4 (some models) |
| Model building | `keras.Sequential([...])` | `nn.Module` subclass |
| Training | `model.fit()` | Manual training loop |
| Best for | Quick prototyping, production | Research, flexibility |

> **Note:** This course primarily uses Keras. PyTorch is used for some pre-trained models (HuggingFace, YOLO).

---

## 2. Building Models

### 2.1 Sequential API

Use the Sequential API when your model is a **linear stack of layers** (most common case).

**Regression example:**

```python
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)  # No activation for regression output
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()
```

**Binary classification example:**

```python
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')  # Sigmoid for binary output
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

**Multi-class classification example:**

```python
num_classes = 10

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')  # Softmax for multi-class
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

### 2.2 Functional API

Use the Functional API when you need **multiple inputs, multiple outputs, or skip connections**.

```python
# Same model as Sequential, written with Functional API
inputs = keras.Input(shape=(X_train.shape[1],))
x = layers.Dense(64, activation='relu')(inputs)
x = layers.Dropout(0.2)(x)
x = layers.Dense(32, activation='relu')(x)
outputs = layers.Dense(1)(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
```

### 2.3 Sequential vs Functional

| Feature | Sequential | Functional |
|---------|-----------|------------|
| Simplicity | Simpler, less code | More verbose |
| Linear models | Yes | Yes |
| Multiple inputs/outputs | No | Yes |
| Skip connections | No | Yes |
| Shared layers | No | Yes |
| **Use when** | Simple architectures | Complex architectures |

---

## 3. Layer Types Reference

### 3.1 Dense (Fully Connected)

Every neuron connects to every neuron in the previous layer.

```python
layers.Dense(64, activation='relu')           # 64 neurons, ReLU activation
layers.Dense(1)                                # Output layer (regression)
layers.Dense(10, activation='softmax')         # Output layer (10 classes)
```

### 3.2 Conv2D (Convolutional)

Extracts features from images using learnable filters.

```python
layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3))
layers.Conv2D(64, (3, 3), activation='relu', padding='same')
```

- `32` = number of filters
- `(3, 3)` = filter size
- `padding='same'` = output has same spatial dimensions as input

### 3.3 MaxPooling2D

Reduces spatial dimensions by taking the maximum value in each window.

```python
layers.MaxPooling2D((2, 2))   # Halves width and height
```

### 3.4 Flatten

Converts multi-dimensional data to 1D. Used between convolutional and dense layers.

```python
layers.Flatten()  # (batch, 7, 7, 64) -> (batch, 3136)
```

### 3.5 Dropout

Randomly sets a fraction of neurons to zero during training. **Prevents overfitting.**

```python
layers.Dropout(0.3)  # Drop 30% of neurons randomly
```

> **Note:** Dropout is automatically disabled during prediction (`model.predict()`).

### 3.6 BatchNormalization

Normalizes the output of the previous layer. **Stabilizes and speeds up training.**

```python
layers.BatchNormalization()
```

Typically placed after the dense/conv layer and before the activation, or after the activation.

### 3.7 LSTM (Long Short-Term Memory)

For sequence data (text, time series).

```python
layers.LSTM(64, return_sequences=True)   # Returns output at each time step
layers.LSTM(32)                           # Returns output at last time step only
```

### 3.8 Layer Types Summary

| Layer | Purpose | Key Parameters | Typical Use |
|-------|---------|---------------|-------------|
| Dense | Fully connected | units, activation | All networks |
| Conv2D | Feature extraction | filters, kernel_size | Image processing |
| MaxPooling2D | Downsampling | pool_size | After Conv2D |
| Flatten | Reshape to 1D | -- | Before Dense (after Conv) |
| Dropout | Regularization | rate (0.1-0.5) | Between Dense/Conv layers |
| BatchNormalization | Stabilize training | -- | After Dense/Conv layers |
| LSTM | Sequence processing | units | Text, time series |
| GlobalAveragePooling2D | Spatial averaging | -- | Transfer learning |

---

## 4. Activation Functions

| Activation | Output Range | Use For | Code |
|-----------|-------------|---------|------|
| **ReLU** | [0, infinity) | Hidden layers (default) | `activation='relu'` |
| **Sigmoid** | (0, 1) | Binary classification output | `activation='sigmoid'` |
| **Softmax** | (0, 1), sums to 1 | Multi-class output | `activation='softmax'` |
| **Tanh** | (-1, 1) | Hidden layers (alternative) | `activation='tanh'` |
| **LeakyReLU** | (-infinity, infinity) | When ReLU causes dead neurons | `layers.LeakyReLU(0.1)` |
| **Linear/None** | (-infinity, infinity) | Regression output | No activation |

**Rule of thumb:**
- Hidden layers: use **ReLU**
- Binary output: use **sigmoid**
- Multi-class output: use **softmax**
- Regression output: use **no activation** (linear)

---

## 5. Loss Functions and Optimizers

### 5.1 Loss Functions by Task

| Task | Loss Function | Output Activation | Labels Format |
|------|--------------|-------------------|--------------|
| Regression | `'mse'` | None (linear) | Continuous values |
| Regression | `'mae'` | None (linear) | Continuous values |
| Binary classification | `'binary_crossentropy'` | sigmoid | 0 or 1 |
| Multi-class | `'sparse_categorical_crossentropy'` | softmax | Integer labels (0, 1, 2...) |
| Multi-class | `'categorical_crossentropy'` | softmax | One-hot encoded labels |

### 5.2 Optimizers

```python
# Adam (recommended default)
model.compile(optimizer='adam', loss='mse')

# Adam with custom learning rate
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')

# SGD with momentum
model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01, momentum=0.9), loss='mse')
```

| Optimizer | Typical LR | Best For |
|-----------|-----------|----------|
| **Adam** | 0.001 | Default choice, works well in most cases |
| **SGD** | 0.01 | When you need more control, large-scale training |
| **RMSprop** | 0.001 | Recurrent networks |

---

## 6. Training a Model

### 6.1 Complete Training Workflow

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

# 1. Load and prepare data
df = pd.read_csv('../Datasets/FuelConsumptionCo2.csv')
feature_cols = ['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']
X = df[feature_cols].values
y = df['CO2EMISSIONS'].values

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Build model
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    layers.Dropout(0.2),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 5. Train
history = model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,     # Use 20% of training data for validation
    verbose=1                  # Show progress bar
)

# 6. Evaluate
test_loss, test_mae = model.evaluate(X_test_scaled, y_test)
print(f"\nTest MAE: {test_mae:.3f}")
```

### 6.2 Key `model.fit()` Parameters

| Parameter | What It Does | Typical Value |
|-----------|-------------|---------------|
| `epochs` | Number of full passes through data | 50-200 |
| `batch_size` | Samples per gradient update | 16, 32, 64 |
| `validation_split` | Fraction of training data for validation | 0.2 |
| `validation_data` | Explicit validation set | `(X_val, y_val)` |
| `callbacks` | Functions called during training | `[EarlyStopping(...)]` |
| `verbose` | Output mode (0=silent, 1=progress, 2=one line per epoch) | 1 |

### 6.3 Understanding Training Output

```
Epoch 1/100
27/27 [==============================] - 1s 5ms/step - loss: 4521.3 - mae: 52.1 - val_loss: 3102.5 - val_mae: 43.2
```

- `loss` / `mae`: performance on **training** data
- `val_loss` / `val_mae`: performance on **validation** data
- If `val_loss` increases while `loss` decreases: **overfitting**

---

## 7. Callbacks

Callbacks are functions that execute at certain points during training.

### 7.1 EarlyStopping

Stops training when a monitored metric stops improving.

```python
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',          # What to watch
    patience=10,                 # Wait 10 epochs without improvement
    restore_best_weights=True    # Go back to the best model
)
```

### 7.2 ModelCheckpoint

Saves the model at its best performance during training.

```python
checkpoint = keras.callbacks.ModelCheckpoint(
    'best_model.keras',          # File path
    monitor='val_loss',
    save_best_only=True          # Only save when it improves
)
```

### 7.3 ReduceLROnPlateau

Reduces learning rate when training plateaus.

```python
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,                  # Multiply LR by 0.5
    patience=5,                  # Wait 5 epochs before reducing
    min_lr=1e-7                  # Don't go below this
)
```

### 7.4 Using Multiple Callbacks Together

```python
callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
    keras.callbacks.ModelCheckpoint('best_model.keras', save_best_only=True)
]

history = model.fit(
    X_train_scaled, y_train,
    epochs=200,                  # Can set high -- EarlyStopping will stop it
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks
)
```

---

## 8. Plotting Training History

The `history` object from `model.fit()` contains loss and metric values for each epoch.

```python
def plot_training_history(history):
    """Plot training and validation loss and metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot loss
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('Loss Over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot metric (adjust key name based on your metric)
    metric_name = [k for k in history.history.keys() if k not in ['loss', 'val_loss']][0]
    axes[1].plot(history.history[metric_name], label=f'Train {metric_name}')
    axes[1].plot(history.history[f'val_{metric_name}'], label=f'Val {metric_name}')
    axes[1].set_title(f'{metric_name.upper()} Over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel(metric_name)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

plot_training_history(history)
```

**What to look for:**
- **Train and val loss both decrease and converge:** good fit
- **Val loss stops decreasing while train loss continues:** overfitting -- use EarlyStopping
- **Both losses remain high:** underfitting -- increase model capacity

---

## 9. Saving and Loading Models

### 9.1 Save and Load Entire Model

```python
# Save (includes architecture, weights, optimizer state)
model.save('my_model.keras')

# Load
loaded_model = keras.models.load_model('my_model.keras')
y_pred = loaded_model.predict(X_test_scaled)
```

### 9.2 Save and Load Weights Only

```python
# Save weights
model.save_weights('model_weights.weights.h5')

# Load weights (model architecture must be defined first)
model = keras.Sequential([...])  # Rebuild same architecture
model.load_weights('model_weights.weights.h5')
```

---

## 10. GPU Usage

### 10.1 Checking GPU Availability

```python
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs available: {len(gpus)}")

# If no GPU, TensorFlow uses CPU automatically -- no code changes needed
```

### 10.2 Memory Growth Configuration

Prevents TensorFlow from allocating all GPU memory at once.

```python
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

### 10.3 Mixed Precision Training

Uses float16 for speed and float32 for accuracy. Reduces memory usage.

```python
tf.keras.mixed_precision.set_global_policy('mixed_float16')
# Then build and train your model as normal
```

---

## 11. Common Architectures Overview

| Architecture | Type | Use Case | Course Week |
|-------------|------|----------|-------------|
| **MLP** | Dense layers | Tabular data (regression, classification) | Week 2 |
| **CNN** | Conv + Pool + Dense | Image classification, object detection | Week 3 |
| **RNN / LSTM** | Recurrent layers | Text, time series | Week 3 |
| **Transformer** | Self-attention | NLP, LLMs, translation | Week 3 |
| **Autoencoder** | Encoder + Decoder | Dimensionality reduction, anomaly detection | Week 3 |
| **GAN** | Generator + Discriminator | Image generation | Week 3-4 |
| **Diffusion** | Noise → Denoise | High-quality image generation | Week 4 |

---

## 12. Quick Reference Tables

### 12.1 Model Building Checklist

| Step | Code | Note |
|------|------|------|
| Import | `from tensorflow import keras` | |
| Build | `model = keras.Sequential([...])` | Define layers |
| Compile | `model.compile(optimizer, loss, metrics)` | Set optimizer and loss |
| Summary | `model.summary()` | Check parameter count |
| Train | `history = model.fit(X, y, ...)` | Returns history object |
| Evaluate | `model.evaluate(X_test, y_test)` | Returns loss + metrics |
| Predict | `model.predict(X_new)` | Returns predictions |
| Save | `model.save('model.keras')` | Save to file |

### 12.2 compile() Parameters

| Parameter | Options | Example |
|-----------|---------|---------|
| optimizer | `'adam'`, `'sgd'`, `Adam(lr=0.001)` | `optimizer='adam'` |
| loss | `'mse'`, `'binary_crossentropy'`, `'sparse_categorical_crossentropy'` | `loss='mse'` |
| metrics | `['mae']`, `['accuracy']`, `['precision']` | `metrics=['mae']` |

### 12.3 fit() Parameters

| Parameter | Default | Common Values |
|-----------|---------|---------------|
| epochs | -- | 50-200 |
| batch_size | 32 | 16, 32, 64, 128 |
| validation_split | 0.0 | 0.2 |
| callbacks | None | [EarlyStopping, ReduceLR] |
| verbose | 1 | 0 (silent), 1 (progress), 2 (summary) |

---

## 13. Resources

- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Keras Documentation](https://keras.io/api/)
- [Keras Sequential Guide](https://keras.io/guides/sequential_model/)
- [Keras Functional API Guide](https://keras.io/guides/functional_api/)

---

**Deep learning is powerful but not magic -- start simple (fewer layers, fewer neurons) and add complexity only if needed!**

---

[← Previous: Reinforcement Learning](20_REINFORCEMENT_LEARNING_GUIDE.md) | [Index](README.md) | [Next: PyTorch Reference →](22_PYTORCH_GUIDE.md)
