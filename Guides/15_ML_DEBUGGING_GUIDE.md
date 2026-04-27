# Machine Learning Debugging Guide

Debugging ML models is different from debugging regular software -- the code may run without errors yet produce poor results. This guide provides a systematic approach to diagnosing and fixing ML problems, from overfitting to memory issues.

**Table of Contents**

1. [Overfitting vs Underfitting](#1-overfitting-vs-underfitting)
2. [Data Leakage](#2-data-leakage)
3. [Debugging Poor Model Performance](#3-debugging-poor-model-performance)
4. [Common Shape Errors and Fixes](#4-common-shape-errors-and-fixes)
5. [NaN and Inf in Neural Networks](#5-nan-and-inf-in-neural-networks)
6. [Memory Issues with Large Models](#6-memory-issues-with-large-models)
7. [Learning Rate Problems](#7-learning-rate-problems)
8. [Debugging Data Pipelines](#8-debugging-data-pipelines)
9. [Reproducibility Issues](#9-reproducibility-issues)
10. [Troubleshooting Flowchart](#10-troubleshooting-flowchart)
11. [Quick Reference Tables](#11-quick-reference-tables)
12. [Resources](#12-resources)

---

## 1. Overfitting vs Underfitting

### 1.1 What Are They?

| | Underfitting | Good Fit | Overfitting |
|---|-------------|----------|-------------|
| **Train score** | Low | High | Very high |
| **Test score** | Low | High (close to train) | Low (far from train) |
| **Problem** | Model too simple | -- | Model too complex |
| **Also called** | High bias | -- | High variance |

### 1.2 How to Diagnose

```python
# Train and evaluate
model.fit(X_train, y_train)
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"Train R2: {train_score:.3f}")
print(f"Test R2:  {test_score:.3f}")
print(f"Gap:      {train_score - test_score:.3f}")

if train_score < 0.5:
    print(">> Likely UNDERFITTING: train score itself is low")
elif train_score - test_score > 0.1:
    print(">> Likely OVERFITTING: train score much higher than test score")
else:
    print(">> Model appears well-balanced")
```

### 1.3 Fixes for Overfitting

| Fix | How | Code |
|-----|-----|------|
| More training data | Collect or augment | -- |
| Regularization (L2) | Penalize large weights | `Ridge(alpha=1.0)` |
| Regularization (L1) | Penalize and zero out weights | `Lasso(alpha=1.0)` |
| Reduce features | Feature selection | `SelectKBest(k=5)` |
| Dropout (NN) | Randomly disable neurons | `layers.Dropout(0.3)` |
| Early stopping (NN) | Stop training before overfitting | `EarlyStopping(patience=10)` |
| Reduce polynomial degree | Lower model complexity | `PolynomialFeatures(degree=2)` |
| Simplify tree models | Limit depth | `max_depth=10` |

### 1.4 Fixes for Underfitting

| Fix | How | Code |
|-----|-----|------|
| Add more features | Feature engineering | `PolynomialFeatures(degree=3)` |
| Use a more complex model | Try ensemble or NN | `RandomForestRegressor()` |
| Reduce regularization | Lower alpha/penalty | `Ridge(alpha=0.01)` |
| Train longer (NN) | More epochs | `epochs=200` |
| Increase model capacity (NN) | More layers/neurons | `Dense(128)` |

---

## 2. Data Leakage

### 2.1 What is Data Leakage?

**Data leakage** occurs when information from outside the training set influences the model during training. This makes the model appear to perform well during evaluation but fail on truly unseen data.

### 2.2 Common Causes

| Cause | Example | How to Detect |
|-------|---------|--------------|
| Scaling before splitting | `fit_transform(X)` then split | Check if scaler sees test data |
| Target leakage | Feature derived from the target | Suspiciously high correlation |
| Temporal leakage | Using future data to predict past | Check time ordering |
| Preprocessing on full data | Imputing missing values before split | Review preprocessing order |

### 2.3 Wrong vs Correct Approach

```python
# WRONG -- data leakage!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)      # Scaler sees ALL data including test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
```

```python
# CORRECT -- no leakage
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit only on train
X_test_scaled = scaler.transform(X_test)         # Transform only
```

### 2.4 How to Prevent

1. **Always split first**, then preprocess
2. Use **sklearn Pipelines** (they handle this automatically)
3. Never use test data for any fitting, imputing, or encoding decisions
4. Be suspicious of **R2 > 0.99** or **accuracy > 99%** -- it may indicate leakage

```python
# Pipeline prevents leakage automatically
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

# Cross-validation with pipeline = zero leakage
cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
```

---

## 3. Debugging Poor Model Performance

### 3.1 The Debugging Checklist

Work through these steps in order when your model performs poorly:

**Step 1: Check the data**
```python
print(f"Shape: {X.shape}")
print(f"Missing: {X.isnull().sum().sum()}")
print(f"Dtypes:\n{X.dtypes}")
print(f"Target distribution:\n{y.describe()}")
```

**Step 2: Establish a baseline**
```python
from sklearn.dummy import DummyRegressor, DummyClassifier

# Regression baseline
dummy = DummyRegressor(strategy='mean')
dummy.fit(X_train, y_train)
baseline = dummy.score(X_test, y_test)
print(f"Baseline R2 (predict mean): {baseline:.3f}")
print(f"Your model R2:              {model.score(X_test, y_test):.3f}")

# If your model can't beat the baseline, the problem is fundamental
```

**Step 3: Check the features**
```python
# Correlation with target
if hasattr(X_train, 'columns'):
    correlations = pd.DataFrame(X_train).corrwith(pd.Series(y_train))
    print(correlations.sort_values(ascending=False))
```

**Step 4: Try a different model**
```python
from sklearn.ensemble import RandomForestRegressor

# If linear model fails, try a non-linear one
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print(f"Random Forest R2: {rf.score(X_test, y_test):.3f}")
```

**Step 5: Check for data leakage** (see Section 2)

**Step 6: Tune hyperparameters** (see Model Evaluation Guide)

### 3.2 Feature Importance

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Feature importance
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

plt.figure(figsize=(10, 6))
importances.plot(kind='bar')
plt.title('Feature Importance')
plt.ylabel('Importance')
plt.tight_layout()
plt.show()
```

---

## 4. Common Shape Errors and Fixes

### 4.1 "Expected 2D array, got 1D array"

```python
# ERROR
X = np.array([1, 2, 3, 4, 5])  # shape: (5,)
model.fit(X, y)  # ValueError: Expected 2D array, got 1D array

# FIX: reshape to 2D
X = X.reshape(-1, 1)  # shape: (5, 1)
model.fit(X, y)       # Works!
```

### 4.2 "Inconsistent numbers of samples"

```python
# ERROR
X_train = np.array([[1, 2], [3, 4], [5, 6]])  # 3 samples
y_train = np.array([1, 2])                      # 2 samples -- mismatch!

# FIX: ensure X and y have same number of rows
print(f"X shape: {X_train.shape}, y shape: {y_train.shape}")
# Fix the data source to have matching lengths
```

### 4.3 Keras input_shape Mismatch

```python
# ERROR
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(10,))  # expects 10 features
])
model.fit(X_train, y_train)  # X_train has 8 features -- error!

# FIX: use actual feature count
model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],))
])
```

### 4.4 Shape Debugging Template

Print this before any training call:

```python
print("=== Shape Debugging ===")
print(f"X shape:       {X.shape}")
print(f"y shape:       {y.shape}")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape:  {y_test.shape}")
print(f"X_train dtype: {X_train.dtype}")
print(f"y_train dtype: {y_train.dtype}")
print(f"NaN in X_train: {np.isnan(X_train).sum()}")
print(f"NaN in y_train: {np.isnan(y_train).sum()}")
```

---

## 5. NaN and Inf in Neural Networks

### 5.1 Common Causes

| Cause | Symptom | Fix |
|-------|---------|-----|
| Learning rate too high | Loss jumps to NaN immediately | Reduce LR (e.g., 0.001 to 0.0001) |
| Missing values in input | NaN appears after first batch | Clean data: `np.isnan(X).sum()` |
| Exploding gradients | Loss grows rapidly, then NaN | Gradient clipping or BatchNorm |
| Division by zero in custom loss | NaN in custom loss output | Add small epsilon (1e-8) |
| Numerical instability | Gradual NaN accumulation | Use mixed precision or BatchNorm |

### 5.2 How to Detect

```python
# Check input data
print(f"NaN in X_train: {np.isnan(X_train).sum()}")
print(f"Inf in X_train: {np.isinf(X_train).sum()}")
print(f"X_train range: [{X_train.min():.3f}, {X_train.max():.3f}]")

# Check predictions
y_pred = model.predict(X_test)
print(f"NaN in predictions: {np.isnan(y_pred).sum()}")
print(f"Inf in predictions: {np.isinf(y_pred).sum()}")
```

### 5.3 Fixes

```python
# Lower learning rate
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='mse')

# Gradient clipping
model.compile(optimizer=keras.optimizers.Adam(clipnorm=1.0), loss='mse')

# Add BatchNormalization
model = keras.Sequential([
    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),       # Stabilizes training
    layers.Dense(32, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(1)
])
```

---

## 6. Memory Issues with Large Models

### 6.1 Symptoms

- `ResourceExhaustedError: OOM when allocating tensor`
- `CUDA out of memory`
- Kernel crashes without error message
- System becomes unresponsive

### 6.2 Fixes

```python
# Reduce batch size (most common fix)
model.fit(X_train, y_train, batch_size=16, epochs=50)  # Try 16 instead of 32

# Clear GPU memory
import torch
torch.cuda.empty_cache()  # PyTorch

# Use mixed precision (faster + less memory)
import tensorflow as tf
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Predict in smaller batches
predictions = model.predict(X_test, batch_size=32)

# Use a smaller model
# Reduce number of layers, neurons, or filters
```

### 6.3 Monitoring Memory

```python
# Check GPU memory (PyTorch)
import torch
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.0f} MB")
print(f"Reserved:  {torch.cuda.memory_reserved() / 1024**2:.0f} MB")

# Check GPU availability (TensorFlow)
import tensorflow as tf
print(f"GPUs: {tf.config.list_physical_devices('GPU')}")
```

---

## 7. Learning Rate Problems

### 7.1 Learning Rate Too High

**Symptoms:** loss oscillates wildly, jumps up, or goes to NaN.

```python
# Fix: reduce by a factor of 10
optimizer = keras.optimizers.Adam(learning_rate=0.0001)  # was 0.001
```

### 7.2 Learning Rate Too Low

**Symptoms:** loss decreases very slowly, plateaus early, training takes very long.

```python
# Fix: increase learning rate
optimizer = keras.optimizers.Adam(learning_rate=0.01)  # was 0.0001
```

### 7.3 Learning Rate Scheduler

Automatically reduce learning rate when training plateaus.

```python
# Reduce LR when validation loss stops improving
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,          # Multiply LR by 0.5
    patience=5,          # Wait 5 epochs before reducing
    min_lr=1e-7          # Don't go below this
)

history = model.fit(X_train, y_train, callbacks=[reduce_lr], epochs=100,
                    validation_split=0.2)
```

### 7.4 Typical Learning Rates

| Optimizer | Starting LR | Fine-tuning LR |
|-----------|-------------|----------------|
| Adam | 0.001 | 0.00001 |
| SGD | 0.01 | 0.0001 |
| SGD + momentum | 0.01 | 0.0001 |

---

## 8. Debugging Data Pipelines

### 8.1 Check at Every Stage

Print the shape and a sample after each transformation step.

```python
# After loading
print(f"1. Loaded:     {df.shape}")

# After cleaning
df = df.dropna()
print(f"2. After clean: {df.shape}")

# After encoding
df_encoded = pd.get_dummies(df, drop_first=True)
print(f"3. After encode: {df_encoded.shape}")

# After split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(f"4. X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"   X_test:  {X_test.shape},  y_test:  {y_test.shape}")

# After scaling
X_train_scaled = scaler.fit_transform(X_train)
print(f"5. Scaled X_train: {X_train_scaled.shape}, dtype: {X_train_scaled.dtype}")
```

### 8.2 Common Pipeline Issues

| Issue | Symptom | Fix |
|-------|---------|-----|
| Forgot to encode categoricals | `ValueError: could not convert string to float` | Apply OneHotEncoder or pd.get_dummies |
| Column names lost after transform | Cannot reference columns by name | Use ColumnTransformer with named steps |
| Target included in features | Suspiciously high accuracy | Check `X.columns` does not contain target |
| Wrong column selected | Model performs poorly or crashes | Print `X.columns` and verify |
| Forgot to drop original categorical | Duplicate information | Drop original after encoding |

---

## 9. Reproducibility Issues

### 9.1 Setting All Random Seeds

Different libraries have different random number generators. Set all of them.

```python
import random
import numpy as np

SEED = 42

# Python built-in
random.seed(SEED)

# NumPy
np.random.seed(SEED)

# TensorFlow
import tensorflow as tf
tf.random.set_seed(SEED)

# PyTorch (if using)
# import torch
# torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)
```

### 9.2 Other Reproducibility Tips

- Always use `random_state=42` (or your student ID) in sklearn functions
- Run cells in order from top to bottom
- Restart kernel and run all cells to verify results

---

## 10. Troubleshooting Flowchart

```
Model runs but performs poorly?
├── Is TRAIN accuracy/R2 also low?
│   └── UNDERFITTING
│       ├── Try more features (PolynomialFeatures)
│       ├── Use a more complex model (Random Forest, NN)
│       ├── Reduce regularization (lower alpha)
│       └── Train longer (more epochs)
│
├── Is TRAIN accuracy high but TEST accuracy low?
│   └── OVERFITTING
│       ├── Get more training data
│       ├── Add regularization (Ridge, Dropout)
│       ├── Reduce model complexity
│       └── Use early stopping
│
├── Is accuracy suspiciously high (>99%)?
│   └── CHECK FOR DATA LEAKAGE
│       ├── Did you scale before splitting?
│       ├── Is the target column in the features?
│       └── Are features derived from the target?
│
└── Model cannot beat the baseline?
    └── FUNDAMENTAL PROBLEM
        ├── Check if features are relevant to the target
        ├── Check for data quality issues
        └── The task may not be learnable from this data

Model crashes?
├── Shape error?
│   ├── "Expected 2D, got 1D" → .reshape(-1, 1)
│   ├── "Inconsistent samples" → check X.shape vs y.shape
│   └── Keras input_shape → use X_train.shape[1:]
│
├── NaN in loss?
│   ├── Check for NaN in input data
│   ├── Lower learning rate
│   ├── Add gradient clipping
│   └── Add BatchNormalization
│
└── Out of memory?
    ├── Reduce batch size
    ├── Use a smaller model
    ├── Clear GPU cache
    └── Use mixed precision training
```

---

## 11. Quick Reference Tables

### 11.1 Overfitting vs Underfitting

| | Underfitting | Overfitting |
|---|-------------|-------------|
| Train score | Low | High |
| Test score | Low | Much lower than train |
| Bias | High | Low |
| Variance | Low | High |
| Fix | More complex model | Regularization, more data |

### 11.2 Common One-Line Fixes

| Error | One-Line Fix |
|-------|-------------|
| `Expected 2D array, got 1D` | `X = X.reshape(-1, 1)` |
| `could not convert string to float` | `pd.get_dummies(df, drop_first=True)` |
| `name 'pd' is not defined` | `import pandas as pd` |
| NaN in neural network loss | `optimizer=Adam(learning_rate=0.0001)` |
| CUDA out of memory | `batch_size=16` |
| Inconsistent sample numbers | Print `X.shape` and `y.shape` |
| Suspiciously high accuracy | Check for data leakage |
| Model worse than baseline | Check feature relevance |

### 11.3 Random Seed Functions

| Library | Function |
|---------|----------|
| Python | `random.seed(42)` |
| NumPy | `np.random.seed(42)` |
| TensorFlow | `tf.random.set_seed(42)` |
| PyTorch | `torch.manual_seed(42)` |
| sklearn | `random_state=42` parameter |

---

## 12. Resources

- [Scikit-learn Troubleshooting](https://scikit-learn.org/stable/common_pitfalls.html)
- [TensorFlow Debugging Guide](https://www.tensorflow.org/guide/keras/debugging)
- [Common ML Pitfalls](https://scikit-learn.org/stable/common_pitfalls.html)

---

**When your model isn't working, don't guess -- follow the debugging checklist in Section 3.1 and work through it step by step!**

---

[← Previous: Model Evaluation](14_MODEL_EVALUATION_GUIDE.md) | [Index](README.md) | [Next: Testing ML Code →](16_TESTING_ML_CODE_GUIDE.md)
