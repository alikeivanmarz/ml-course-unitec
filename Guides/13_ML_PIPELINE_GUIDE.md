# Complete ML Pipeline Guide

Your other guides teach each ML component individually. This guide shows how to **put them all together** into a complete, working project -- from raw CSV to final evaluation. Follow these examples step by step for assignments and projects.

**Table of Contents**

1. [The ML Pipeline Overview](#1-the-ml-pipeline-overview)
2. [Decision Guide: Choosing the Right Approach](#2-decision-guide-choosing-the-right-approach)
3. [Example 1: Simple Regression (Numeric Data)](#3-example-1-simple-regression-numeric-data)
4. [Example 2: Mixed Data Regression (Numeric + Categorical)](#4-example-2-mixed-data-regression-numeric--categorical)
5. [Example 3: Classification with Full Pipeline](#5-example-3-classification-with-full-pipeline)
6. [Pipeline Assembly Checklist](#6-pipeline-assembly-checklist)
7. [Common Integration Mistakes](#7-common-integration-mistakes)
8. [Quick Reference](#8-quick-reference)
9. [Resources](#9-resources)

---

## 1. The ML Pipeline Overview

Every ML project follows the same fundamental steps. The order matters -- doing steps out of sequence is the most common source of bugs.

```
Step 1: Load & Explore Data
    |
    v
Step 2: Clean & Preprocess
    |
    v
Step 3: Split into Train / Test
    |
    v
Step 4: Feature Engineering & Scaling (fit on train only!)
    |
    v
Step 5: Train Model(s)
    |
    v
Step 6: Evaluate & Compare
    |
    v
Step 7: Tune Best Model
    |
    v
Step 8: Final Evaluation on Test Set
```

> **Critical Rule:** Steps 4-7 must only use training data. The test set is held out until Step 8. Violating this causes **data leakage** (see ML Debugging Guide).

---

## 2. Decision Guide: Choosing the Right Approach

### 2.1 What Type of Problem?

| Your Target Variable | Problem Type | Output | Start With |
|---------------------|-------------|--------|------------|
| Continuous number (price, temperature, CO2) | **Regression** | A number | LinearRegression |
| Two categories (spam/ham, yes/no) | **Binary Classification** | 0 or 1 | LogisticRegression |
| Multiple categories (species, quality level) | **Multi-class Classification** | One of N classes | LogisticRegression or RandomForest |

### 2.2 Which Scaler?

| Your Data | Scaler | Why |
|----------|--------|-----|
| No outliers | StandardScaler | Most common, works with most algorithms |
| Has outliers you want to keep | RobustScaler | Uses median/IQR, not affected by outliers |
| Neural network input | MinMaxScaler | Networks prefer [0, 1] range |
| Tree-based model (RF, XGBoost) | No scaling needed | Trees split on thresholds, not magnitudes |

### 2.3 Which Encoding for Categorical Variables?

| Categories | Encoding | Why |
|-----------|----------|-----|
| No natural order (State, Color) | One-Hot (pd.get_dummies) | No false ordering implied |
| Natural order (Low/Med/High) | OrdinalEncoder | Preserves meaningful order |
| Many categories (>20 unique) | Target encoding or drop | One-hot creates too many columns |
| Tree-based models | LabelEncoder is fine | Trees handle encoded integers well |

### 2.4 Which Evaluation Metric?

| Problem | Default Metric | Alternative |
|---------|---------------|-------------|
| Regression | R2 + RMSE | MAE (if outliers matter less) |
| Balanced classification | Accuracy + F1 | Classification report |
| Imbalanced classification | F1 + AUC-ROC | Precision or Recall (depends on cost) |

---

## 3. Example 1: Simple Regression (Numeric Data)

**Goal:** Predict CO2 emissions from vehicle characteristics.
**Dataset:** FuelConsumptionCo2.csv
**Difficulty:** Beginner -- numeric features only, no missing values.

### Step 1: Load and Explore

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 300
sns.set_style('whitegrid')
np.random.seed(42)

# Load data
df = pd.read_csv('../Datasets/FuelConsumptionCo2.csv')

# Quick overview
print(f"Shape: {df.shape}")
print(f"\n{df.head()}")
print(f"\n{df.describe()}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nData types:\n{df.dtypes}")
```

### Step 2: Exploratory Data Analysis (EDA)

```python
# Distribution of target variable
plt.figure(figsize=(10, 5))
plt.hist(df['CO2EMISSIONS'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Distribution of CO2 Emissions')
plt.xlabel('CO2 Emissions (g/km)')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Correlation with target
numeric_df = df.select_dtypes(include=[np.number])
correlations = numeric_df.corr()['CO2EMISSIONS'].sort_values(ascending=False)
print("Correlation with CO2EMISSIONS:")
print(correlations)

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

# Scatter plots of top features vs target
top_features = ['FUELCONSUMPTION_COMB', 'FUELCONSUMPTION_CITY', 'ENGINESIZE', 'CYLINDERS']
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, feature in zip(axes.flatten(), top_features):
    ax.scatter(df[feature], df['CO2EMISSIONS'], alpha=0.3)
    ax.set_xlabel(feature)
    ax.set_ylabel('CO2EMISSIONS')
    ax.set_title(f'{feature} vs CO2')
plt.tight_layout()
plt.show()
```

### Step 3: Select Features and Split

```python
# Choose features based on correlation analysis
feature_cols = ['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']
X = df[feature_cols]
y = df['CO2EMISSIONS']

# Split into train and test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set:     {X_test.shape[0]} samples")
```

### Step 4: Scale Features

```python
# Scale features (fit on train only)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Step 5: Train and Compare Models

```python
# Define models to compare
models = {
    'Linear':     LinearRegression(),
    'Ridge':      Ridge(alpha=1.0),
    'Lasso':      Lasso(alpha=1.0),
}

# Cross-validation on training data
print("Cross-Validation Results (5-fold):")
print(f"{'Model':<15} {'CV R2':>10} {'CV RMSE':>12}")
print("-" * 40)

cv_results = {}
for name, model in models.items():
    r2_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
    mse_scores = -cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(mse_scores)

    cv_results[name] = {'r2': r2_scores.mean(), 'rmse': rmse_scores.mean()}
    print(f"{name:<15} {r2_scores.mean():>10.3f} {rmse_scores.mean():>12.3f}")
```

### Step 6: Evaluate Best Model on Test Set

```python
# Train best model on full training set
best_model = LinearRegression()
best_model.fit(X_train_scaled, y_train)

# Predict on test set
y_pred = best_model.predict(X_test_scaled)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("=== Final Test Set Results ===")
print(f"MSE:  {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"MAE:  {mae:.3f}")
print(f"R2:   {r2:.3f}")
```

### Step 7: Visualize Results

```python
# Actual vs Predicted
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Actual vs Predicted
axes[0].scatter(y_test, y_pred, alpha=0.5, color='teal')
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
             'r--', linewidth=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual CO2 Emissions')
axes[0].set_ylabel('Predicted CO2 Emissions')
axes[0].set_title(f'Actual vs Predicted (R2={r2:.3f})')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Residuals
residuals = y_test - y_pred
axes[1].scatter(y_pred, residuals, alpha=0.5, color='teal')
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=2)
axes[1].set_xlabel('Predicted Values')
axes[1].set_ylabel('Residuals')
axes[1].set_title('Residual Plot')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 4. Example 2: Mixed Data Regression (Numeric + Categorical)

**Goal:** Predict company profit from spending and state.
**Dataset:** 1000_Companies.csv
**Difficulty:** Intermediate -- has categorical features, needs encoding + scaling pipeline.

### Complete Pipeline

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)

# ===== STEP 1: LOAD AND EXPLORE =====
df = pd.read_csv('../Datasets/1000_Companies.csv')
print(f"Shape: {df.shape}")
print(f"\n{df.head()}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nCategorical columns unique values:")
for col in df.select_dtypes(include='object').columns:
    print(f"  {col}: {df[col].nunique()} unique -> {df[col].unique()[:5]}")

# ===== STEP 2: SEPARATE FEATURES AND TARGET =====
X = df.drop('Profit', axis=1)
y = df['Profit']

# Identify column types automatically
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
print(f"\nNumeric features:     {numeric_features}")
print(f"Categorical features: {categorical_features}")

# ===== STEP 3: SPLIT DATA =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain: {X_train.shape[0]} samples")
print(f"Test:  {X_test.shape[0]} samples")

# ===== STEP 4: BUILD PREPROCESSING + MODEL PIPELINE =====
# This handles encoding AND scaling in the correct order, with no data leakage
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
])

# ===== STEP 5: COMPARE MODELS WITH CROSS-VALIDATION =====
models = {
    'Linear':        Pipeline([('prep', preprocessor), ('model', LinearRegression())]),
    'Ridge':         Pipeline([('prep', preprocessor), ('model', Ridge(alpha=1.0))]),
    'Random Forest': Pipeline([('prep', preprocessor), ('model', RandomForestRegressor(n_estimators=100, random_state=42))]),
}

print("\nCross-Validation Results (5-fold):")
print(f"{'Model':<20} {'CV R2':>10} {'CV RMSE':>12}")
print("-" * 45)

best_name, best_score = None, -np.inf
for name, pipeline in models.items():
    r2_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
    mse_scores = -cross_val_score(pipeline, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(mse_scores)

    print(f"{name:<20} {r2_scores.mean():>10.3f} {rmse_scores.mean():>12.3f}")

    if r2_scores.mean() > best_score:
        best_score = r2_scores.mean()
        best_name = name

print(f"\nBest model: {best_name} (CV R2 = {best_score:.3f})")

# ===== STEP 6: TRAIN BEST MODEL AND EVALUATE ON TEST SET =====
best_pipeline = models[best_name]
best_pipeline.fit(X_train, y_train)
y_pred = best_pipeline.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n=== Final Test Results ({best_name}) ===")
print(f"R2:   {r2:.3f}")
print(f"RMSE: {rmse:.3f}")

# ===== STEP 7: VISUALIZE =====
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Profit')
plt.ylabel('Predicted Profit')
plt.title(f'{best_name}: Actual vs Predicted (R2={r2:.3f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**Key takeaway:** The `ColumnTransformer` + `Pipeline` pattern handles mixed data types cleanly and prevents data leakage automatically. Use this pattern for any dataset with both numeric and categorical columns.

---

## 5. Example 3: Classification with Full Pipeline

**Goal:** Classify wine quality as low or high.
**Dataset:** winequality-red.csv
**Difficulty:** Advanced -- binary classification, class imbalance, hyperparameter tuning.

### Complete Pipeline

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_auc_score, roc_curve)

np.random.seed(42)

# ===== STEP 1: LOAD AND EXPLORE =====
df = pd.read_csv('../Datasets/winequality-red.csv', sep=';')
print(f"Shape: {df.shape}")
print(f"\n{df.head()}")
print(f"\nTarget distribution:\n{df['quality'].value_counts().sort_index()}")

# ===== STEP 2: CREATE BINARY TARGET =====
# Convert quality scores to binary: 0 = low (<=6), 1 = high (>=7)
df['quality_label'] = (df['quality'] >= 7).astype(int)
print(f"\nBinary class distribution:\n{df['quality_label'].value_counts()}")
print(f"Class ratio: {df['quality_label'].mean():.2%} positive")

# ===== STEP 3: EDA =====
# Correlation with target
correlations = df.corr()['quality_label'].drop(['quality', 'quality_label']).sort_values(ascending=False)
print(f"\nCorrelation with quality_label:\n{correlations}")

# ===== STEP 4: PREPARE FEATURES =====
feature_cols = df.columns.drop(['quality', 'quality_label']).tolist()
X = df[feature_cols]
y = df['quality_label']

# Stratified split (maintains class proportions)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain: {X_train.shape[0]} samples (positive: {y_train.mean():.2%})")
print(f"Test:  {X_test.shape[0]} samples (positive: {y_test.mean():.2%})")

# ===== STEP 5: BUILD PIPELINES AND COMPARE =====
models = {
    'Logistic':      Pipeline([('scaler', StandardScaler()),
                               ('model', LogisticRegression(class_weight='balanced', max_iter=1000))]),
    'Random Forest': Pipeline([('scaler', StandardScaler()),
                               ('model', RandomForestClassifier(class_weight='balanced', n_estimators=100, random_state=42))]),
}

# Stratified cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nCross-Validation Results:")
print(f"{'Model':<20} {'CV F1':>10} {'CV AUC':>10}")
print("-" * 42)

for name, pipeline in models.items():
    f1_scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='f1')
    auc_scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='roc_auc')
    print(f"{name:<20} {f1_scores.mean():>10.3f} {auc_scores.mean():>10.3f}")

# ===== STEP 6: HYPERPARAMETER TUNING =====
param_grid = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [5, 10, 20, None],
    'model__min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    models['Random Forest'],
    param_grid,
    cv=skf,
    scoring='f1',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV F1:      {grid_search.best_score_:.3f}")

# ===== STEP 7: FINAL EVALUATION ON TEST SET =====
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print("\n=== Final Test Results ===")
print(classification_report(y_test, y_pred, target_names=['Low Quality', 'High Quality']))

# AUC
auc = roc_auc_score(y_test, y_prob)
print(f"AUC-ROC: {auc:.3f}")

# ===== STEP 8: VISUALIZE RESULTS =====
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion Matrix
ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=['Low Quality', 'High Quality'],
    cmap='Blues', ax=axes[0]
)
axes[0].set_title('Confusion Matrix')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[1].plot(fpr, tpr, label=f'Model (AUC = {auc:.3f})')
axes[1].plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Key takeaways from this example:**
- Use `stratify=y` in `train_test_split` for imbalanced data
- Use `class_weight='balanced'` to handle class imbalance
- Use `StratifiedKFold` for cross-validation with classification
- Use F1 and AUC-ROC instead of accuracy for imbalanced data
- Use `GridSearchCV` to find the best hyperparameters

---

## 6. Pipeline Assembly Checklist

Use this checklist before submitting any assignment or project.

### Before Training

- [ ] Loaded data and checked shape, dtypes, head
- [ ] Checked for missing values (`df.isnull().sum()`)
- [ ] Checked for duplicates (`df.duplicated().sum()`)
- [ ] Explored target distribution (histogram or value_counts)
- [ ] Checked correlations between features and target
- [ ] Identified numeric vs categorical features
- [ ] Split data **before** any preprocessing
- [ ] Used `stratify=y` for classification splits

### Feature Engineering

- [ ] Encoded categorical variables (one-hot or label)
- [ ] Scaled numeric features (fit on train, transform on test)
- [ ] OR used `ColumnTransformer` + `Pipeline` to handle both
- [ ] Did NOT fit any transformer on test data

### Model Training

- [ ] Set `random_state` for reproducibility
- [ ] Used cross-validation to compare models (not just train/test)
- [ ] Tried at least 2-3 different model types
- [ ] Used appropriate scoring metric for the problem type

### Evaluation

- [ ] Evaluated on the **held-out test set** (not training data)
- [ ] Reported appropriate metrics (R2/RMSE for regression, F1/AUC for classification)
- [ ] Created visualization (actual vs predicted, or confusion matrix)
- [ ] Checked for overfitting (train score vs test score gap)

### Code Quality

- [ ] All cells run top-to-bottom without errors
- [ ] Code is commented and organized
- [ ] Plots have titles, axis labels, and legends

---

## 7. Common Integration Mistakes

### 7.1 Wrong Order of Operations

| Mistake | Why It's Wrong | Fix |
|---------|---------------|-----|
| Scale before split | Test data leaks into scaler | Split first, then scale |
| Encode before split | Test categories leak into encoder | Split first, then encode |
| SMOTE before split | Synthetic test samples inflate metrics | Split first, then SMOTE on train only |
| Feature selection on full data | Test information guides selection | Select features on train only |

### 7.2 Forgetting Steps

| What You Forgot | Symptom | Fix |
|----------------|---------|-----|
| Encoding categoricals | `ValueError: could not convert string to float` | Use `pd.get_dummies()` or `OneHotEncoder` |
| Scaling for SVM/k-NN | Model performs terribly | Add `StandardScaler` to pipeline |
| Setting random_state | Different results each run | Add `random_state=42` everywhere |
| Stratifying split | Train/test class ratios differ | Add `stratify=y` to `train_test_split` |

### 7.3 Pipeline vs Manual Preprocessing

| Approach | Pros | Cons |
|----------|------|------|
| **Manual** (step by step) | Easier to understand, more control | Easy to make leakage mistakes |
| **Pipeline** (sklearn) | No leakage possible, cleaner code | Slightly harder to debug |

> **Recommendation:** Use Pipelines for assignments and projects. Use manual steps when learning concepts for the first time.

---

## 8. Quick Reference

### 8.1 The Standard Pipeline Pattern

```python
# For numeric-only datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', YourModel())
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

```python
# For mixed numeric + categorical datasets
from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(drop='first'), categorical_cols)
])

pipeline = Pipeline([
    ('prep', preprocessor),
    ('model', YourModel())
])
```

### 8.2 Complete Workflow Summary

| Step | Regression | Classification |
|------|-----------|---------------|
| 1. Load | `pd.read_csv()` | `pd.read_csv()` |
| 2. Explore | `.describe()`, correlation heatmap | `.value_counts()`, class distribution |
| 3. Split | `train_test_split(test_size=0.2)` | `train_test_split(stratify=y)` |
| 4. Preprocess | StandardScaler + Pipeline | StandardScaler + Pipeline |
| 5. Compare models | `cross_val_score(scoring='r2')` | `cross_val_score(scoring='f1')` |
| 6. Tune | GridSearchCV | GridSearchCV |
| 7. Evaluate | R2, RMSE, MAE | F1, AUC, confusion matrix |
| 8. Visualize | Actual vs predicted, residuals | Confusion matrix, ROC curve |

### 8.3 Imports You'll Need

```python
# Always
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Splitting and validation
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

# Preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Regression models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Regression metrics
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Classification metrics
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, roc_auc_score, roc_curve,
                             f1_score, accuracy_score)
```

---

## 9. Resources

- [Data Preprocessing Guide](12_DATA_PREPROCESSING_GUIDE.md) -- detailed preprocessing reference
- [Model Evaluation Guide](14_MODEL_EVALUATION_GUIDE.md) -- all metrics and tuning methods
- [ML Debugging Guide](15_ML_DEBUGGING_GUIDE.md) -- troubleshooting when things go wrong
- [Python Essentials Guide](07_PYTHON_ESSENTIALS_FOR_ML.md) -- Python and library basics
- [Scikit-learn Pipeline Documentation](https://scikit-learn.org/stable/modules/compose.html)

---

**Follow the checklist in Section 6, use Pipelines to prevent data leakage, and always evaluate on a held-out test set. The steps are always the same -- only the data and models change!**

---

[← Previous: Data Preprocessing](12_DATA_PREPROCESSING_GUIDE.md) | [Index](README.md) | [Next: Model Evaluation →](14_MODEL_EVALUATION_GUIDE.md)
