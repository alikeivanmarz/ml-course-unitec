# Model Evaluation and Selection Guide

Evaluation metrics tell you how well your model performs. Choosing the right metric and evaluation strategy is as important as choosing the right model. This guide covers all the metrics, cross-validation techniques, and hyperparameter tuning methods used in this course.

**Table of Contents**

1. [Regression Metrics](#1-regression-metrics)
2. [Classification Metrics](#2-classification-metrics)
3. [Cross-Validation](#3-cross-validation)
4. [Hyperparameter Tuning](#4-hyperparameter-tuning)
5. [Model Selection Workflow](#5-model-selection-workflow)
6. [Visual Evaluation](#6-visual-evaluation)
7. [When to Use Which Metric](#7-when-to-use-which-metric)
8. [Quick Reference Tables](#8-quick-reference-tables)
9. [Resources](#9-resources)

---

## 1. Regression Metrics

### 1.1 Mean Squared Error (MSE)

Averages the squared differences between predicted and actual values. **Penalizes large errors** more heavily.

```python
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse:.3f}")

# Manual calculation
mse_manual = np.mean((y_test - y_pred) ** 2)
```

- **Range:** 0 to infinity (lower is better)
- **Units:** squared units of the target (e.g., dollars squared)

### 1.2 Root Mean Squared Error (RMSE)

Square root of MSE. **Same units as the target variable**, making it easier to interpret.

```python
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.3f}")
```

- **Range:** 0 to infinity (lower is better)
- **Interpretation:** on average, predictions are off by approximately RMSE units

### 1.3 Mean Absolute Error (MAE)

Averages the absolute differences. **Less sensitive to outliers** than MSE.

```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae:.3f}")
```

- **Range:** 0 to infinity (lower is better)
- **Use when:** outliers in the target should not dominate the error

### 1.4 R-squared (R2)

The proportion of variance in the target that is explained by the model. The most commonly used regression metric.

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print(f"R2: {r2:.3f}")
```

- **Range:** -infinity to 1 (higher is better)
- **R2 = 1.0:** perfect predictions
- **R2 = 0.0:** model is as good as predicting the mean
- **R2 < 0:** model is worse than predicting the mean

### 1.5 Adjusted R-squared

R2 increases whenever you add features, even useless ones. **Adjusted R2** penalizes unnecessary features.

```python
n = len(y_test)          # number of samples
p = X_test.shape[1]      # number of features
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
print(f"Adjusted R2: {adj_r2:.3f}")
```

- **Use when:** comparing models with different numbers of features

### 1.6 Complete Regression Evaluation

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_regression(y_true, y_pred, X_test=None):
    """Print all regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"MSE:  {mse:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE:  {mae:.3f}")
    print(f"R2:   {r2:.3f}")

    if X_test is not None:
        n = len(y_true)
        p = X_test.shape[1]
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        print(f"Adj R2: {adj_r2:.3f}")

evaluate_regression(y_test, y_pred, X_test)
```

---

## 2. Classification Metrics

### 2.1 Confusion Matrix

A table showing how predictions compare to actual labels.

```
                Predicted
              Negative  Positive
Actual  Neg    TN         FP
        Pos    FN         TP
```

- **TP (True Positive):** correctly predicted positive
- **TN (True Negative):** correctly predicted negative
- **FP (False Positive):** predicted positive but actually negative (Type I error)
- **FN (False Negative):** predicted negative but actually positive (Type II error)

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visual confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
```

### 2.2 Accuracy

The proportion of correct predictions. Simple but **misleading for imbalanced data**.

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
```

- **Problem:** with 95% negative and 5% positive, always predicting "negative" gives 95% accuracy but misses all positives

### 2.3 Precision, Recall, and F1-Score

| Metric | Formula | Question It Answers |
|--------|---------|-------------------|
| Precision | TP / (TP + FP) | Of all positive predictions, how many are correct? |
| Recall | TP / (TP + FN) | Of all actual positives, how many did we find? |
| F1-Score | 2 * (Precision * Recall) / (Precision + Recall) | Balance between precision and recall |

```python
from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1-Score:  {f1:.3f}")
```

**When precision matters most:** spam detection -- you don't want real emails marked as spam (minimize FP).

**When recall matters most:** disease detection -- you don't want to miss sick patients (minimize FN).

### 2.4 Classification Report

All metrics in one function.

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, target_names=['Low Quality', 'High Quality']))
```

Output:
```
               precision    recall  f1-score   support
  Low Quality       0.95      0.98      0.96       250
 High Quality       0.82      0.68      0.74        50
     accuracy                           0.93       300
    macro avg       0.89      0.83      0.85       300
 weighted avg       0.93      0.93      0.93       300
```

### 2.5 AUC-ROC Curve

The **ROC curve** plots True Positive Rate vs False Positive Rate at different thresholds. **AUC** (Area Under Curve) summarizes it in a single number.

```python
from sklearn.metrics import roc_curve, roc_auc_score

# Need probability predictions, not class labels
y_prob = model.predict_proba(X_test)[:, 1]

# Calculate AUC
auc = roc_auc_score(y_test, y_prob)
print(f"AUC: {auc:.3f}")

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Model (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

- **AUC = 1.0:** perfect classifier
- **AUC = 0.5:** no better than random
- **AUC < 0.5:** worse than random (model is inverted)

### 2.6 Multi-class Metrics

For problems with more than 2 classes, metrics need an **averaging strategy**.

| Average | What It Does | When to Use |
|---------|-------------|-------------|
| `macro` | Average across classes equally | All classes equally important |
| `weighted` | Average weighted by class frequency | Account for class imbalance |
| `micro` | Calculate globally (total TP, FP, FN) | Overall performance |

```python
# Multi-class F1
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')
print(f"F1 (macro):    {f1_macro:.3f}")
print(f"F1 (weighted): {f1_weighted:.3f}")
```

---

## 3. Cross-Validation

A single train/test split can be misleading. **Cross-validation** tests on multiple splits for a more reliable estimate.

### 3.1 K-Fold Cross-Validation

Splits data into K folds, trains on K-1 folds, tests on the remaining fold. Repeats K times.

```python
from sklearn.model_selection import cross_val_score

# 5-fold cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"CV scores: {cv_scores}")
print(f"CV R2: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
```

### 3.2 Stratified K-Fold

For **classification**, maintains the class distribution in each fold.

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
print(f"Stratified CV F1: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
```

### 3.3 Common Scoring Parameters

| Task | Scoring String | Higher = Better |
|------|---------------|-----------------|
| Regression | `'r2'` | Yes |
| Regression | `'neg_mean_squared_error'` | Yes (less negative) |
| Classification | `'accuracy'` | Yes |
| Classification | `'f1'` | Yes |
| Classification | `'precision'` | Yes |
| Classification | `'recall'` | Yes |
| Classification | `'roc_auc'` | Yes |

> **Note:** MSE is returned as negative (`neg_mean_squared_error`) because sklearn maximizes all scores. Take the negative to get the actual MSE.

---

## 4. Hyperparameter Tuning

### 4.1 GridSearchCV

Tries **every combination** of hyperparameters. Best for small parameter spaces.

```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

param_grid = {
    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
}

grid_search = GridSearchCV(
    Ridge(), param_grid, cv=5, scoring='r2', return_train_score=True
)
grid_search.fit(X_train, y_train)

print(f"Best alpha: {grid_search.best_params_['alpha']}")
print(f"Best CV R2: {grid_search.best_score_:.3f}")

# Use the best model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print(f"Test R2:    {r2_score(y_test, y_pred):.3f}")
```

### 4.2 RandomizedSearchCV

Samples random combinations. Better for **large parameter spaces**.

```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint, uniform

param_distributions = {
    'n_estimators': randint(50, 500),
    'max_depth': [None, 5, 10, 20, 30],
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10)
}

random_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_distributions,
    n_iter=50,           # Number of random combinations to try
    cv=5,
    scoring='r2',
    random_state=42
)
random_search.fit(X_train, y_train)

print(f"Best params: {random_search.best_params_}")
print(f"Best CV R2:  {random_search.best_score_:.3f}")
```

### 4.3 Common Hyperparameters by Model

| Model | Key Hyperparameters | Typical Search Range |
|-------|-------------------|---------------------|
| Ridge/Lasso | `alpha` | 0.001 to 100 |
| Random Forest | `n_estimators`, `max_depth`, `min_samples_split` | 50-500, 5-30, 2-20 |
| SVM | `C`, `kernel`, `gamma` | 0.01-100, rbf/linear, scale/auto |
| k-NN | `n_neighbors`, `weights` | 1-30, uniform/distance |
| XGBoost | `learning_rate`, `n_estimators`, `max_depth` | 0.01-0.3, 50-500, 3-10 |

---

## 5. Model Selection Workflow

Follow these steps to systematically select the best model.

```python
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# Step 1: Establish a baseline
baseline = DummyRegressor(strategy='mean')
baseline_scores = cross_val_score(baseline, X, y, cv=5, scoring='r2')
print(f"Baseline (mean):    R2 = {baseline_scores.mean():.3f}")

# Step 2: Compare multiple models
models = {
    'Linear':        LinearRegression(),
    'Ridge':         Ridge(alpha=1.0),
    'Lasso':         Lasso(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"{name:20s} R2 = {scores.mean():.3f} +/- {scores.std():.3f}")

# Step 3: Tune the best model's hyperparameters (see Section 4)
# Step 4: Final evaluation on held-out test set
```

---

## 6. Visual Evaluation

### 6.1 Actual vs Predicted Plot

```python
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

### 6.2 Residual Plot

Residuals (errors) should be randomly scattered around zero with no pattern.

```python
residuals = y_test - y_pred

plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.6, color='teal')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

- **Random scatter:** good model
- **Fan shape:** heteroscedasticity (variance changes with prediction)
- **Curved pattern:** model is missing a non-linear relationship

### 6.3 Learning Curves

Shows how performance changes with more training data. Diagnoses overfitting vs underfitting.

```python
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    model, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10), scoring='r2'
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, label='Training Score')
plt.plot(train_sizes, val_mean, label='Validation Score')
plt.xlabel('Training Set Size')
plt.ylabel('R2 Score')
plt.title('Learning Curves')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

- **Both scores low:** underfitting -- need a more complex model
- **Train high, validation low:** overfitting -- need more data or regularization
- **Both scores converge and are high:** good fit

---

## 7. When to Use Which Metric

| Scenario | Recommended Metric | Why |
|----------|-------------------|-----|
| Regression (general) | R2, RMSE | R2 for comparison, RMSE for interpretability |
| Regression with outliers | MAE | Less sensitive to extreme errors |
| Comparing models with different features | Adjusted R2 | Penalizes unnecessary features |
| Binary classification (balanced) | Accuracy, F1 | Simple and reliable |
| Binary classification (imbalanced) | F1, AUC-ROC | Accuracy is misleading |
| Spam detection | Precision | Minimize false positives (real mail marked spam) |
| Disease detection | Recall | Minimize false negatives (missed diagnoses) |
| Multi-class classification | Macro F1 | Treats all classes equally |
| Ranking problems | AUC-ROC | Threshold-independent |

---

## 8. Quick Reference Tables

### 8.1 Regression Metrics

| Metric | Function | Range | Lower/Higher Better |
|--------|----------|-------|-------------------|
| MSE | `mean_squared_error(y, pred)` | [0, inf) | Lower |
| RMSE | `np.sqrt(mean_squared_error(y, pred))` | [0, inf) | Lower |
| MAE | `mean_absolute_error(y, pred)` | [0, inf) | Lower |
| R2 | `r2_score(y, pred)` | (-inf, 1] | Higher |

### 8.2 Classification Metrics

| Metric | Function | Range | Higher Better |
|--------|----------|-------|-------------|
| Accuracy | `accuracy_score(y, pred)` | [0, 1] | Yes |
| Precision | `precision_score(y, pred)` | [0, 1] | Yes |
| Recall | `recall_score(y, pred)` | [0, 1] | Yes |
| F1 | `f1_score(y, pred)` | [0, 1] | Yes |
| AUC | `roc_auc_score(y, prob)` | [0, 1] | Yes |

### 8.3 Cross-Validation Methods

| Method | Code | When to Use |
|--------|------|-------------|
| K-Fold | `cross_val_score(model, X, y, cv=5)` | Regression, general |
| Stratified K-Fold | `StratifiedKFold(n_splits=5)` | Classification (maintains class ratio) |
| Repeated K-Fold | `RepeatedKFold(n_splits=5, n_repeats=3)` | More stable estimates |

---

## 9. Resources

- [Scikit-learn Metrics Documentation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Scikit-learn Cross-validation Guide](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Scikit-learn Hyperparameter Tuning](https://scikit-learn.org/stable/modules/grid_search.html)

---

**The best model isn't always the one with the highest accuracy -- it's the one that uses the right metric for the right problem!**

---

[← Previous: ML Pipeline](13_ML_PIPELINE_GUIDE.md) | [Index](README.md) | [Next: ML Debugging →](15_ML_DEBUGGING_GUIDE.md)
