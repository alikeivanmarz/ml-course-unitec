# Data Preprocessing Guide

Data preprocessing transforms raw data into a format suitable for machine learning algorithms. Poor preprocessing is the most common cause of poor model performance. This guide covers every preprocessing step used in this course, from cleaning data to building reusable pipelines.

**Table of Contents**

1. [Data Cleaning](#1-data-cleaning)
2. [Feature Scaling](#2-feature-scaling)
3. [Encoding Categorical Variables](#3-encoding-categorical-variables)
4. [Feature Selection](#4-feature-selection)
5. [Handling Imbalanced Data](#5-handling-imbalanced-data)
6. [Building Pipelines with sklearn](#6-building-pipelines-with-sklearn)
7. [Complete Preprocessing Workflow](#7-complete-preprocessing-workflow)
8. [Quick Reference Tables](#8-quick-reference-tables)
9. [Resources](#9-resources)

---

## 1. Data Cleaning

### 1.1 Handling Missing Values

Missing values can cause errors or bias in your models. Always check for them first.

```python
import pandas as pd
import numpy as np

df = pd.read_csv('../Datasets/1000_Companies.csv')

# Detect missing values
print(df.isnull().sum())              # Count per column
print(df.isnull().sum().sum())        # Total missing values
print(f"Missing %: {df.isnull().mean() * 100}")  # Percentage per column
```

**Strategies for handling missing values:**

| Strategy | When to Use | Code |
|----------|-------------|------|
| Drop rows | Few missing values (<5%) | `df.dropna()` |
| Drop columns | Column mostly missing (>50%) | `df.drop('col', axis=1)` |
| Fill with mean | Numeric, no outliers | `df['col'].fillna(df['col'].mean())` |
| Fill with median | Numeric, has outliers | `df['col'].fillna(df['col'].median())` |
| Fill with mode | Categorical data | `df['col'].fillna(df['col'].mode()[0])` |
| Forward fill | Time series data | `df['col'].fillna(method='ffill')` |

```python
# Drop rows with missing values
df_clean = df.dropna()

# Drop only rows missing specific columns
df_clean = df.dropna(subset=['R&D Spend', 'Profit'])

# Fill missing numeric values with median
df['R&D Spend'] = df['R&D Spend'].fillna(df['R&D Spend'].median())

# Using sklearn SimpleImputer (recommended for ML pipelines)
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')  # or 'mean', 'most_frequent', 'constant'
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
```

### 1.2 Removing Duplicates

```python
# Check for duplicates
print(f"Duplicate rows: {df.duplicated().sum()}")

# Remove duplicates
df = df.drop_duplicates()

# Remove duplicates based on specific columns
df = df.drop_duplicates(subset=['column1', 'column2'])
```

### 1.3 Detecting and Handling Outliers

**IQR Method** (most common in this course):

```python
def detect_outliers_iqr(df, column):
    """Detect outliers using the Interquartile Range method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    print(f"{column}: {len(outliers)} outliers detected")
    print(f"  Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
    return outliers

outliers = detect_outliers_iqr(df, 'Profit')
```

**Z-Score Method:**

```python
from scipy import stats

z_scores = np.abs(stats.zscore(df['Profit']))
outliers = df[z_scores > 3]  # Values more than 3 std from mean
print(f"Outliers (Z > 3): {len(outliers)}")
```

**Visualizing outliers:**

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].boxplot(df['Profit'])
axes[0].set_title('Boxplot of Profit')
sns.histplot(df['Profit'], bins=30, ax=axes[1])
axes[1].set_title('Distribution of Profit')
plt.tight_layout()
plt.show()
```

> **Note:** Whether to remove outliers depends on your domain. In some cases, outliers are valid data points (e.g., high-revenue companies). Tree-based models (Random Forest, XGBoost) are robust to outliers.

**Removing or capping outliers:**

```python
# Remove outlier rows
df_clean = df[(df['Profit'] >= lower_bound) & (df['Profit'] <= upper_bound)]

# Cap outliers (winsorization) -- keeps all rows, clips extreme values
df['Profit'] = df['Profit'].clip(lower=lower_bound, upper=upper_bound)
```

---

## 2. Feature Scaling

Different features often have different ranges (e.g., age 0-100 vs salary 20000-200000). Many algorithms (SVM, k-NN, neural networks, PCA) are sensitive to feature magnitude and require scaling.

### 2.1 StandardScaler (Z-score Normalization)

Transforms each feature to have **mean = 0** and **standard deviation = 1**.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Check the result
print(f"Mean: {X_train_scaled.mean(axis=0).round(2)}")   # ~0
print(f"Std:  {X_train_scaled.std(axis=0).round(2)}")     # ~1
```

**When to use:** SVM, k-NN, neural networks, PCA, logistic regression, any distance-based algorithm.

### 2.2 MinMaxScaler

Scales features to a range of **[0, 1]**.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# All values now between 0 and 1
print(f"Min: {X_train_scaled.min(axis=0)}")  # 0
print(f"Max: {X_train_scaled.max(axis=0)}")  # 1
```

**When to use:** Neural networks (especially with sigmoid/tanh activations), algorithms that require bounded input.

### 2.3 RobustScaler

Uses **median** and **IQR** instead of mean and std. Less sensitive to outliers.

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**When to use:** Data with significant outliers that you don't want to remove.

### 2.4 Scaler Comparison

| Scaler | Centers On | Scales By | Sensitive to Outliers | Best For |
|--------|-----------|-----------|----------------------|----------|
| StandardScaler | Mean | Std | Yes | Most ML algorithms |
| MinMaxScaler | Min | Max - Min | Yes | Neural networks, bounded input |
| RobustScaler | Median | IQR | No | Data with outliers |

### 2.5 Critical Rule: Fit on Train, Transform on Test

> **Important:** Always call `fit_transform` on training data and `transform` on test data. Never fit the scaler on the full dataset or on test data -- this causes **data leakage**.

```python
# CORRECT approach
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Learn from train
X_test_scaled = scaler.transform(X_test)         # Apply to test

# WRONG approach -- causes data leakage!
# scaler.fit_transform(X)  # Learns from ALL data including test
# X_train, X_test = ...    # Test info has leaked into the scaler
```

---

## 3. Encoding Categorical Variables

ML models work with numbers, not text. Categorical variables (like "State" or "Fuel Type") must be converted to numeric form.

### 3.1 One-Hot Encoding

Creates a **binary column for each category**. Use for **nominal** (unordered) categories.

```python
# Using pandas (quick and easy)
dummies = pd.get_dummies(df['State'], prefix='State', drop_first=True)
df = pd.concat([df, dummies], axis=1)
df = df.drop('State', axis=1)

# Using sklearn (better for pipelines)
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded = encoder.fit_transform(df[['State']])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out())
```

> **Note:** Use `drop_first=True` to avoid the **dummy variable trap** (multicollinearity). If you have 3 states, you only need 2 binary columns.

### 3.2 Label Encoding

Assigns a **single integer** to each category. Use for **tree-based models** or when order matters.

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['State_encoded'] = le.fit_transform(df['State'])
# California=0, Florida=1, New York=2

# To reverse
original = le.inverse_transform(df['State_encoded'])
```

> **Note:** Label encoding implies an order (0 < 1 < 2). This is fine for tree-based models (Random Forest, XGBoost) but can mislead linear models. Use one-hot encoding for linear models.

### 3.3 Ordinal Encoding

For categories with a **meaningful order** (e.g., Low < Medium < High).

```python
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder(categories=[['Low', 'Medium', 'High']])
df['priority_encoded'] = encoder.fit_transform(df[['priority']])
# Low=0, Medium=1, High=2
```

### 3.4 Encoding Comparison

| Method | When to Use | Creates Multiple Columns | Preserves Order |
|--------|-------------|------------------------|-----------------|
| One-Hot | Nominal categories, linear models | Yes | No |
| Label | Tree-based models, binary categories | No | Implied |
| Ordinal | Ordered categories (Low/Med/High) | No | Yes |

---

## 4. Feature Selection

Too many features can cause overfitting and slow training. Feature selection identifies the most useful features.

### 4.1 Correlation Analysis

Remove features that are highly correlated with each other (redundant) or weakly correlated with the target.

```python
# Correlation with the target
correlations = df.corr()['Profit'].sort_values(ascending=False)
print(correlations)

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.show()

# Remove highly correlated features (threshold > 0.9)
corr_matrix = df.corr().abs()
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > 0.9)]
df = df.drop(to_drop, axis=1)
```

### 4.2 Mutual Information

Captures **non-linear** relationships (unlike correlation which only captures linear).

```python
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

# For regression
mi_scores = mutual_info_regression(X, y, random_state=42)
mi_df = pd.DataFrame({'Feature': X.columns, 'MI Score': mi_scores})
mi_df = mi_df.sort_values('MI Score', ascending=False)
print(mi_df)
```

### 4.3 sklearn Feature Selectors

```python
from sklearn.feature_selection import SelectKBest, f_regression

# Select top K features
selector = SelectKBest(score_func=f_regression, k=5)
X_selected = selector.fit_transform(X, y)

# See which features were selected
selected_mask = selector.get_support()
selected_features = X.columns[selected_mask]
print(f"Selected features: {list(selected_features)}")
```

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor

# Select features based on model importance
selector = SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42))
selector.fit(X_train, y_train)
selected_features = X.columns[selector.get_support()]
print(f"Selected features: {list(selected_features)}")
```

---

## 5. Handling Imbalanced Data

When one class is much more frequent than another (e.g., 95% non-spam, 5% spam), models tend to predict the majority class and still get high accuracy.

### 5.1 Detecting Imbalance

```python
# Check class distribution
print(y.value_counts())
print(f"\nClass proportions:\n{y.value_counts(normalize=True)}")

# Visualize
y.value_counts().plot(kind='bar')
plt.title('Class Distribution')
plt.ylabel('Count')
plt.show()
```

### 5.2 Stratified Splits

Always use stratification when splitting imbalanced data to maintain class proportions.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Verify proportions are maintained
print(f"Train: {pd.Series(y_train).value_counts(normalize=True).to_dict()}")
print(f"Test:  {pd.Series(y_test).value_counts(normalize=True).to_dict()}")
```

### 5.3 Class Weights

Most sklearn classifiers support `class_weight='balanced'`, which penalizes misclassifying the minority class more heavily.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Logistic Regression with balanced weights
model = LogisticRegression(class_weight='balanced', random_state=42)

# Random Forest with balanced weights
model = RandomForestClassifier(class_weight='balanced', random_state=42)
```

### 5.4 SMOTE (Synthetic Minority Oversampling)

Creates **synthetic samples** for the minority class by interpolating between existing samples.

```python
from imblearn.over_sampling import SMOTE

print(f"Before SMOTE: {pd.Series(y_train).value_counts().to_dict()}")

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"After SMOTE:  {pd.Series(y_train_resampled).value_counts().to_dict()}")
```

> **Note:** Apply SMOTE only to training data, never to test data. The test set should reflect the real-world distribution.

### 5.5 Imbalanced Data Strategies

| Strategy | Pros | Cons | Code |
|----------|------|------|------|
| Stratified split | Simple, preserves proportions | Doesn't fix imbalance | `stratify=y` |
| Class weights | No data modification needed | May not be enough | `class_weight='balanced'` |
| SMOTE | Creates balanced dataset | Can create noise | `SMOTE().fit_resample()` |
| Undersampling | Simple, fast | Loses data | `RandomUnderSampler()` |

---

## 6. Building Pipelines with sklearn

**Pipelines** chain preprocessing and modeling steps together. They prevent data leakage and make code cleaner.

### 6.1 Simple Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

# fit and predict work on the entire pipeline
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
r2 = pipeline.score(X_test, y_test)
print(f"R2: {r2:.3f}")
```

### 6.2 Pipeline with ColumnTransformer

Handle **numeric** and **categorical** columns differently in the same pipeline.

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

# Define column types
numeric_features = ['R&D Spend', 'Administration', 'Marketing Spend']
categorical_features = ['State']

# Create preprocessor
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first'), categorical_features)
])

# Create full pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# Use like any sklearn model
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(f"R2: {r2_score(y_test, y_pred):.3f}")
```

### 6.3 Pipeline with Cross-Validation

Pipelines work seamlessly with cross-validation, ensuring no data leakage.

```python
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='r2')
print(f"CV R2: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
```

---

## 7. Complete Preprocessing Workflow

End-to-end example using the 1000_Companies dataset.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# 1. Load data
df = pd.read_csv('../Datasets/1000_Companies.csv')
print(f"Shape: {df.shape}")
print(df.head())

# 2. Check for issues
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nDuplicates: {df.duplicated().sum()}")
print(f"\nData types:\n{df.dtypes}")

# 3. Separate features and target
X = df.drop('Profit', axis=1)
y = df['Profit']

# 4. Identify column types
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()
print(f"\nNumeric: {numeric_features}")
print(f"Categorical: {categorical_features}")

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Build preprocessing + model pipeline
preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(drop='first'), categorical_features)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# 7. Train and evaluate
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"\nResults:")
print(f"  R2:   {r2:.3f}")
print(f"  RMSE: {rmse:.3f}")
```

---

## 8. Quick Reference Tables

### 8.1 Missing Value Strategies

| Strategy | Code | When to Use |
|----------|------|-------------|
| Drop rows | `df.dropna()` | Few missing (<5%) |
| Drop column | `df.drop('col', axis=1)` | Mostly missing (>50%) |
| Fill mean | `df['col'].fillna(df['col'].mean())` | Numeric, no outliers |
| Fill median | `df['col'].fillna(df['col'].median())` | Numeric, has outliers |
| Fill mode | `df['col'].fillna(df['col'].mode()[0])` | Categorical |
| SimpleImputer | `SimpleImputer(strategy='median')` | Pipeline-compatible |

### 8.2 Preprocessing Steps Checklist

| Step | What to Do | Common Tools |
|------|-----------|--------------|
| 1. Explore | `.shape`, `.info()`, `.describe()` | pandas |
| 2. Missing values | Detect and handle | `.isnull()`, SimpleImputer |
| 3. Duplicates | Remove | `.drop_duplicates()` |
| 4. Outliers | Detect, decide to keep/remove/cap | IQR, Z-score, boxplot |
| 5. Encode categoricals | Convert text to numbers | OneHotEncoder, LabelEncoder |
| 6. Split data | Train/test split | `train_test_split` |
| 7. Scale features | Normalize numeric features | StandardScaler, MinMaxScaler |
| 8. Feature selection | Remove irrelevant features | Correlation, SelectKBest |

### 8.3 sklearn Preprocessing Classes

| Class | Import | Purpose |
|-------|--------|---------|
| `StandardScaler` | `sklearn.preprocessing` | Z-score normalization |
| `MinMaxScaler` | `sklearn.preprocessing` | Scale to [0, 1] |
| `RobustScaler` | `sklearn.preprocessing` | Outlier-robust scaling |
| `OneHotEncoder` | `sklearn.preprocessing` | Categorical to binary columns |
| `LabelEncoder` | `sklearn.preprocessing` | Categorical to integers |
| `OrdinalEncoder` | `sklearn.preprocessing` | Ordered categorical to integers |
| `SimpleImputer` | `sklearn.impute` | Fill missing values |
| `Pipeline` | `sklearn.pipeline` | Chain preprocessing + model |
| `ColumnTransformer` | `sklearn.compose` | Different transforms per column |
| `SelectKBest` | `sklearn.feature_selection` | Select top K features |

---

## 9. Resources

- [Scikit-learn Preprocessing Guide](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Pandas Missing Data Documentation](https://pandas.pydata.org/docs/user_guide/missing_data.html)
- [Imbalanced-learn Documentation](https://imbalanced-learn.org/stable/)
- [Scikit-learn Pipeline Guide](https://scikit-learn.org/stable/modules/compose.html)

---

**Good preprocessing is often the difference between a mediocre model and a great one. When in doubt, follow the checklist in Section 8.2!**
