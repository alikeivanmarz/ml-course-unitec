# Python Essentials for Machine Learning

This guide covers the Python fundamentals you need for this ML course. It is designed as a reference -- scan the table of contents and jump to the section you need. All code examples work in your `mlcourse` conda environment with the **ML Course (Python 3.10)** kernel in VSCode.

**Table of Contents**

1. [Core Python](#1-core-python)
2. [NumPy Essentials](#2-numpy-essentials)
3. [Pandas Essentials](#3-pandas-essentials)
4. [Visualization with Matplotlib and Seaborn](#4-visualization-with-matplotlib-and-seaborn)
5. [Common ML Code Patterns](#5-common-ml-code-patterns)
6. [Quick Reference Tables](#6-quick-reference-tables)
7. [Common Errors and Debugging Tips](#7-common-errors-and-debugging-tips)
8. [Resources](#8-resources)

---

## 1. Core Python

### 1.1 Variables and Data Types

Python does not require type declarations. You assign a value and Python figures out the type.

```python
# Variable assignment
age = 25                    # int (whole number)
accuracy = 0.95             # float (decimal number)
model_name = "LinearReg"    # str (text)
is_trained = True           # bool (True or False)
result = None               # NoneType (no value)

# Check type
print(type(accuracy))       # <class 'float'>

# Type conversion
epochs = int("100")            # str to int
learning_rate = float("0.01")  # str to float
label = str(42)                # int to str
```

### 1.2 Operators

```python
# Arithmetic
total = 10 + 3              # 13   addition
diff = 10 - 3               # 7    subtraction
product = 10 * 3            # 30   multiplication
ratio = 10 / 3              # 3.33 division (always returns float)
floor = 10 // 3             # 3    floor division (drops decimals)
remainder = 10 % 3          # 1    modulo (remainder)
power = 2 ** 10             # 1024 exponentiation

# Common in ML formulas
mse = total_error / n_samples
rmse = mse ** 0.5
batch_number = 100 // 32        # = 3 full batches

# Comparison (return True or False)
r2_score > 0.8
accuracy == 1.0
accuracy != 0.5

# Logical
(accuracy > 0.9) and (loss < 0.1)
(model == "Ridge") or (model == "Lasso")
not is_trained

# Assignment shortcuts
count += 1                  # same as count = count + 1
total -= 5                  # same as total = total - 5
```

### 1.3 Strings and f-strings

**f-strings** are the most common way to format output in this course. Put `f` before the quote and use `{}` to embed expressions.

```python
# Basic f-string
name = "Linear Regression"
print(f"Model: {name}")         # Model: Linear Regression

# Format specifiers (used heavily in course notebooks)
mse = 0.04523
r2 = 0.9134
print(f"Test MSE: {mse:.3f}")   # Test MSE: 0.045   (3 decimal places)
print(f"Test R2: {r2:.3f}")     # Test R2: 0.913
print(f"Accuracy: {r2:.1%}")    # Accuracy: 91.3%   (as percentage)

salary = 85000
print(f"Mean salary: ${salary:,.0f}")  # Mean salary: $85,000

# Embedding expressions
print(f"Shape: {X_train.shape}")       # Shape: (800, 8)
print(f"Features: {len(feature_cols)}")

# Common string methods
col_name = "Annual_Spending"
col_name.lower()                        # "annual_spending"
col_name.upper()                        # "ANNUAL_SPENDING"
col_name.replace("_", " ")             # "Annual Spending"
col_name.startswith("Annual")          # True
"  hello  ".strip()                    # "hello" (removes whitespace)
"a,b,c".split(",")                     # ["a", "b", "c"]
```

### 1.4 Lists

**Lists** are ordered, mutable collections. They are used everywhere in Python.

```python
# Creating lists
scores = [85, 92, 78, 95, 88]
features = ["age", "income", "spending_score"]
empty = []

# Indexing (starts at 0)
scores[0]                    # 85  (first element)
scores[-1]                   # 88  (last element)
scores[-2]                   # 95  (second to last)

# Slicing [start:stop] -- stop is excluded
scores[1:3]                  # [92, 78]  (index 1 and 2)
scores[:3]                   # [85, 92, 78]  (first 3)
scores[2:]                   # [78, 95, 88]  (from index 2 onward)

# Common methods
scores.append(91)            # Add to end: [85, 92, 78, 95, 88, 91]
scores.insert(0, 100)        # Insert at position 0
scores.remove(78)            # Remove first occurrence of 78
scores.pop()                 # Remove and return last element
scores.sort()                # Sort in place (ascending)

# Built-in functions with lists
len(scores)                  # Number of elements
sum(scores)                  # Sum of all elements
min(scores)                  # Smallest element
max(scores)                  # Largest element
sum(scores) / len(scores)    # Average

# Lists in ML context
feature_cols = ["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_COMB"]
degrees = [1, 2, 3, 5, 10]  # Polynomial degrees to compare
```

### 1.5 Dictionaries

**Dictionaries** store key-value pairs. They are used to store model results, hyperparameters, and mappings.

```python
# Creating a dictionary
model_scores = {
    "Linear": 0.85,
    "Polynomial": 0.92,
    "Ridge": 0.91
}

# Access values
model_scores["Linear"]              # 0.85
model_scores.get("Lasso", "N/A")    # "N/A" (safe access, returns default if key missing)

# Add or update entries
model_scores["Lasso"] = 0.89

# Useful methods
model_scores.keys()                  # dict_keys(["Linear", "Polynomial", ...])
model_scores.values()                # dict_values([0.85, 0.92, ...])
model_scores.items()                 # dict_items([("Linear", 0.85), ...])

# Iterate over dictionary
for name, score in model_scores.items():
    print(f"{name}: R2 = {score:.3f}")

# Create from two lists using zip
model_names = ["Linear", "Ridge", "Lasso"]
r2_values = [0.85, 0.91, 0.89]
results = dict(zip(model_names, r2_values))
```

### 1.6 Tuples and Sets

```python
# Tuples: immutable (cannot change after creation)
# Used for shapes and multiple return values
shape = (100, 8)                     # rows, columns
rows, cols = shape                   # tuple unpacking

# Multiple return values (common in sklearn)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Sets: unordered collection of unique elements
unique_labels = {0, 1, 2}
unique_labels.add(3)
len(unique_labels)                   # 4
```

### 1.7 Conditionals

```python
# if / elif / else
r2 = 0.85
if r2 >= 0.9:
    print("Excellent model")
elif r2 >= 0.7:
    print("Good model")
elif r2 >= 0.5:
    print("Moderate model")
else:
    print("Needs improvement")

# Ternary (one-line conditional)
status = "overfitting" if train_r2 - test_r2 > 0.1 else "ok"

# Checking for None
if result is not None:
    print(result)
```

### 1.8 Loops

```python
# for loop with range
for i in range(5):                   # 0, 1, 2, 3, 4
    print(f"Epoch {i + 1}")

# for loop over a list
for feature in feature_cols:
    print(feature)

# enumerate: get index AND value (used in plotting code)
features_to_plot = ["alcohol", "color_intensity", "flavanoids"]
for idx, feature in enumerate(features_to_plot):
    axes[idx].set_title(f"Distribution of {feature}")

# zip: iterate multiple lists in parallel
for name, score in zip(model_names, r2_values):
    print(f"{name}: {score:.3f}")

# Nested loops (seen in grid visualizations)
for i in range(grid_size):
    for j in range(grid_size):
        value = matrix[i][j]

# while loop
epoch = 0
while epoch < 100:
    # training code here
    epoch += 1

# break: exit the loop early
for i in range(100):
    if accuracy > 0.99:
        break
```

### 1.9 Functions

```python
# Basic function
def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error."""
    mse = np.mean((y_true - y_pred) ** 2)
    return np.sqrt(mse)

# Call the function
score = calculate_rmse(y_test, y_pred)

# Function with default arguments
def evaluate_model(model, X_test, y_test, metric="r2"):
    y_pred = model.predict(X_test)
    if metric == "r2":
        return r2_score(y_test, y_pred)
    return mean_squared_error(y_test, y_pred)

# Using default vs overriding
r2 = evaluate_model(model, X_test, y_test)              # uses default metric="r2"
mse = evaluate_model(model, X_test, y_test, metric="mse")  # override default

# Multiple return values
def get_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2

mse, r2 = get_metrics(y_test, y_pred)  # unpack returned tuple
```

### 1.10 List Comprehensions

A compact way to create lists. Think of it as a one-line `for` loop.

```python
# Basic list comprehension
squares = [x ** 2 for x in range(10)]
# Equivalent to:
# squares = []
# for x in range(10):
#     squares.append(x ** 2)

# With condition (filtering)
region_cols = [col for col in df.columns if col.startswith("region_")]

# From course notebooks
positions = [0.1 + i * 0.1 for i in range(len(steps))]
labels = [target_names[i] for i in y_wine]
```

### 1.11 Lambda Functions

**Lambda** functions are small anonymous (unnamed) functions written in one line.

```python
# Lambda: a one-line anonymous function
square = lambda x: x ** 2
square(5)                              # 25

# From course notebooks (defining mathematical functions)
y_true_func = lambda x: np.sin(2 * np.pi * x)

# Common use: custom sorting
models_sorted = sorted(results, key=lambda x: x["score"], reverse=True)
```

---

## 2. NumPy Essentials

**NumPy** provides fast array operations. Every ML library in Python builds on NumPy.

```python
import numpy as np
```

### 2.1 Array Creation

```python
# From a list
arr = np.array([1, 2, 3, 4, 5])
matrix = np.array([[1, 2, 3], [4, 5, 6]])

# Zeros and ones
zeros = np.zeros((3, 4))              # 3x4 matrix of zeros
ones = np.ones((2, 3))                # 2x3 matrix of ones

# Evenly spaced values
x = np.linspace(0, 1, 60)             # 60 points from 0 to 1 (inclusive)
x = np.arange(0, 10, 0.5)             # 0, 0.5, 1.0, ..., 9.5

# Random numbers (used in course for reproducibility and data generation)
np.random.seed(42)                     # Set seed for reproducibility
noise = np.random.normal(0, 0.3, 100) # mean=0, std=0.3, 100 samples
ages = np.random.uniform(0, 15, 200)  # uniform between 0 and 15
idx = np.random.randint(0, 100, 20)   # 20 random ints from 0-99
data = np.random.rand(5, 3)           # 5x3 matrix of random floats [0, 1)
```

### 2.2 Array Properties and Indexing

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

# Properties
arr.shape                              # (2, 3) -- 2 rows, 3 columns
arr.ndim                               # 2 -- number of dimensions
arr.dtype                              # int64 -- data type
arr.size                               # 6 -- total number of elements

# Indexing
arr[0, 1]                              # 2 (row 0, col 1)
arr[0, :]                              # [1, 2, 3] (entire first row)
arr[:, 0]                              # [1, 4] (entire first column)
arr[0:2, 1:3]                          # sub-matrix

# Boolean indexing (used for filtering data)
values = np.array([10, 25, 5, 30, 15])
values[values > 15]                    # array([25, 30])
values[values != 5]                    # array([10, 25, 30, 15])
```

### 2.3 Array Operations

```python
# Element-wise operations (no loops needed)
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
a + b                                  # array([5, 7, 9])
a * b                                  # array([4, 10, 18])
a ** 2                                 # array([1, 4, 9])
a / b                                  # array([0.25, 0.4, 0.5])

# Scalar operations (applied to every element)
a * 2                                  # array([2, 4, 6])
a + 10                                 # array([11, 12, 13])

# Aggregation (used in evaluation metrics)
predictions = np.array([3.1, 2.8, 4.5])
actuals = np.array([3.0, 3.0, 4.0])
errors = actuals - predictions
np.mean(errors ** 2)                   # MSE
np.sqrt(np.mean(errors ** 2))          # RMSE

# Common aggregation functions
arr.mean()                             # Average
arr.std()                              # Standard deviation
arr.sum()                              # Sum
arr.min()                              # Minimum value
arr.max()                              # Maximum value

# Mathematical functions
np.sqrt(16)                            # 4.0
np.abs(-5)                             # 5
np.round(3.14159, 2)                   # 3.14
np.log(np.e)                           # 1.0
np.exp(1)                              # 2.718...
```

### 2.4 Reshaping

Reshaping is critical because **scikit-learn expects 2D input** (rows x columns), but sometimes your data is 1D.

```python
# The problem: sklearn needs 2D
X = np.linspace(0, 1, 60)             # shape: (60,) -- 1D
# model.fit(X, y)  -->  ERROR: Expected 2D array, got 1D

# The fix: reshape to column vector
X = X.reshape(-1, 1)                   # shape: (60, 1) -- 2D
# -1 means "figure out this dimension automatically"

# Flatten back to 1D
y = predictions.ravel()                # shape: (n,)
y = predictions.flatten()              # same result, returns a copy

# Transpose
matrix = np.array([[1, 2], [3, 4]])
matrix.T                               # [[1, 3], [2, 4]]
```

---

## 3. Pandas Essentials

**Pandas** is the primary tool for loading, exploring, and preprocessing data in this course.

```python
import pandas as pd
```

### 3.1 Series and DataFrame

A **Series** is a labeled 1D array. A **DataFrame** is a labeled 2D table (like a spreadsheet).

```python
# Series
scores = pd.Series([85, 92, 78], index=["Alice", "Bob", "Carol"])
scores["Alice"]                        # 85

# DataFrame from a dictionary
df = pd.DataFrame({
    "feature_1": [1.2, 3.4, 5.6],
    "feature_2": [7.8, 9.0, 1.2],
    "target": [0, 1, 1]
})

# Access a column (returns a Series)
df["target"]
df.target                              # Same thing (only works if no spaces in name)

# Access multiple columns (returns a DataFrame)
df[["feature_1", "feature_2"]]
```

### 3.2 Loading Data

```python
# Standard data loading pattern for this course
df = pd.read_csv('../Datasets/FuelConsumptionCo2.csv')

# Excel files
df = pd.read_excel('../Datasets/ENB2012_data.xlsx')

# Common parameters
df = pd.read_csv('file.csv', sep=';')       # Different separator
df = pd.read_csv('file.csv', header=0)      # First row is header (default)
```

### 3.3 Exploring Data

This is the standard exploration pattern used throughout the course notebooks.

```python
# Shape: how many rows and columns
print("Shape:", df.shape)                    # (1067, 12)

# First and last rows
print(df.head())                             # First 5 rows
print(df.tail(3))                            # Last 3 rows
print(df.sample(5))                          # 5 random rows

# Column names and types
print(df.columns)                            # List of column names
print(df.dtypes)                             # Data type of each column
print(df.info())                             # Summary: columns, types, non-null counts

# Statistical summary
print(df.describe())                         # Count, mean, std, min, 25%, 50%, 75%, max

# Missing values
print(df.isnull().sum())                     # Count of missing values per column

# Unique values in a column
print(df["FUELTYPE"].nunique())              # Number of unique values
print(df["FUELTYPE"].value_counts())         # Frequency of each value
```

### 3.4 Selecting and Filtering Data

```python
# Select single column
y = df["CO2EMISSIONS"]

# Select multiple columns
feature_cols = ["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_COMB"]
X = df[feature_cols]

# Filter rows with a condition
high_emission = df[df["CO2EMISSIONS"] > 300]

# Multiple conditions: use & (AND), | (OR), wrap each in parentheses
filtered = df[(df["CO2EMISSIONS"] > 200) & (df["FUELTYPE"] == "Z")]
either = df[(df["FUELTYPE"] == "X") | (df["FUELTYPE"] == "Z")]

# .loc: label-based access (row labels + column names)
df.loc[0:5, "ENGINESIZE"]                    # Rows 0-5, column ENGINESIZE
df.loc[df["CO2EMISSIONS"] > 300, "MAKE"]     # MAKE column where CO2 > 300

# .iloc: position-based access (row/column numbers)
df.iloc[0:5, 0:3]                            # First 5 rows, first 3 columns
df.iloc[0, :]                                # First row, all columns
```

### 3.5 Handling Missing Data

```python
# Check for missing values
print(df.isnull().sum())

# Drop rows with any missing values
df_clean = df.dropna()

# Drop rows with missing values in specific columns only
df_clean = df.dropna(subset=["income", "spending_score"])

# Fill missing values with a constant
df["income"] = df["income"].fillna(0)

# Fill with column statistics
df["income"] = df["income"].fillna(df["income"].median())
df["income"] = df["income"].fillna(df["income"].mean())

# Using sklearn SimpleImputer (more robust approach)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
```

### 3.6 GroupBy and Aggregation

```python
# Group by a column and calculate mean
avg_co2_by_fuel = df.groupby("FUELTYPE")["CO2EMISSIONS"].mean()

# Sort results
avg_co2_by_fuel = avg_co2_by_fuel.sort_values(ascending=False)

# Multiple aggregations
stats = df.groupby("FUELTYPE")["CO2EMISSIONS"].agg(["mean", "std", "count"])

# From course notebooks (cluster analysis)
cluster_means = df.groupby("Cluster")[
    ["Annual_Spending", "Visit_Frequency", "Avg_Purchase_Value"]
].mean()

# Sort a DataFrame by column values
df_sorted = df.sort_values("CO2EMISSIONS", ascending=False)
```

### 3.7 Adding and Modifying Columns

```python
# Create a new column from existing ones
df["log_income"] = np.log(df["income"])
df["price_per_sqft"] = df["price"] / df["sqft"]

# Drop columns
df = df.drop("unwanted_column", axis=1)
df = df.drop(["col1", "col2"], axis=1)

# Map values (encoding categories to numbers)
membership_mapping = {"Basic": 0, "Silver": 1, "Gold": 2}
df["membership_encoded"] = df["membership_type"].map(membership_mapping)

# Apply a custom function
df["age_group"] = df["age"].apply(lambda x: "senior" if x > 60 else "adult")

# One-hot encoding (converting categories to binary columns)
region_dummies = pd.get_dummies(df["region"], prefix="region")
df = pd.concat([df, region_dummies], axis=1)

# Concatenate DataFrames
df = pd.concat([df1, df2], axis=0)     # Stack rows (vertically)
df = pd.concat([X, y], axis=1)         # Join columns (side by side)
```

---

## 4. Visualization with Matplotlib and Seaborn

```python
import matplotlib.pyplot as plt
import seaborn as sns
```

### 4.1 Figure and Axes Setup

```python
# Configuration (standard in course notebooks)
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 300
sns.set_style("whitegrid")

# Single plot
plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title("My Plot")
plt.tight_layout()
plt.show()

# Multiple subplots (used extensively in the course)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes[0, 0].plot(x, y)                 # Access subplot by [row, col]
axes[0, 1].scatter(x, y)
axes[1, 0].hist(data)
plt.tight_layout()
plt.show()

# Single row of subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].plot(x, y)                     # No row index needed for 1 row
```

### 4.2 Common Plot Types

```python
# Line plot (trends, model predictions)
plt.plot(x, y, color="blue", linewidth=2, label="Predicted")

# Scatter plot (actual vs predicted, feature relationships)
plt.scatter(y_test, y_pred, alpha=0.6, color="teal", label="Predictions")
plt.plot([y_min, y_max], [y_min, y_max], "r--", label="Perfect Prediction")

# Histogram (distribution of a variable)
plt.hist(df["CO2EMISSIONS"], bins=30, alpha=0.7, color="skyblue", edgecolor="black")

# Bar chart (comparing categories or models)
plt.bar(model_names, r2_scores, color=["blue", "green", "orange"])
for i, v in enumerate(r2_scores):
    plt.text(i, v, f"{v:.3f}", ha="center", va="bottom")

# Quick DataFrame histograms (for EDA)
df.hist(figsize=(12, 8))
plt.tight_layout()
plt.show()
```

### 4.3 Plot Customization

```python
# Full plot with labels, title, legend, and grid
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, alpha=0.6, label="Actual")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Predicted")
plt.title("Model Predictions vs Actual Values", fontsize=14)
plt.xlabel("Feature Values")
plt.ylabel("Target Variable")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Rotate x-axis labels (useful for long category names)
plt.xticks(rotation=45, ha="right")

# Reference lines
ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
ax.axvline(x=mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.1f}")

# Text annotations
ax.text(x_pos, y_pos, f"R2 = {r2:.3f}", fontsize=12, ha="center")
```

### 4.4 Seaborn for Statistical Plots

```python
# Correlation heatmap (used in every EDA section of the course)
correlation_matrix = df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0,
            square=True, linewidths=1, fmt=".3f")
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.show()

# Style and palette settings
sns.set_style("whitegrid")            # Clean grid background
sns.set_palette("husl")               # Color palette
```

---

## 5. Common ML Code Patterns

These patterns are used repeatedly throughout the course. Learn them once and reuse everywhere.

### 5.1 Standard Imports

```python
# Data manipulation and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

# Configuration
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 300
sns.set_style("whitegrid")
np.random.seed(42)  # Change to last 2 digits of your student ID for assignments
```

### 5.2 Data Loading and EDA Pattern

```python
# 1. Load data
df = pd.read_csv('../Datasets/FuelConsumptionCo2.csv')

# 2. Explore structure
print("Shape:", df.shape)
print(df.head())
print(df.describe())
print(df.isnull().sum())

# 3. Visualize distributions
df.hist(figsize=(12, 8))
plt.tight_layout()
plt.show()

# 4. Check correlations
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()
```

### 5.3 Feature and Target Split

```python
# Select specific feature columns
feature_cols = ["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_COMB"]
X = df[feature_cols]
y = df["CO2EMISSIONS"]

# Alternative: select all columns except the target
X = df.drop("CO2EMISSIONS", axis=1)
y = df["CO2EMISSIONS"]
```

### 5.4 Train/Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set:     {X_test.shape[0]} samples")
```

### 5.5 Feature Scaling

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit AND transform training data
X_test_scaled = scaler.transform(X_test)         # Only transform test data
```

**Important:** Never call `fit_transform` on test data -- only `transform`. The scaler must learn its parameters (mean, standard deviation) from training data only. Using test data to fit the scaler would leak information.

### 5.6 Model Training and Prediction

This 3-step pattern is the **same for every scikit-learn model**.

```python
from sklearn.linear_model import LinearRegression

# Step 1: Instantiate the model
model = LinearRegression()

# Step 2: Fit (train) the model on training data
model.fit(X_train, y_train)

# Step 3: Make predictions on test data
y_pred = model.predict(X_test)
```

### 5.7 Model Evaluation

```python
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print(f"Test MSE:  {mse:.3f}")
print(f"Test RMSE: {rmse:.3f}")
print(f"Test MAE:  {mae:.3f}")
print(f"Test R2:   {r2:.3f}")
```

### 5.8 Cross-Validation

Cross-validation gives a more reliable estimate of model performance by testing on multiple data splits.

```python
from sklearn.model_selection import cross_val_score

cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
print(f"CV R2 scores: {cv_scores}")
print(f"CV R2: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
```

### 5.9 Comparing Multiple Models

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso

models = {
    "Linear": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=1.0)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"{name:12s} | R2: {r2:.3f} | RMSE: {rmse:.3f}")
```

---

## 6. Quick Reference Tables

### 6.1 Python Built-in Functions

| Function | What It Does | Example |
|----------|-------------|---------|
| `len()` | Length of a sequence | `len(my_list)` |
| `type()` | Type of an object | `type(42)` returns `int` |
| `range()` | Sequence of integers | `range(0, 10, 2)` |
| `enumerate()` | Index + value pairs | `for i, v in enumerate(lst)` |
| `zip()` | Pair elements from lists | `for a, b in zip(x, y)` |
| `sorted()` | Return sorted copy | `sorted(scores, reverse=True)` |
| `print()` | Display output | `print(f"R2: {r2:.3f}")` |
| `isinstance()` | Check type | `isinstance(x, np.ndarray)` |
| `round()` | Round a number | `round(3.14159, 2)` returns `3.14` |
| `abs()` | Absolute value | `abs(-5)` returns `5` |
| `sum()` | Sum of iterable | `sum([1, 2, 3])` returns `6` |
| `min()` / `max()` | Min/max of iterable | `max([1, 5, 3])` returns `5` |

### 6.2 NumPy Cheat Sheet

| Operation | Code | Note |
|-----------|------|------|
| Create array | `np.array([1, 2, 3])` | From list |
| Zeros / Ones | `np.zeros((3, 4))` | Shape as tuple |
| Range | `np.linspace(0, 1, 50)` | 50 evenly spaced |
| Random normal | `np.random.normal(0, 1, 100)` | mean, std, size |
| Set seed | `np.random.seed(42)` | Reproducibility |
| Shape | `arr.shape` | Returns tuple |
| Reshape | `arr.reshape(-1, 1)` | -1 = auto-infer |
| Mean | `arr.mean()` or `np.mean(arr)` | Along axis if needed |
| Std | `arr.std()` | Standard deviation |
| Square root | `np.sqrt(value)` | Element-wise |
| Round | `np.round(arr, 3)` | To 3 decimals |
| Absolute | `np.abs(arr)` | Element-wise |

### 6.3 Pandas Cheat Sheet

| Operation | Code | Note |
|-----------|------|------|
| Load CSV | `pd.read_csv('file.csv')` | Returns DataFrame |
| First rows | `df.head()` | Default 5 rows |
| Shape | `df.shape` | (rows, cols) |
| Column types | `df.dtypes` | All column types |
| Statistics | `df.describe()` | Count, mean, std, etc. |
| Missing values | `df.isnull().sum()` | Per column |
| Drop missing | `df.dropna()` | Remove rows with NaN |
| Fill missing | `df.fillna(value)` | Replace NaN |
| Select columns | `df[["col1", "col2"]]` | Double brackets |
| Filter rows | `df[df["col"] > 5]` | Boolean indexing |
| Group by | `df.groupby("col").mean()` | Aggregate by group |
| Value counts | `df["col"].value_counts()` | Frequency of each value |
| Sort | `df.sort_values("col")` | ascending=True default |
| New column | `df["new"] = df["a"] + df["b"]` | Element-wise |
| Drop column | `df.drop("col", axis=1)` | axis=1 for columns |
| One-hot encode | `pd.get_dummies(df["col"])` | Creates binary columns |
| Concatenate | `pd.concat([df1, df2], axis=1)` | axis=0 rows, axis=1 cols |

### 6.4 Matplotlib Cheat Sheet

| Plot Type | Code | When to Use |
|-----------|------|-------------|
| Line | `plt.plot(x, y)` | Trends, predictions |
| Scatter | `plt.scatter(x, y)` | Relationships between variables |
| Histogram | `plt.hist(data, bins=30)` | Distribution of a variable |
| Bar | `plt.bar(labels, values)` | Comparing categories |
| Heatmap | `sns.heatmap(matrix)` | Correlation matrices |
| Subplots | `fig, axes = plt.subplots(2, 3)` | Multiple plots in a grid |

### 6.5 Scikit-learn Pattern Summary

| Step | Code | Note |
|------|------|------|
| Split data | `train_test_split(X, y, test_size=0.2)` | Returns 4 arrays |
| Scale features | `scaler.fit_transform(X_train)` | Fit only on training data |
| Scale test | `scaler.transform(X_test)` | Never fit on test data |
| Train model | `model.fit(X_train, y_train)` | Learns parameters |
| Predict | `model.predict(X_test)` | Returns predictions |
| Evaluate | `r2_score(y_test, y_pred)` | Compare actual vs predicted |
| Cross-validate | `cross_val_score(model, X, y, cv=5)` | Returns array of scores |

---

## 7. Common Errors and Debugging Tips

### 7.1 Python Errors

| Error | Common Cause | Fix |
|-------|-------------|-----|
| `NameError: name 'pd' is not defined` | Forgot to import pandas | Add `import pandas as pd` |
| `IndentationError` | Inconsistent spaces/tabs | Use 4 spaces consistently |
| `TypeError: unsupported operand type` | Wrong type in operation | Check `type(variable)` |
| `IndexError: list index out of range` | Index too large | Check `len(list)` first |
| `KeyError: 'column_name'` | Column not in DataFrame | Check `df.columns` |
| `SyntaxError: unexpected EOF` | Missing closing bracket or quote | Check matching `()`, `[]`, `""` |

### 7.2 Data Errors

| Error | Common Cause | Fix |
|-------|-------------|-----|
| `FileNotFoundError` | Wrong file path | Use `'../Datasets/filename.csv'` |
| `ValueError: could not convert string to float` | Non-numeric data in features | Check `df.dtypes`, encode categoricals |
| NaN in results | Missing values in data | Use `df.isnull().sum()` to find them |
| Unexpected shape | Wrong array dimensions | Print `array.shape` to debug |

### 7.3 Scikit-learn Errors

| Error | Common Cause | Fix |
|-------|-------------|-----|
| `ValueError: Expected 2D array, got 1D` | sklearn needs 2D input | Use `.reshape(-1, 1)` |
| `NotFittedError` | Called predict before fit | Call `model.fit()` first |
| `ValueError: inconsistent numbers of samples` | X and y have different lengths | Check `.shape` of both |
| Low R2 score | Model underfitting | Try more features or polynomial features |
| Train R2 much higher than test R2 | Model overfitting | Reduce complexity or add regularization |

### 7.4 Debugging Checklist

1. **Read the error message** -- the last line tells you the error type and message
2. **Check the line number** mentioned in the traceback
3. **Print the variable** to inspect its type, shape, and value
4. **Make sure all previous cells have been run** in order (top to bottom)
5. **Restart kernel and run all cells** from the top if things seem out of sync
6. Use `%whos` in a notebook cell to list all current variables

```python
# Quick debugging pattern
print(f"Type:  {type(your_variable)}")
print(f"Shape: {getattr(your_variable, 'shape', 'N/A')}")
print(f"Value: {your_variable}")
```

---

## 8. Resources

- [Python Documentation](https://docs.python.org/3/)
- [NumPy Documentation](https://numpy.org/doc/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Tutorials](https://matplotlib.org/stable/tutorials/index.html)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Python for Data Science Handbook (free)](https://jakevdp.github.io/PythonDataScienceHandbook/)

---

**You don't need to memorize everything here. Bookmark this guide and come back to it whenever you need a quick reminder!**

---

[← Previous: VSCode Tips](06_VSCODE_TIPS_GUIDE.md) | [Index](README.md) | [Next: Mathematics for ML →](08_MATH_FOR_ML_GUIDE.md)
