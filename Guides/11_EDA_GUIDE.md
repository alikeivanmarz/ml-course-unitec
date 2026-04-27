# Exploratory Data Analysis

Exploratory data analysis (EDA) is the inspection phase that precedes any modelling decision. Its purpose is to surface the structure, quality, and pitfalls of a dataset before feature engineering or model selection begins. This guide presents a workflow for tabular data: what to inspect, in what order, and what to record. Code examples use pandas, matplotlib, seaborn, and scikit-learn.

**Table of Contents**

1. [Initial Inspection](#1-initial-inspection)
2. [Missing Data Audit](#2-missing-data-audit)
3. [Univariate Distributions](#3-univariate-distributions)
4. [Bivariate Relationships](#4-bivariate-relationships)
5. [Target Leakage Checks](#5-target-leakage-checks)
6. [Outlier Detection](#6-outlier-detection)
7. [Feature–Target Relationships](#7-feature-target-relationships)
8. [EDA Report Structure](#8-eda-report-structure)
9. [Resources](#9-resources)

---

## 1. Initial Inspection

The first ten minutes with a new dataset establish what is present, what is typed correctly, and what is suspicious. No transformations or imputations should be made before this phase completes.

### 1.1 Shape and Types

```python
import pandas as pd

df = pd.read_csv("data.csv")

df.shape          # (n_rows, n_cols)
df.dtypes         # column-wise dtypes
df.info()         # dtypes + non-null counts in one view
df.head()         # first 5 rows
df.sample(5)      # random rows — better than head() for sorted data
```

`info()` is the single most useful summary: it reports row count, column dtypes, and non-null counts in one call.

### 1.2 Summary Statistics

```python
df.describe()                          # numeric columns only by default
df.describe(include="object")          # categorical / string columns
df.describe(include="all")             # everything
```

Inspect for: minima below physical possibility (negative ages, zero prices), maxima at suspicious round numbers (data caps), means far from medians (skew), and standard deviations of zero (constant columns).

### 1.3 First-10-Minutes Checklist

| # | Check | Method |
|---|-------|--------|
| 1 | Row and column count | `df.shape` |
| 2 | Column dtypes — any object that should be numeric or datetime | `df.dtypes` |
| 3 | Missing-value counts per column | `df.isna().sum()` |
| 4 | Duplicate rows | `df.duplicated().sum()` |
| 5 | Constant columns (zero variance) | `df.nunique() == 1` |
| 6 | High-cardinality categoricals (potential IDs) | `df.select_dtypes("object").nunique()` |
| 7 | Numeric ranges (min / max / mean) | `df.describe()` |
| 8 | Class balance, if classification | `df["target"].value_counts(normalize=True)` |
| 9 | Date columns parsed correctly | `pd.to_datetime(df["date"], errors="coerce").isna().sum()` |
| 10 | Sample of raw rows | `df.sample(10)` |

---

## 2. Missing Data Audit

Missingness affects every downstream choice: imputation strategy, feature drop decisions, model selection (some models tolerate NaN, most do not), and the validity of summary statistics.

### 2.1 Counts and Percentages

```python
missing = df.isna().sum().sort_values(ascending=False)
missing_pct = (missing / len(df) * 100).round(2)
pd.concat([missing, missing_pct], axis=1, keys=["count", "percent"])
```

### 2.2 Visualizing Missingness

The `missingno` library renders missingness patterns directly:

```python
# pip install missingno
import missingno as msno

msno.matrix(df)       # column-wise pattern across rows
msno.heatmap(df)      # nullity correlation between columns
msno.bar(df)          # bar chart of present values per column
```

A pattern in the missingness matrix (vertical stripes aligned across columns) suggests structural missingness — typically caused by a data-collection event rather than randomness.

### 2.3 MCAR, MAR, MNAR

| Type | Meaning | Implication |
|------|---------|-------------|
| MCAR | Missing Completely At Random | Safe to drop or impute with a simple statistic |
| MAR | Missing At Random — depends on other observed columns | Model-based imputation appropriate |
| MNAR | Missing Not At Random — depends on the missing value itself | Hardest case; consider an explicit "missing" indicator |

Distinguishing MAR from MNAR cannot be done from the data alone — it requires domain knowledge of the collection process.

---

## 3. Univariate Distributions

Examine one variable at a time before considering relationships. Distributions reveal skew, multimodality, and discretization artefacts.

### 3.1 Numeric Variables

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.histplot(df["x"], kde=True, ax=axes[0])     # distribution shape
sns.boxplot(x=df["x"], ax=axes[1])              # outliers and quartiles
sns.violinplot(x=df["x"], ax=axes[2])           # density + quartiles combined
```

Right-skewed numeric features often benefit from log or Box-Cox transformation before linear models. Bimodality suggests a mixture of subpopulations and may justify a categorical split feature.

### 3.2 Categorical Variables

```python
counts = df["category"].value_counts()
counts.head(20).plot(kind="bar")                # truncate when cardinality is large
```

For high-cardinality categoricals (>50 unique values), inspect both the head (most common) and the long tail. Long tails often justify grouping rare categories into "Other" before encoding.

### 3.3 Plot-Type Decision Table

| Variable type | Recommended plot | Notes |
|---------------|------------------|-------|
| Single numeric | Histogram + KDE | Box plot to focus on outliers |
| Single categorical | Bar chart of value counts | Sort by frequency |
| Numeric vs numeric | Scatter (small n) or hexbin (large n) | Add `alpha=0.3` for overdraw |
| Numeric vs categorical | Box plot or violin plot per category | Strip plot for small n |
| Categorical vs categorical | Stacked bar or heatmap of crosstab | Normalize rows to compare proportions |
| Time series | Line plot | Resample if sampling rate is irregular |

---

## 4. Bivariate Relationships

After establishing univariate behaviour, investigate how variables relate. Bivariate analysis suggests interactions, redundancies, and candidate features.

### 4.1 Numeric–Numeric

```python
# Pairwise scatter for a small set of features
sns.pairplot(df[["x1", "x2", "x3", "target"]], hue="target")

# Correlation matrix
corr = df.select_dtypes("number").corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
```

Pearson correlation captures linear association only. Spearman (`df.corr(method="spearman")`) captures monotonic relationships and is robust to outliers.

### 4.2 Numeric–Categorical

```python
sns.boxplot(data=df, x="category", y="value")
df.groupby("category")["value"].agg(["mean", "median", "std", "count"])
```

Wide differences in group medians suggest the categorical variable carries predictive signal.

### 4.3 Categorical–Categorical

```python
# Contingency table
ct = pd.crosstab(df["a"], df["b"])

# Normalized — proportions within each row
pd.crosstab(df["a"], df["b"], normalize="index")

sns.heatmap(ct, annot=True, fmt="d", cmap="Blues")
```

A chi-square test (`scipy.stats.chi2_contingency(ct)`) quantifies whether the observed frequencies differ from independence.

---

## 5. Target Leakage Checks

Target leakage is the contamination of training features with information that would not be available at prediction time. It produces inflated validation scores and silent production failures. Common sources:

| Source | Example | Detection |
|--------|---------|-----------|
| Future information | A "settlement_date" used to predict claim status | Compare timestamps against the prediction-time horizon |
| Outcome encoding | A "loan_default_amount" feature used to predict default | Suspiciously high correlation with the target |
| Group leakage | Same patient in train and test sets | Check for shared identifiers across splits |
| Preprocessing leakage | Scaler fit on train+test combined | Audit the preprocessing pipeline order |

A single feature with a >0.95 correlation to a binary target is almost always leakage and should be investigated before modelling proceeds.

---

## 6. Outlier Detection

Outliers may be measurement errors, data-entry mistakes, or genuine extreme observations. The treatment depends on the cause; the first step is identification.

### 6.1 IQR Rule

```python
q1 = df["x"].quantile(0.25)
q3 = df["x"].quantile(0.75)
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
outliers = df[(df["x"] < lower) | (df["x"] > upper)]
```

Robust to skew but tied to a fixed multiplier (1.5) that may not suit all distributions.

### 6.2 Z-Score

```python
from scipy.stats import zscore

z = zscore(df["x"].dropna())
outliers = df.loc[df["x"].dropna().index][abs(z) > 3]
```

Assumes approximate normality. Sensitive to the outliers themselves (the standard deviation is inflated by them).

### 6.3 Isolation Forest

For multivariate outlier detection, isolation forest scores points by how easily they are isolated by random splits.

```python
from sklearn.ensemble import IsolationForest

iso = IsolationForest(contamination=0.05, random_state=0)
labels = iso.fit_predict(df.select_dtypes("number").dropna())
# -1 indicates an outlier; 1 indicates an inlier
```

`contamination` is the assumed fraction of outliers; tune it to match prior expectation.

---

## 7. Feature–Target Relationships

Quantify how informative each feature is about the target. These rankings inform feature selection and model interpretation, but should not be used as the sole basis for inclusion — interactions can make individually weak features valuable in combination.

### 7.1 Correlation

For numeric features and a numeric target:

```python
df.corr(numeric_only=True)["target"].sort_values(ascending=False)
```

For ordinal or non-linear monotonic relationships, use Spearman instead of Pearson.

### 7.2 Mutual Information

Mutual information captures arbitrary (including non-linear) dependencies. It works for both regression and classification targets.

```python
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

mi = mutual_info_classif(X, y, random_state=0)   # for classification
mi = mutual_info_regression(X, y, random_state=0)  # for regression

import pandas as pd
pd.Series(mi, index=X.columns).sort_values(ascending=False)
```

Mutual information is non-negative; zero means independence. Values are not directly comparable across datasets but ranking within a dataset is meaningful.

### 7.3 ANOVA F-Test

For numeric features and a categorical target, the ANOVA F-statistic measures separation of class means:

```python
from sklearn.feature_selection import f_classif

f, p = f_classif(X, y)
```

High F values indicate that class means differ relative to within-class variance. Assumes approximately normal within-class distributions and equal variances.

---

## 8. EDA Report Structure

An EDA report serves two audiences: the analyst's future self (a record of what was checked and discovered) and downstream consumers (modellers, reviewers, stakeholders). A standard structure:

1. **Dataset overview** — source, time range, row and column counts, schema.
2. **Data quality summary** — missingness, duplicates, constant columns, type coercions applied.
3. **Univariate findings** — distribution shape and notable features per column. Skip uninteresting columns.
4. **Bivariate findings** — pairs that show strong relationships, suspicious correlations, candidate interactions.
5. **Target analysis** — class balance (classification) or target distribution (regression); top features by correlation and mutual information.
6. **Leakage and integrity checks** — identifiers, timestamps, splits, suspicious near-perfect predictors.
7. **Open questions** — anomalies that require domain input before modelling.

Discard: pretty plots that confirm nothing, repeated views of the same finding, and exhaustive enumeration when a summary table suffices.

---

## 9. Resources

- [Pandas user guide — exploratory data analysis](https://pandas.pydata.org/docs/user_guide/index.html) — official patterns for inspection, indexing, and aggregation.
- [Seaborn — statistical data visualization](https://seaborn.pydata.org/) — plot gallery covering most EDA needs.
- [`missingno`](https://github.com/ResidentMario/missingno) — visual missingness analysis.
- [`pandas-profiling` / `ydata-profiling`](https://github.com/ydataai/ydata-profiling) — automated EDA reports as a starting point.
- [Tukey, *Exploratory Data Analysis*](https://www.pearson.com/store/p/exploratory-data-analysis/P100000888193) — the original treatment of EDA as a discipline.
- [scikit-learn — feature selection module](https://scikit-learn.org/stable/modules/feature_selection.html) — mutual information, ANOVA, and other univariate selectors.

---

[← Previous: Datasets](10_DATASETS_GUIDE.md) | [Index](README.md) | [Next: Data Preprocessing →](12_DATA_PREPROCESSING_GUIDE.md)
