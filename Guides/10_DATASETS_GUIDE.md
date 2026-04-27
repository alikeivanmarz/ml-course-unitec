# Dataset Sourcing and Loading

Choosing a dataset, obtaining it, and loading it correctly are upstream of every modelling decision. This guide covers the major sources of public datasets, the loading patterns associated with each, the file formats encountered in practice, and the splitting and sampling strategies needed to build honest train/validation/test partitions. Licensing and documentation conventions are included for use in shared or published work.

**Table of Contents**

1. [Dataset Sources](#1-dataset-sources)
2. [Loading Patterns](#2-loading-patterns)
3. [Common File Formats](#3-common-file-formats)
4. [Train / Validation / Test Splitting](#4-train--validation--test-splitting)
5. [Sampling Strategies](#5-sampling-strategies)
6. [Licensing and Attribution](#6-licensing-and-attribution)
7. [Documenting a Dataset](#7-documenting-a-dataset)
8. [Resources](#8-resources)

---

## 1. Dataset Sources

### 1.1 General-Purpose Repositories

| Source | Strengths | Notes |
|--------|-----------|-------|
| [Kaggle Datasets](https://www.kaggle.com/datasets) | Largest community catalogue; includes many competition datasets | Requires account; some licenses restrict commercial use |
| [HuggingFace Datasets Hub](https://huggingface.co/datasets) | Strong NLP, vision, audio coverage; streaming support for large sets | Loaded via `datasets` library; first-class versioning |
| [UCI Machine Learning Repository](https://archive.ics.uci.edu/) | Long-running collection of classical tabular datasets | Often small; useful for benchmarking |
| [OpenML](https://www.openml.org/) | Curated tabular datasets with task definitions and shared results | Programmatic access via `openml` package |
| [Google Dataset Search](https://datasetsearch.research.google.com/) | Cross-source discovery engine | Indexes data published with schema.org metadata |
| [Data.gov](https://data.gov/) and national equivalents | Government statistics, geospatial, public records | Licensing usually permissive |

### 1.2 Library Built-Ins

| Library | Module | Coverage |
|---------|--------|----------|
| scikit-learn | `sklearn.datasets` | Classical small datasets (iris, digits, california housing) and synthetic generators (`make_classification`, `make_regression`, `make_blobs`) |
| TensorFlow | `tf.keras.datasets` | MNIST, CIFAR-10/100, IMDB, Reuters, Fashion-MNIST |
| TensorFlow Datasets | `tensorflow_datasets` | Hundreds of datasets with a unified loader |
| TorchVision | `torchvision.datasets` | MNIST, CIFAR, ImageNet, COCO, and related |
| TorchText / TorchAudio | `torchtext.datasets`, `torchaudio.datasets` | Text and audio benchmarks |

Built-in datasets are useful for prototyping and reproducible examples but are saturated benchmarks; results on them rarely transfer to applied problems.

### 1.3 Source Comparison

| Source | Account required | Typical size | License clarity | API quality |
|--------|------------------|--------------|------------------|-------------|
| Kaggle | Yes | MB–GB | Per-dataset; varies | Web download or `kaggle` CLI |
| HuggingFace Datasets | No | KB–TB (streaming) | Per-dataset; usually clear | Single `load_dataset()` call |
| UCI | No | KB–MB | Generally permissive | Direct file download |
| OpenML | Optional | KB–GB | Usually clear | `openml` Python client |
| sklearn built-ins | No | KB–MB | Public domain or BSD | One function per dataset |

---

## 2. Loading Patterns

### 2.1 Pandas

The default loader for tabular data in CSV, TSV, Excel, JSON, Parquet, and SQL formats.

```python
import pandas as pd

df = pd.read_csv("data.csv")
df = pd.read_csv("data.csv", dtype={"id": "string"}, parse_dates=["timestamp"])
df = pd.read_parquet("data.parquet")
df = pd.read_excel("data.xlsx", sheet_name="Sheet1")
df = pd.read_json("data.jsonl", lines=True)
```

For files larger than memory, use `chunksize` to iterate:

```python
for chunk in pd.read_csv("large.csv", chunksize=100_000):
    process(chunk)
```

### 2.2 HuggingFace Datasets

```python
# pip install datasets
from datasets import load_dataset

# Load a full dataset
ds = load_dataset("imdb")
ds["train"][0]                          # first example as a dict

# Stream without downloading
ds = load_dataset("c4", "en", split="train", streaming=True)
for example in ds.take(5):
    print(example)
```

The `datasets` library handles caching, splits, and conversion to PyTorch / TensorFlow / NumPy.

### 2.3 scikit-learn Built-Ins

```python
from sklearn.datasets import load_iris, fetch_california_housing, make_classification

iris = load_iris(as_frame=True)
X, y = iris.data, iris.target

housing = fetch_california_housing(as_frame=True)

# Synthetic data for prototyping
X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=10,
    n_classes=3, random_state=0,
)
```

Synthetic generators (`make_classification`, `make_regression`, `make_blobs`, `make_moons`) are useful for unit-testing pipelines without licensing concerns.

### 2.4 TorchVision and TFDS

```python
import torchvision

train = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True,
    transform=torchvision.transforms.ToTensor(),
)
```

```python
import tensorflow_datasets as tfds

train_ds, test_ds = tfds.load(
    "cifar10", split=["train", "test"], as_supervised=True,
)
```

Both libraries handle download, caching, and conversion to framework-native batch iterators.

---

## 3. Common File Formats

| Format | Extension | Strengths | Weaknesses |
|--------|-----------|-----------|------------|
| CSV | `.csv` | Universal, human-readable | Slow to parse, no types, no compression |
| Parquet | `.parquet` | Columnar, compressed, typed; fast for analytics | Binary; not human-inspectable |
| JSON Lines | `.jsonl` | One record per line; streamable | Verbose; slow to parse at scale |
| HDF5 | `.h5`, `.hdf5` | Nested, typed, supports random access | Binary; library dependency (`h5py`, `tables`) |
| Arrow / Feather | `.arrow`, `.feather` | Zero-copy in-memory format | Less ubiquitous than Parquet on disk |
| Image folders | `.jpg`, `.png` in directories | Native to image pipelines | Inefficient for many small files |
| TFRecord / WebDataset | `.tfrecord`, `.tar` | Sharded binary for high-throughput training | Pipeline-specific |

For tabular data above ~1 GB, prefer Parquet over CSV. For image datasets above ~10,000 files, prefer a sharded binary format over loose files.

### 3.1 CSV → Parquet Conversion

```python
df = pd.read_csv("data.csv")
df.to_parquet("data.parquet", compression="snappy")
```

A typical CSV-to-Parquet conversion reduces file size by 5–10× and read time by 10–30×.

---

## 4. Train / Validation / Test Splitting

A faithful split is the single most important guard against optimistic evaluation. The right strategy depends on the data's structure.

### 4.1 Random Split

Appropriate when samples are independent and identically distributed.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0,
)
```

### 4.2 Stratified Split

Preserves class proportions across splits — essential for imbalanced classification.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0,
)
```

For cross-validation, use `StratifiedKFold`:

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for train_idx, val_idx in skf.split(X, y):
    ...
```

### 4.3 Group-Aware Split

When samples share a grouping that should not span splits (e.g., the same patient, user, or session in train and test), use group-aware splitters.

```python
from sklearn.model_selection import GroupKFold

gkf = GroupKFold(n_splits=5)
for train_idx, val_idx in gkf.split(X, y, groups=patient_ids):
    ...
```

Failing to honour groups produces inflated scores that collapse on new groups.

### 4.4 Time-Aware Split

For time-series, the test set must be later than the training set. Random splitting leaks future information into training.

```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X):
    ...
```

`TimeSeriesSplit` produces expanding-window folds, where each training set includes all earlier folds. For non-overlapping forecast horizons or seasonality concerns, custom splitters are often required.

### 4.5 Choosing a Split Strategy

| Data structure | Split strategy |
|----------------|----------------|
| IID samples | Random or stratified |
| Imbalanced classes | Stratified |
| Repeated subjects | Group-aware (`GroupKFold`, `LeaveOneGroupOut`) |
| Temporal | `TimeSeriesSplit` or manual cutoff date |
| Hierarchical (e.g., samples within sites) | Group-aware on the outer level |

---

## 5. Sampling Strategies

### 5.1 Class-Imbalance Handling

| Strategy | Mechanism | When to use |
|----------|-----------|-------------|
| Class weights | Penalize errors on minority class more in the loss | First option; supported by most sklearn classifiers via `class_weight="balanced"` |
| Random undersampling | Drop majority-class samples | Large datasets where majority is dominant |
| Random oversampling | Duplicate minority-class samples | Small datasets; risk of overfitting to duplicates |
| SMOTE | Synthesize new minority samples by interpolation | Tabular numeric data; do not apply before splitting |
| Threshold tuning | Adjust the decision threshold post-training | Always available; often sufficient on its own |

### 5.2 SMOTE Example

```python
# pip install imbalanced-learn
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=0)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

Apply resampling only to the training set, never to validation or test sets.

### 5.3 Stratified Subsampling for Prototyping

When iterating on pipelines with a large dataset, work on a stratified subsample:

```python
from sklearn.model_selection import train_test_split

_, X_sample, _, y_sample = train_test_split(
    X, y, test_size=0.05, stratify=y, random_state=0,
)
```

A 5% stratified sample preserves class proportions and speeds iteration by 20×; final evaluation runs on the full data.

---

## 6. Licensing and Attribution

Every dataset has a license, even when it is not prominently displayed. Using a dataset outside its license is a legal and ethical issue regardless of whether the source is freely accessible.

### 6.1 Common Licenses

| License | Permits | Restrictions |
|---------|---------|--------------|
| CC0 / Public Domain | Any use | None — attribution courteous but not required |
| CC BY | Any use, including commercial | Attribution required |
| CC BY-SA | Any use, including commercial | Attribution + share-alike (derivatives same license) |
| CC BY-NC | Non-commercial use only | Attribution required; no commercial use |
| MIT / BSD / Apache | Any use | Attribution required (license text retained) |
| Custom / proprietary | Per terms | Read carefully; often restricts redistribution |

Datasets without a stated license should be treated as "all rights reserved" — not freely usable.

### 6.2 Citation

When publishing, cite both the dataset and the original collection paper if one exists. Most large repositories provide a recommended citation block; if not, a minimal citation includes title, source, version or access date, and URL.

---

## 7. Documenting a Dataset

### 7.1 Dataset Cards

A dataset card is a short structured document that accompanies a dataset and answers the questions a downstream user is likely to ask. HuggingFace and Google publish templates; the common sections are:

- **Description** — what the data represents
- **Source and collection** — how it was gathered and by whom
- **Composition** — schema, size, splits
- **Intended uses** — tasks the data is appropriate for
- **Out-of-scope uses** — tasks it is unsuitable for
- **Bias and limitations** — known skews or gaps
- **License and citation**

### 7.2 Datasheet Questions

The Gebru et al. *Datasheets for Datasets* framework proposes a more thorough audit. Key prompts:

- For what purpose was the dataset created?
- Who funded its creation?
- What does each instance represent?
- Are relationships between instances made explicit (groups, sequences)?
- Is there a sampling strategy? Is the sample representative?
- Were individuals informed about data collection? Could they consent?
- Has the dataset been used for any tasks already? Are there known errors?

A short dataset card is appropriate for any shared dataset; a full datasheet is appropriate for any dataset used in published research or production deployment.

---

## 8. Resources

- [HuggingFace Datasets documentation](https://huggingface.co/docs/datasets) — comprehensive guide to the `datasets` library.
- [scikit-learn dataset loading utilities](https://scikit-learn.org/stable/datasets.html) — built-in datasets and synthetic generators.
- [Pandas IO documentation](https://pandas.pydata.org/docs/user_guide/io.html) — every supported format and its options.
- [Apache Parquet format specification](https://parquet.apache.org/docs/) — for understanding columnar storage.
- [Gebru et al., *Datasheets for Datasets* (2021)](https://arxiv.org/abs/1803.09010) — the canonical framework for dataset documentation.
- [Mitchell et al., *Model Cards for Model Reporting* (2019)](https://arxiv.org/abs/1810.03993) — the model-side counterpart to dataset cards.
- [Creative Commons licenses](https://creativecommons.org/licenses/) — license terms in summary form.

---

[← Previous: Statistics for ML](09_STATISTICS_FOR_ML_GUIDE.md) | [Index](README.md) | [Next: Exploratory Data Analysis →](11_EDA_GUIDE.md)
