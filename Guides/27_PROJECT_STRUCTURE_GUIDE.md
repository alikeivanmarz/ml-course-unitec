# ML Project Structure

A consistent project layout reduces friction across collaborators, simplifies reproduction, and makes deployment less error-prone. This guide describes a standard structure for a self-contained machine learning repository, the conventions for each directory, and the tooling choices that surround them. The recommendations apply to projects ranging from a single experiment notebook to a full pipeline ready for production.

**Table of Contents**

1. [Recommended Directory Layout](#1-recommended-directory-layout)
2. [`src/` vs `notebooks/`](#2-src-vs-notebooks)
3. [Dependency Management](#3-dependency-management)
4. [Configuration](#4-configuration)
5. [Data Layout](#5-data-layout)
6. [Model Artefacts](#6-model-artefacts)
7. [README Anatomy](#7-readme-anatomy)
8. [Reproducibility Checklist](#8-reproducibility-checklist)
9. [Resources](#9-resources)

---

## 1. Recommended Directory Layout

```
project-name/
├── README.md
├── pyproject.toml          # or requirements.txt + setup.cfg
├── .gitignore
├── .python-version         # for pyenv / uv
├── .pre-commit-config.yaml # optional: lint/format hooks
│
├── data/
│   ├── raw/                # never modified; original source files
│   ├── interim/            # cleaned but not feature-engineered
│   └── processed/          # ready for modelling
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   └── 03_final_model.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data.py             # loaders, splitters
│   ├── features.py         # feature engineering
│   ├── models.py           # model definitions
│   ├── train.py            # training entry point
│   ├── evaluate.py         # evaluation script
│   └── predict.py          # inference function
│
├── models/                 # serialized model artefacts
│   └── <run-id>/
│       ├── model.joblib
│       ├── metrics.json
│       └── config.yaml
│
├── tests/
│   ├── test_data.py
│   └── test_features.py
│
├── configs/
│   ├── default.yaml
│   └── experiment_a.yaml
│
└── scripts/                # one-off utility scripts
    └── download_data.py
```

This layout is influenced by the Cookiecutter Data Science template. Not every project needs every directory; a small experiment may only have `notebooks/`, `data/`, and `src/`.

---

## 2. `src/` vs `notebooks/`

The two have distinct roles.

| | `notebooks/` | `src/` |
|---|---|---|
| Purpose | Exploration, narrative, presentation | Reusable, testable code |
| Audience | The author and reviewers | The author and any program that imports it |
| Lifetime | Often discarded after the experiment | Maintained alongside the project |
| Version control | Commit cleared outputs only | Standard code review |
| Testing | Difficult | Standard unit testing |

A common pattern: notebooks import from `src/`, never the reverse. As code stabilizes in a notebook cell, it is refactored into a function in `src/` and imported back.

```python
# notebooks/02_baseline.ipynb (cell)
from src.data import load_processed
from src.models import build_baseline

df = load_processed("dataset_v1")
model = build_baseline()
model.fit(df.drop("target", axis=1), df["target"])
```

### 2.1 Notebook Hygiene

| Practice | Reason |
|----------|--------|
| Strip outputs before commit | Smaller diffs; avoids accidental data leakage in commits |
| Restart-and-run-all before commit | Confirms reproducibility within the notebook |
| Keep notebooks chronological and numbered | Communicates progression without README explanation |
| Move repeated code into `src/` after the third paste | Standard "rule of three" |

`nbstripout` (a pre-commit hook) automates output stripping; `jupytext` enables Markdown- or Python-format synchronization for review-friendly diffs.

---

## 3. Dependency Management

Choose one tool per project; mixing tools causes lock conflicts.

| Tool | Best for | Pros | Cons |
|------|----------|------|------|
| `pip` + `requirements.txt` | Simple projects | Universal; no extra install | No transitive lock by default |
| `pip-tools` | Pip-native projects with locks | Compiles `.in` to `.txt`; clear separation of declared and locked deps | Manual recompile required |
| `conda` + `environment.yml` | Projects requiring non-Python binaries (CUDA, R, geospatial libs) | Handles compiled dependencies | Slower; large env sizes |
| `poetry` | Application-style projects | Dependency resolver, lockfile, virtualenv management in one | Added concept overhead |
| `uv` | New projects | Very fast; `pyproject.toml` native; compatible with pip-tools | Newer; smaller community |

### 3.1 Minimal `pyproject.toml`

```toml
[project]
name = "project-name"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "numpy>=1.26",
    "pandas>=2.2",
    "scikit-learn>=1.4",
]

[project.optional-dependencies]
dev = ["pytest", "ruff", "pre-commit"]
notebook = ["jupyter", "ipykernel"]
```

Optional-dependency groups (`dev`, `notebook`) keep the production install slim while supporting development workflows.

---

## 4. Configuration

Hard-coded paths and hyperparameters spread across notebooks become a maintenance burden. Centralize them.

### 4.1 YAML Configuration

```yaml
# configs/default.yaml
data:
  raw_path: data/raw/dataset.csv
  processed_path: data/processed/dataset.parquet
  test_size: 0.2
  random_state: 42

model:
  type: random_forest
  n_estimators: 200
  max_depth: 10

training:
  cv_folds: 5
  scoring: f1_macro
```

```python
import yaml
from pathlib import Path

with open("configs/default.yaml") as f:
    config = yaml.safe_load(f)

df = pd.read_csv(config["data"]["raw_path"])
```

### 4.2 Library Choices

| Library | Adds beyond raw YAML |
|---------|----------------------|
| `pydantic` | Type-safe config classes with validation |
| `hydra` | Compositional configs (mix base + overrides), CLI integration |
| `omegaconf` | Hydra's underlying object model; usable standalone |
| `dynaconf` | Environment-variable layering, secrets handling |

A single project rarely needs more than one of these. For experiments with many variants, Hydra's composition is widely used; for production services, Pydantic settings are usually sufficient.

### 4.3 Path Conventions

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
```

Anchoring paths to the project root (rather than the current working directory) makes scripts runnable from any location.

---

## 5. Data Layout

```
data/
├── raw/         # immutable; never modified
├── interim/     # intermediate transformations
└── processed/   # final, model-ready
```

### 5.1 Conventions

| Convention | Reason |
|------------|--------|
| Raw data is read-only | Allows full pipeline re-runs from a known starting point |
| All files in `data/` are gitignored except small reference files | Datasets pollute git history; hosting them in cloud storage is preferred |
| Filenames include a version or date | `train_2024_01.parquet`, not `train.parquet` |
| Provide a download script | `scripts/download_data.py`; document the source in README |
| Track data hashes (SHA-256) for reproducibility | Detect accidental corruption or replacement |

### 5.2 `.gitignore` for ML Projects

```gitignore
# Data
data/raw/
data/interim/
data/processed/
*.csv
*.parquet
*.h5
*.npz

# Models
models/
*.joblib
*.pkl
*.pt
*.keras
*.onnx

# Notebook artefacts
.ipynb_checkpoints/

# Python
__pycache__/
*.pyc
.venv/
.pytest_cache/

# Editor
.vscode/settings.json
.idea/

# Secrets
.env
*.key
```

For data and models, prefer "deny by default" (entire directories ignored) with explicit exceptions (`!data/raw/sample.csv`) for small reference files needed in the repo.

---

## 6. Model Artefacts

```
models/
└── 2026-04-15_random_forest_v3/
    ├── model.joblib          # the serialized model
    ├── config.yaml           # the config used for training
    ├── metrics.json          # held-out evaluation results
    ├── feature_names.json    # input column names
    └── git_sha.txt           # commit hash at training time
```

A run directory captures everything needed to reproduce, evaluate, or compare a trained model. Naming with `<date>_<algorithm>_<version>` orders chronologically and reads at a glance.

### 6.1 Versioning Tools

| Tool | Adds |
|------|------|
| Plain directory + git tag | Minimum viable; sufficient for solo work |
| MLflow | Experiment tracking, artefact storage, model registry |
| Weights & Biases | Experiment tracking with strong UI; cloud-hosted |
| DVC | Data and model versioning that integrates with git |
| Neptune | Experiment tracking; pricing-tier alternative |

For projects with more than ~10 model versions, an experiment-tracking tool replaces ad-hoc directories.

---

## 7. README Anatomy

A README is the single document every visitor reads first. The standard structure:

1. **One-line description** — what the project does, in one sentence.
2. **Status** — production / experimental / archived. Optionally a CI badge.
3. **Quickstart** — clone, install, run a single command. Should fit in 5 lines.
4. **Project structure** — annotated directory tree of the top level only.
5. **Data** — source, license, how to obtain. Do not commit large files.
6. **Reproducibility** — exact commands to reproduce results: training, evaluation, prediction.
7. **Results** — headline metrics in a table; brief comparison of model variants.
8. **License** — link to LICENSE file; state copyright holder.

A README is not a tutorial. Long tutorials belong in `docs/` or notebooks. The README answers: "what is this, how do I run it, where do I look next?"

---

## 8. Reproducibility Checklist

| Item | Method |
|------|--------|
| Pinned dependencies | Lockfile (`requirements.txt` from pip-compile, `poetry.lock`, `uv.lock`) |
| Python version pinned | `.python-version` or `requires-python` in `pyproject.toml` |
| Random seeds set | `np.random.seed`, `torch.manual_seed`, `random.seed`, framework-specific equivalents |
| Deterministic operations enabled | `torch.use_deterministic_algorithms(True)` where supported |
| Data version captured | Hash of input file recorded with each run |
| Code version captured | Git SHA written into the run directory |
| Hardware noted | CPU/GPU model recorded for benchmarks |
| Run command logged | The exact CLI invocation written into the run directory |

Reproducibility on the same machine is achievable with these steps. Cross-machine reproducibility for deep learning is harder — GPU non-determinism, CUDA version differences, and floating-point variation can cause small numerical drift even with seeds set.

---

## 9. Resources

- [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) — opinionated project template; influenced this guide.
- [Hidden Technical Debt in ML Systems (Sculley et al.)](https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html) — operational considerations beyond model code.
- [The Twelve-Factor App](https://12factor.net/) — application-architecture principles applicable to ML services.
- [DVC — Data Version Control](https://dvc.org/doc) — versioning data and models alongside git.
- [MLflow documentation](https://mlflow.org/docs/latest/index.html) — experiment tracking, model registry, and deployment.
- [Python Packaging User Guide — `pyproject.toml`](https://packaging.python.org/en/latest/specifications/pyproject-toml/) — declarative project configuration.
- [`nbstripout`](https://github.com/kynan/nbstripout) — pre-commit hook to strip notebook outputs.
