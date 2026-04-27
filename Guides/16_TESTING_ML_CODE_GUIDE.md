# Testing Machine Learning Code

Machine learning code includes data transformations, model definitions, training loops, and inference paths — all of which can fail silently in ways that produce plausible but incorrect output. Testing ML code requires patterns beyond standard software testing: shape contracts on tensors, properties of mathematical operations, regression checks on model outputs, and integration tests that exercise full pipelines. This guide covers what to test, how to structure tests with pytest, the patterns that catch the most common ML bugs, and how to integrate tests into a continuous-integration workflow.

**Table of Contents**

1. [What to Test in ML Code](#1-what-to-test-in-ml-code)
2. [pytest Basics for ML](#2-pytest-basics-for-ml)
3. [Testing Data Loaders and Transformations](#3-testing-data-loaders-and-transformations)
4. [Testing Model Contracts](#4-testing-model-contracts)
5. [Property-Based Testing](#5-property-based-testing)
6. [Snapshot and Regression Testing](#6-snapshot-and-regression-testing)
7. [Integration Tests for Pipelines](#7-integration-tests-for-pipelines)
8. [Test Organisation and Fixtures](#8-test-organisation-and-fixtures)
9. [Continuous Integration for ML](#9-continuous-integration-for-ml)
10. [Resources](#10-resources)

---

## 1. What to Test in ML Code

| Layer | Tests appropriate to it |
|-------|-------------------------|
| Data loading | Schema, dtypes, missing-value handling, file existence |
| Preprocessing | Shape preservation, no NaN injection, scaler statistics |
| Feature engineering | Output shape, output range, deterministic for same input |
| Model construction | Parameter count, output shape, gradient flow |
| Training step | Loss decreases on a tiny batch, no NaN/Inf in outputs |
| Inference | Output shape and dtype, batch invariance, deterministic given seed |
| Pipelines (end-to-end) | Train → save → load → predict produces expected types |

Performance metrics (accuracy, F1) are not unit tests — they are evaluation outputs that vary across runs. Tests should be deterministic and binary (pass/fail), not metric thresholds (which require tolerances and may flake on data updates).

---

## 2. pytest Basics for ML

`pytest` is the standard Python test runner. Tests are functions named `test_*` in files named `test_*.py`.

```python
# tests/test_features.py
import numpy as np
from src.features import standardize


def test_standardize_zero_mean_unit_variance():
    x = np.random.randn(100)
    y = standardize(x)
    assert np.allclose(y.mean(), 0.0, atol=1e-7)
    assert np.allclose(y.std(), 1.0, atol=1e-7)


def test_standardize_preserves_length():
    x = np.random.randn(50)
    assert len(standardize(x)) == 50
```

Run: `pytest tests/`. Useful flags:

| Flag | Effect |
|------|--------|
| `-x` | Stop on first failure |
| `-v` | Verbose: list each test |
| `-s` | Don't capture stdout |
| `-k pattern` | Run only tests matching pattern |
| `--lf` | Run only tests that failed last time |
| `--cov=src` | Measure code coverage (requires `pytest-cov`) |
| `--durations=10` | Print 10 slowest tests |

### 2.1 Floating-Point Comparisons

Direct equality on floats is unreliable. Use `np.allclose` (NumPy), `pytest.approx` (scalars), or `torch.allclose` (PyTorch tensors):

```python
import pytest

def test_close_enough():
    result = compute_something()
    assert result == pytest.approx(0.123, abs=1e-6)
    assert np.allclose(array, expected, rtol=1e-5, atol=1e-7)
```

`atol` is absolute tolerance; `rtol` is relative tolerance. For values near zero, `atol` matters; for large values, `rtol` matters.

---

## 3. Testing Data Loaders and Transformations

Data loaders are the most common source of silent ML bugs. Tests should verify schema, types, and value ranges.

```python
# tests/test_data.py
import pandas as pd
from src.data import load_processed


def test_loaded_data_has_expected_schema():
    df = load_processed()
    expected_cols = {"feature_a", "feature_b", "target"}
    assert set(df.columns) == expected_cols


def test_loaded_data_has_no_missing_target():
    df = load_processed()
    assert df["target"].notna().all()


def test_target_values_are_in_valid_range():
    df = load_processed()
    assert df["target"].between(0, 1).all()


def test_numeric_columns_have_expected_dtype():
    df = load_processed()
    assert df["feature_a"].dtype.kind in {"f", "i"}    # float or integer
```

### 3.1 Test Data Strategy

Tests should use a small, fixed test dataset that lives with the tests rather than the full production dataset.

| Test data approach | Use |
|--------------------|-----|
| Synthetic via `make_classification` / `make_regression` | Most unit tests; deterministic given seed |
| Fixed CSV in `tests/data/` | Schema tests; integration tests |
| Mock objects (`unittest.mock`) | Database/API loaders without hitting real services |
| Tiny subsample of real data | Smoke tests; not committed if data is sensitive |

A test that depends on real production data (large file, network access, credentials) is slow and brittle. Confine such tests to a separate `tests/integration/` directory and run them less frequently than the unit suite.

### 3.2 Pipeline Determinism

Preprocessing pipelines should be deterministic given the same input:

```python
def test_pipeline_is_deterministic():
    pipeline = make_pipeline()
    X1 = pipeline.fit_transform(X)
    X2 = pipeline.fit_transform(X)
    np.testing.assert_array_equal(X1, X2)
```

Non-determinism here typically signals either an unset random state or unintentional in-place mutation of inputs.

---

## 4. Testing Model Contracts

Model code can be tested without training a real model. The contract — input shape in, output shape out, parameter counts, gradient flow — is what unit tests target.

### 4.1 Output Shapes

```python
import torch
from src.models import build_classifier


def test_classifier_output_shape():
    model = build_classifier(input_dim=20, n_classes=3)
    x = torch.randn(8, 20)        # batch of 8
    out = model(x)
    assert out.shape == (8, 3)
```

### 4.2 Gradient Flow

A model with detached or frozen sub-modules will silently train sub-optimally. Test that gradients propagate to all parameters that should be trainable:

```python
def test_all_trainable_params_receive_gradient():
    model = build_classifier(input_dim=20, n_classes=3)
    x = torch.randn(4, 20)
    y = torch.tensor([0, 1, 2, 0])

    loss = torch.nn.functional.cross_entropy(model(x), y)
    loss.backward()

    for name, p in model.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"{name} did not receive a gradient"
            assert not torch.isnan(p.grad).any(), f"{name} has NaN gradient"
```

### 4.3 Loss Decrease on a Single Batch

A correctly implemented training step should reduce loss on a tiny dataset within a few iterations:

```python
def test_training_reduces_loss_on_overfit_batch():
    model = build_classifier(input_dim=20, n_classes=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = torch.randn(8, 20)
    y = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])

    losses = []
    for _ in range(50):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.1   # at least 10x reduction
```

This test catches broken training loops, wrong loss functions, frozen parameters, and exploding/vanishing gradients on a single tiny example.

---

## 5. Property-Based Testing

Property-based testing generates many random inputs that satisfy specified constraints, and asserts that an output property holds for all of them. The `hypothesis` library is the standard tool.

```python
# pip install hypothesis
from hypothesis import given, strategies as st
from hypothesis.extra.numpy import arrays
import numpy as np
from src.features import standardize


@given(arrays(dtype=np.float64,
              shape=st.integers(min_value=2, max_value=1000),
              elements=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False)))
def test_standardize_output_has_finite_values(x):
    if x.std() > 0:
        y = standardize(x)
        assert np.isfinite(y).all()
```

`hypothesis` will automatically generate hundreds of random arrays, including pathological cases (very small, very large, near-constant). It also *shrinks* failing inputs to the minimal failing example, accelerating debugging.

### 5.1 When Property-Based Testing Helps

| Code under test | Property worth checking |
|-----------------|--------------------------|
| Numerical transforms | No NaN/Inf; output bounds; idempotency for normalisation |
| Encoders | Round-trip equivalence (encode-then-decode = identity) |
| Tokenizers | Output token count bounded by input length; no empty outputs for non-empty inputs |
| Preprocessing pipelines | Deterministic; preserves row count |

Property-based testing adds genuine coverage that example-based tests miss — random inputs frequently uncover edge cases the author did not anticipate.

---

## 6. Snapshot and Regression Testing

Snapshot tests record an artefact's current value and assert that future runs produce the same value. They catch unintended changes in model output across code refactors.

### 6.1 Snapshotting Predictions

```python
# tests/test_predictions.py
import json
from pathlib import Path
import numpy as np

SNAPSHOT_PATH = Path(__file__).parent / "snapshots" / "predictions.json"


def test_predictions_match_snapshot():
    model = load_pretrained_model()
    inputs = load_canonical_test_inputs()
    predictions = model.predict(inputs).tolist()

    if not SNAPSHOT_PATH.exists():
        SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT_PATH.write_text(json.dumps(predictions))

    expected = json.loads(SNAPSHOT_PATH.read_text())
    np.testing.assert_allclose(predictions, expected, atol=1e-5)
```

Snapshot updates require a deliberate action (delete and regenerate the snapshot file). This makes intentional model changes visible in code review.

### 6.2 Snapshot vs Metric Test

| Test type | Catches | Misses |
|-----------|---------|--------|
| Snapshot of predictions | Any change in output | Whether the change is good or bad |
| Metric threshold (`accuracy > 0.85`) | Severe regressions | Small drops; minor numerical drift |

Both are useful at different times: snapshots in CI on every commit, metric thresholds before release.

---

## 7. Integration Tests for Pipelines

Integration tests exercise the full path from raw data to predictions. They are slower than unit tests but catch interaction bugs that unit tests miss.

```python
# tests/integration/test_full_pipeline.py
import tempfile
from pathlib import Path

from src.train import train
from src.predict import load_model_and_predict


def test_train_save_load_predict_round_trip():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.joblib"

        train(data_path="tests/data/tiny_train.csv", model_path=model_path)
        assert model_path.exists()

        predictions = load_model_and_predict(
            model_path=model_path,
            data_path="tests/data/tiny_test.csv",
        )
        assert len(predictions) == 5     # tiny_test has 5 rows
        assert all(p in {0, 1} for p in predictions)
```

The temp directory ensures the test does not pollute the working tree and runs hermetically.

---

## 8. Test Organisation and Fixtures

### 8.1 Directory Layout

```
tests/
├── conftest.py            # shared fixtures
├── unit/
│   ├── test_data.py
│   ├── test_features.py
│   └── test_models.py
├── integration/
│   └── test_pipeline.py
└── data/
    ├── tiny_train.csv
    └── tiny_test.csv
```

`conftest.py` is auto-discovered by pytest; fixtures defined there are available to all tests in the directory.

### 8.2 Fixtures

Fixtures are reusable test setup. They reduce duplication and make tests faster by computing expensive setup once per scope.

```python
# tests/conftest.py
import pytest
import numpy as np
from sklearn.datasets import make_classification


@pytest.fixture(scope="session")
def synthetic_dataset():
    X, y = make_classification(n_samples=200, n_features=10, random_state=0)
    return X, y


@pytest.fixture(scope="function")
def fresh_model():
    from src.models import build_classifier
    return build_classifier(input_dim=10, n_classes=2)
```

| Scope | Fixture rebuilt | Use |
|-------|-----------------|-----|
| `function` | Once per test | Default; isolates tests |
| `class` | Once per test class | Shared setup within class |
| `module` | Once per module | Expensive setup; tests don't mutate state |
| `session` | Once per pytest run | Most expensive setup; e.g., loading a large dataset |

### 8.3 Parametrisation

Parametrised tests run the same logic over multiple inputs:

```python
import pytest

@pytest.mark.parametrize("n_features", [10, 50, 100])
@pytest.mark.parametrize("n_classes", [2, 3, 5])
def test_classifier_handles_various_shapes(n_features, n_classes):
    model = build_classifier(input_dim=n_features, n_classes=n_classes)
    out = model(torch.randn(4, n_features))
    assert out.shape == (4, n_classes)
```

This generates 9 test cases (3 × 3) from a single test function.

---

## 9. Continuous Integration for ML

CI runs tests automatically on every commit, catching regressions before code is merged. A minimal GitHub Actions workflow for an ML project:

```yaml
# .github/workflows/test.yml
name: tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"
      - run: pip install -e ".[dev]"
      - run: ruff check .
      - run: pytest tests/unit --cov=src --cov-report=term-missing
      - run: pytest tests/integration -m "not slow"
```

### 9.1 What Belongs in CI

| Test type | Run on every commit | Run on schedule | Run before release |
|-----------|---------------------|-----------------|---------------------|
| Unit tests | Yes | — | Yes |
| Fast integration tests | Yes | — | Yes |
| Slow integration tests | No | Nightly | Yes |
| Full training runs | No | No | Yes (with reduced epochs) |
| Snapshot tests | Yes | — | Yes |
| Metric threshold tests | No | Nightly | Yes |

CI should complete within a few minutes for the change-blocking tests. Tests that take longer should run on a separate schedule or on-demand.

### 9.2 Markers for Test Selection

```python
# Mark slow tests
@pytest.mark.slow
def test_train_full_dataset():
    ...
```

Run only fast tests in CI: `pytest -m "not slow"`. Run only slow tests on a schedule: `pytest -m slow`.

---

## 10. Resources

- [pytest documentation](https://docs.pytest.org/) — official reference for test discovery, fixtures, parametrization.
- [`hypothesis`](https://hypothesis.readthedocs.io/) — property-based testing library.
- [`pytest-cov`](https://pytest-cov.readthedocs.io/) — coverage measurement.
- [`pytest-xdist`](https://pytest-xdist.readthedocs.io/) — parallel test execution.
- [Smith, *Software Engineering for Machine Learning: A Case Study* (2019)](https://arxiv.org/abs/1909.13076) — empirical study of ML testing practices in industry.
- [Beck, *Test-Driven Development: By Example* (2002)](https://www.pearson.com/store/p/test-driven-development-by-example/P100000388488) — foundational treatment of TDD; concepts apply to ML.
- [Google — *ML Test Score* (2017)](https://research.google/pubs/pub46555/) — rubric for ML system testing maturity.

---

[← Previous: ML Debugging](15_ML_DEBUGGING_GUIDE.md) | [Index](README.md) | [Next: Model Interpretability →](17_INTERPRETABILITY_GUIDE.md)
