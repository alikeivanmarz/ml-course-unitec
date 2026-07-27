"""
regression_utils.py
====================
Data, models, metrics and teaching maths for the Regression Playground.

Kept free of Streamlit so it is easy to read, reuse and unit-test. Everything
is plain numpy / pandas / scikit-learn.

Conventions match the lecture slides:
  * parameters are theta (theta0 = intercept, theta1 = slope)
  * learning rate is alpha
  * cost is J(theta) = (1 / 2m) * sum( (h - y)^2 )   -- the half-MSE form
Preprocessing is always fitted inside a Pipeline on the training split only,
so no test information leaks into training.
"""

from __future__ import annotations

from math import comb
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

RANDOM_STATE = 42
DEFAULT_MAX_POLYNOMIAL_TERMS = 5_000
REPO_ROOT = Path(__file__).resolve().parents[3]
DATASETS_DIR = REPO_ROOT / "Datasets"


# ===========================================================================
# Datasets
# ===========================================================================
def make_synthetic(n: int = 60, noise: float = 8.0, curvature: float = 0.0,
                   slope: float = 1.3, intercept: float = 2.0,
                   seed: int = RANDOM_STATE) -> pd.DataFrame:
    """A controllable one-feature sandbox: y = intercept + slope*x + curvature*x^2 + noise."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-5, 5, n)
    y = intercept + slope * x + curvature * (x ** 2) + rng.normal(0, noise, n)
    return pd.DataFrame({"x": x, "y": y})


GRADIENT_DESCENT_PATTERNS = {
    "Linear signal": "linear",
    "Clearly curved signal": "curved",
    "Very noisy linear signal": "noisy",
    "Linear signal with one outlier": "outlier",
    "Changing spread (funnel)": "funnel",
}

_GRADIENT_DESCENT_PATTERN_DESCRIPTIONS = {
    "linear": "A straight-line signal with constant random noise.",
    "curved": (
        "A strong quadratic component: gradient descent still finds the best "
        "straight line, but a straight line remains the wrong shape."
    ),
    "noisy": (
        "A straight-line signal with twice the selected noise, making the "
        "least-squares direction harder to see."
    ),
    "outlier": (
        "A straight-line signal with one central response outlier that pulls "
        "the least-squares fit."
    ),
    "funnel": (
        "A straight-line average with residual spread that increases from "
        "left to right."
    ),
}


def make_gradient_descent_data(pattern: str = "linear", n: int = 60,
                               noise: float = 8.0,
                               seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Return reproducible 1-D teaching data for the gradient-descent lab.

    ``noise`` is a standard-deviation control.  The very-noisy option doubles
    it, while the funnel option varies it smoothly across x.  ``is_special``
    marks the deliberately injected response outlier.
    """
    if isinstance(n, bool) or int(n) != n or n < 20:
        raise ValueError("n must be an integer of at least 20.")
    if not np.isfinite(noise) or noise < 0:
        raise ValueError("noise must be finite and non-negative.")

    lookup = {
        label.casefold(): key
        for label, key in GRADIENT_DESCENT_PATTERNS.items()
    }
    key = str(pattern).strip().casefold()
    key = lookup.get(key, key)
    if key not in _GRADIENT_DESCENT_PATTERN_DESCRIPTIONS:
        choices = ", ".join(_GRADIENT_DESCENT_PATTERN_DESCRIPTIONS)
        raise ValueError(f"Unknown gradient-descent pattern {pattern!r}. Use: {choices}.")

    n = int(n)
    noise = float(noise)
    rng = np.random.default_rng(seed)
    x = np.linspace(-5.0, 5.0, n)
    mean = 4.0 + 2.2 * x
    special = np.zeros(n, dtype=bool)

    if key == "curved":
        mean = mean + 1.15 * x ** 2
        y = mean + rng.normal(0.0, noise, n)
    elif key == "noisy":
        y = mean + rng.normal(0.0, 2.0 * noise, n)
    elif key == "outlier":
        y = mean + rng.normal(0.0, noise, n)
        idx = n // 2
        y[idx] += max(25.0, 4.0 * noise)
        special[idx] = True
    elif key == "funnel":
        relative_position = (x - x.min()) / np.ptp(x)
        error_sd = noise * (0.15 + 1.35 * relative_position)
        y = mean + rng.normal(0.0, error_sd, n)
    else:
        y = mean + rng.normal(0.0, noise, n)

    result = pd.DataFrame({
        "x": x,
        "y": np.asarray(y, dtype=float),
        "is_special": special,
    })
    result.attrs["pattern"] = key
    result.attrs["description"] = _GRADIENT_DESCENT_PATTERN_DESCRIPTIONS[key]
    return result


DIAGNOSTIC_SCENARIOS = {
    "clean": "Clean linear relationship",
    "curved": "Curved relationship",
    "funnel": "Funnel-shaped variance",
    "outlier": "Response outlier",
    "high-leverage": "High-leverage observation",
    "skewed": "Skewed residuals",
    "dependent": "Dependent errors",
}

_DIAGNOSTIC_ALIASES = {
    label.casefold(): key for key, label in DIAGNOSTIC_SCENARIOS.items()
}
_DIAGNOSTIC_ALIASES.update({
    "high leverage": "high-leverage",
    "high_leverage": "high-leverage",
    "non-normal": "skewed",
    "nonnormal": "skewed",
    "serial": "dependent",
})


def make_diagnostic_scenario(scenario: str = "clean", n: int = 120,
                             seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Create a reproducible dataset exhibiting one diagnostic visual clue.

    The frame contains ``x`` and ``y`` for modelling, ``observation`` for an
    order plot, and ``is_special`` to highlight an injected outlier/leverage
    point.  These are deliberately teaching examples, not automated assumption
    tests.
    """
    if n < 20:
        raise ValueError("Diagnostic scenarios need at least 20 observations.")
    key = str(scenario).strip().casefold()
    key = _DIAGNOSTIC_ALIASES.get(key, key)
    if key not in DIAGNOSTIC_SCENARIOS:
        choices = ", ".join(DIAGNOSTIC_SCENARIOS)
        raise ValueError(f"Unknown diagnostic scenario {scenario!r}. Use: {choices}.")

    rng = np.random.default_rng(seed)
    x = np.linspace(-4.0, 4.0, int(n))
    special = np.zeros(int(n), dtype=bool)
    base = 3.0 + 2.0 * x

    if key == "clean":
        error = rng.normal(0.0, 1.5, int(n))
        y = base + error
    elif key == "curved":
        error = rng.normal(0.0, 1.2, int(n))
        y = base + 1.15 * x ** 2 + error
    elif key == "funnel":
        error_sd = 0.35 + 0.55 * (x - x.min())
        y = base + rng.normal(0.0, error_sd, int(n))
    elif key == "outlier":
        y = base + rng.normal(0.0, 1.3, int(n))
        idx = int(n) // 2
        y[idx] += 24.0
        special[idx] = True
    elif key == "high-leverage":
        y = base + rng.normal(0.0, 1.3, int(n))
        idx = int(n) - 1
        x[idx] = 9.0
        y[idx] = -10.0
        special[idx] = True
    elif key == "skewed":
        # Centring preserves E(error) ~= 0 while the long right tail remains.
        error = rng.exponential(scale=1.8, size=int(n)) - 1.8
        y = base + error
    else:  # dependent
        x = rng.uniform(-4.0, 4.0, int(n))
        error = np.empty(int(n), dtype=float)
        innovations = rng.normal(0.0, 0.8, int(n))
        error[0] = innovations[0]
        for i in range(1, int(n)):
            error[i] = 0.88 * error[i - 1] + innovations[i]
        y = 3.0 + 2.0 * x + error

    result = pd.DataFrame({
        "x": x.astype(float),
        "y": np.asarray(y, dtype=float),
        "observation": np.arange(1, int(n) + 1),
        "is_special": special,
    })
    result.attrs["scenario"] = key
    result.attrs["label"] = DIAGNOSTIC_SCENARIOS[key]
    return result


# The four Anscombe datasets (identical summary stats, very different shapes).
_ANSCOMBE = {
    "x_123": [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
    "y1": [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68],
    "y2": [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74],
    "y3": [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73],
    "x4": [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8],
    "y4": [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89],
}


def make_anscombe(which: str = "I") -> pd.DataFrame:
    """Return one of Anscombe's four datasets as columns x, y."""
    if which == "IV":
        return pd.DataFrame({"x": _ANSCOMBE["x4"], "y": _ANSCOMBE["y4"]})
    key = {"I": "y1", "II": "y2", "III": "y3"}.get(which, "y1")
    return pd.DataFrame({"x": _ANSCOMBE["x_123"], "y": _ANSCOMBE[key]})


def load_fuel_consumption() -> pd.DataFrame:
    """FuelConsumption CO2 dataset (the Session 2 class-activity dataset)."""
    return pd.read_csv(DATASETS_DIR / "FuelConsumptionCo2.csv")


FUEL_FEATURES = ["ENGINESIZE", "CYLINDERS", "FUELCONSUMPTION_CITY",
                 "FUELCONSUMPTION_HWY", "FUELCONSUMPTION_COMB"]
FUEL_TARGET = "CO2EMISSIONS"

_ENB_NAMES = {
    "X1": "Relative Compactness", "X2": "Surface Area", "X3": "Wall Area",
    "X4": "Roof Area", "X5": "Overall Height", "X6": "Orientation",
    "X7": "Glazing Area", "X8": "Glazing Area Distribution",
    "Y1": "Heating Load", "Y2": "Cooling Load",
}


def load_energy_efficiency() -> pd.DataFrame:
    """Load ENB2012 with its categorical codes represented safely.

    ``Orientation`` and ``Glazing Area Distribution`` are category identifiers,
    not measurements: the numerical distance from code 2 to code 4 has no
    physical meaning.  They are therefore replaced by numeric one-hot columns.
    The first level is deliberately the reference category so an intercept-based
    linear model does not receive a perfectly redundant set of dummy variables.
    """
    df = pd.read_excel(DATASETS_DIR / "ENB2012_data.xlsx")
    df = df.rename(columns=_ENB_NAMES)

    # The local course files do not document real-world meanings for these
    # codes, so retain honest code labels rather than guessing compass/glazing
    # names.  The lowest observed code is the explicit reference category.
    orientation = pd.to_numeric(df.pop("Orientation"), errors="coerce")
    glazing_distribution = pd.to_numeric(
        df.pop("Glazing Area Distribution"), errors="coerce"
    )
    orientation_levels = [2, 3, 4, 5]
    glazing_levels = [0, 1, 2, 3, 4, 5]
    if (
        orientation.isna().any()
        or glazing_distribution.isna().any()
        or not orientation.isin(orientation_levels).all()
        or not glazing_distribution.isin(glazing_levels).all()
    ):
        raise ValueError("ENB2012 contains an unknown categorical code.")

    orientation = pd.Categorical(
        orientation.astype(int), categories=orientation_levels, ordered=False
    )
    glazing_distribution = pd.Categorical(
        glazing_distribution.astype(int), categories=glazing_levels,
        ordered=False,
    )
    encoded = pd.concat([
        pd.get_dummies(
            orientation, prefix="Orientation code", prefix_sep=" = ",
            drop_first=True, dtype=float,
        ),
        pd.get_dummies(
            glazing_distribution,
            prefix="Glazing distribution code", prefix_sep=" = ",
            drop_first=True, dtype=float,
        ),
    ], axis=1)
    return pd.concat([df, encoded], axis=1)


ENERGY_CONTINUOUS_FEATURES = [
    "Relative Compactness", "Surface Area", "Wall Area", "Roof Area",
    "Overall Height", "Glazing Area",
]
ENERGY_CATEGORICAL_FEATURES = [
    "Orientation code = 3", "Orientation code = 4", "Orientation code = 5",
    "Glazing distribution code = 1", "Glazing distribution code = 2",
    "Glazing distribution code = 3", "Glazing distribution code = 4",
    "Glazing distribution code = 5",
]
ENERGY_FEATURES = ENERGY_CONTINUOUS_FEATURES + ENERGY_CATEGORICAL_FEATURES
ENERGY_TARGET = "Heating Load"


def _load_wine():
    df = pd.read_csv(DATASETS_DIR / "winequality-red.csv", sep=";")
    return df, [c for c in df.columns if c != "quality"], "quality"


def _load_diabetes():
    from sklearn.datasets import load_diabetes
    d = load_diabetes(as_frame=True)
    return d.frame, list(d.feature_names), "target"


def _load_startup():
    df = pd.read_csv(DATASETS_DIR / "1000_Companies.csv")
    return df, ["R&D Spend", "Administration", "Marketing Spend"], "Profit"


def _load_students():
    df = pd.read_csv(DATASETS_DIR / "Students Social Media Addiction.csv")
    feats = ["Age", "Avg_Daily_Usage_Hours", "Sleep_Hours_Per_Night",
             "Mental_Health_Score", "Conflicts_Over_Social_Media"]
    return df, feats, "Addicted_Score"


def get_named_dataset(name: str):
    """Return (df, feature_names, target) for a named dataset."""
    if name == "FuelConsumption CO2":
        return load_fuel_consumption(), FUEL_FEATURES, FUEL_TARGET
    if name == "Energy Efficiency":
        return load_energy_efficiency(), ENERGY_FEATURES, ENERGY_TARGET
    if name == "Wine Quality (red)":
        return _load_wine()
    if name == "Diabetes":
        return _load_diabetes()
    if name == "Startup Profit":
        return _load_startup()
    if name == "Student wellbeing":
        return _load_students()
    if name == "Synthetic sandbox":
        df = make_synthetic(n=120, noise=12, curvature=0.4)
        return df, ["x"], "y"
    raise ValueError(f"Unknown dataset: {name}")


# Datasets with several numeric features (for Feature Lab).
MULTI_FEATURE_DATASETS = ["FuelConsumption CO2", "Energy Efficiency",
                          "Wine Quality (red)", "Diabetes", "Startup Profit",
                          "Student wellbeing"]
# Model Arena also offers the single-feature synthetic sandbox.
ARENA_DATASETS = MULTI_FEATURE_DATASETS + ["Synthetic sandbox"]


# ===========================================================================
# Metrics
# ===========================================================================
def safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return MAPE in percent only when the target makes it interpretable.

    MAPE is not meaningful when *any* target is zero or negative, and silently
    dropping those observations gives a deceptively favourable result.  Values
    extremely close to zero are rejected as well because they make percentage
    errors numerically explosive.  ``nan`` lets the UI display "N/A".
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if (
        y_true.size == 0
        or y_true.shape != y_pred.shape
        or not np.isfinite(y_true).all()
        or not np.isfinite(y_pred).all()
        or np.any(y_true <= 0)
    ):
        return float("nan")
    typical_size = max(float(np.median(np.abs(y_true))), 1.0)
    if float(np.min(y_true)) <= typical_size * 1e-9:
        return float("nan")
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def all_metrics(y_true, y_pred) -> dict:
    """Return the five slide metrics. R2 is allowed to go negative."""
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": safe_mape(y_true, y_pred),
    }


# ===========================================================================
# Models / pipelines  (leakage-safe: scaler fit on train only)
# ===========================================================================
# Linear family first (the Session 2 focus), then the more advanced models.
MODEL_CHOICES = ["Mean baseline", "Linear", "Polynomial", "Ridge", "Lasso",
                 "Decision Tree", "Random Forest", "SVM (RBF)", "K-Nearest Neighbours"]

# LightGBM and XGBoost are optional - only offer them if they import.
try:
    from lightgbm import LGBMRegressor

    class DataFrameSafeLGBMRegressor(LGBMRegressor):
        """Keep LightGBM's generated feature names consistent at prediction."""

        def predict(self, X, *args, **kwargs):
            if not hasattr(X, "columns") and hasattr(self, "feature_name_"):
                X = pd.DataFrame(np.asarray(X), columns=self.feature_name_)
            return super().predict(X, *args, **kwargs)

    MODEL_CHOICES.append("LightGBM")
    _HAS_LGBM = True
except Exception:  # pragma: no cover
    _HAS_LGBM = False
try:
    from xgboost import XGBRegressor
    MODEL_CHOICES.append("XGBoost")
    _HAS_XGB = True
except Exception:  # pragma: no cover
    _HAS_XGB = False

# Tree-based models do not need feature scaling.
_TREE_LIKE = {"Decision Tree", "Random Forest", "LightGBM", "XGBoost"}
LINEAR_MODELS = {"Linear", "Polynomial", "Ridge", "Lasso"}


class MeanRegressor(RegressorMixin, BaseEstimator):
    """Predicts the training-set mean for every row (the R2 = 0 baseline)."""

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def predict(self, X):
        return np.full(len(np.asarray(X)), self.mean_)


def _make_estimator(model: str, alpha: float, params: dict):
    md = params.get("max_depth")
    md = None if not md else int(md)  # 0 / None -> unlimited
    if model in ("Linear", "Polynomial"):
        return LinearRegression()
    if model == "Ridge":
        return Ridge(alpha=alpha)
    if model == "Lasso":
        return Lasso(alpha=alpha, max_iter=10000)
    if model == "Decision Tree":
        return DecisionTreeRegressor(max_depth=md, random_state=RANDOM_STATE)
    if model == "Random Forest":
        return RandomForestRegressor(n_estimators=int(params.get("n_estimators", 200)),
                                     max_depth=md, random_state=RANDOM_STATE, n_jobs=-1)
    if model == "SVM (RBF)":
        return SVR(C=float(params.get("C", 10.0)), gamma="scale")
    if model == "K-Nearest Neighbours":
        return KNeighborsRegressor(n_neighbors=int(params.get("n_neighbors", 5)))
    if model == "LightGBM" and _HAS_LGBM:
        return DataFrameSafeLGBMRegressor(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=md if md else -1,
            random_state=RANDOM_STATE,
            verbose=-1,
        )
    if model == "XGBoost" and _HAS_XGB:
        return XGBRegressor(n_estimators=int(params.get("n_estimators", 300)),
                            max_depth=md if md else 6,
                            random_state=RANDOM_STATE, verbosity=0)
    raise ValueError(f"Unknown model: {model}")


class PolynomialBudgetError(ValueError):
    """Raised before a polynomial expansion would create too many columns."""


def polynomial_term_count(n_features: int, degree: int,
                          include_bias: bool = False) -> int:
    """Return the exact number of output columns from PolynomialFeatures.

    For ``p`` inputs through degree ``d``, scikit-learn creates
    ``C(p + d, d)`` terms including the bias.  Counting first prevents a large
    dense design matrix from exhausting a student's laptop.
    """
    if isinstance(n_features, bool) or int(n_features) != n_features or n_features < 1:
        raise ValueError("n_features must be a positive integer.")
    if isinstance(degree, bool) or int(degree) != degree or degree < 1:
        raise ValueError("degree must be a positive integer.")
    count = comb(int(n_features) + int(degree), int(degree))
    return count if include_bias else count - 1


def polynomial_budget_status(n_features: int, degree: int,
                             n_samples: int | None = None,
                             max_terms: int = DEFAULT_MAX_POLYNOMIAL_TERMS) -> dict:
    """Describe the size and approximate dense-memory cost of an expansion."""
    if max_terms < 1:
        raise ValueError("max_terms must be positive.")
    n_terms = polynomial_term_count(n_features, degree, include_bias=False)
    estimated_mib = None
    if n_samples is not None:
        if n_samples < 0:
            raise ValueError("n_samples cannot be negative.")
        estimated_mib = float(n_samples * n_terms * 8 / (1024 ** 2))
    return {
        "n_features": int(n_features),
        "degree": int(degree),
        "n_terms": n_terms,
        "max_terms": int(max_terms),
        "within_budget": n_terms <= max_terms,
        "estimated_matrix_mib": estimated_mib,
    }


def check_polynomial_budget(n_features: int, degree: int,
                            n_samples: int | None = None,
                            max_terms: int = DEFAULT_MAX_POLYNOMIAL_TERMS) -> int:
    """Return the term count, or raise a student-readable safety error."""
    status = polynomial_budget_status(
        n_features, degree, n_samples=n_samples, max_terms=max_terms
    )
    if not status["within_budget"]:
        memory = ""
        if status["estimated_matrix_mib"] is not None:
            memory = (
                f" (about {status['estimated_matrix_mib']:.0f} MiB for one "
                "dense matrix)"
            )
        raise PolynomialBudgetError(
            f"{n_features} features at degree {degree} create "
            f"{status['n_terms']:,} polynomial terms{memory}; the classroom "
            f"safety limit is {max_terms:,}. Choose fewer features or a "
            "lower degree."
        )
    return int(status["n_terms"])


def build_pipeline(model: str, alpha: float = 1.0, degree: int = 2,
                   scale: bool = True, **params) -> Pipeline:
    """Assemble a leakage-safe pipeline for the chosen model."""
    if model == "Mean baseline":
        return Pipeline([("model", MeanRegressor())])
    steps = []
    if model == "Polynomial":
        steps.append(("poly", PolynomialFeatures(degree=degree, include_bias=False)))
    # Scaling helps linear / SVM / KNN; it is unnecessary for tree models.
    if scale and model not in _TREE_LIKE:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", _make_estimator(model, alpha, params)))
    return Pipeline(steps)


def make_locked_split(df: pd.DataFrame, features: list[str], target: str,
                      test_size: float = 0.2,
                      seed: int = RANDOM_STATE) -> dict[str, Any]:
    """Create one reproducible train/final-test partition and retain its rows.

    The returned object can be reused while students compare features and tune
    hyperparameters.  Those choices should use cross-validation on
    ``train_df``; ``test_df`` remains untouched until the final choice is
    evaluated.  Original index labels and positional indices are included so a
    chart can draw training and final-test observations with different markers.
    """
    features = list(dict.fromkeys(features))
    if not features:
        raise ValueError("Choose at least one feature.")
    missing = [c for c in features + [target] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {', '.join(missing)}")
    if not 0 < float(test_size) < 1:
        raise ValueError("test_size must be between 0 and 1.")

    model_df = df.loc[:, features + [target]].replace([np.inf, -np.inf], np.nan)
    model_df = model_df.dropna().copy()
    if len(model_df) < 3:
        raise ValueError("At least three complete rows are required.")
    # Check numerical compatibility here, before scikit-learn emits an obscure
    # conversion error several interactions later.
    model_df.loc[:, features + [target]] = model_df[
        features + [target]
    ].apply(pd.to_numeric, errors="raise")

    positions = np.arange(len(model_df))
    train_pos, test_pos = train_test_split(
        positions, test_size=test_size, random_state=seed
    )
    train_df = model_df.iloc[train_pos].copy()
    test_df = model_df.iloc[test_pos].copy()
    return {
        "features": features,
        "target": target,
        "test_size": float(test_size),
        "seed": int(seed),
        "train_df": train_df,
        "test_df": test_df,
        "X_train": train_df[features].to_numpy(dtype=float),
        "X_test": test_df[features].to_numpy(dtype=float),
        "y_train": train_df[target].to_numpy(dtype=float),
        "y_test": test_df[target].to_numpy(dtype=float),
        "train_indices": train_df.index.to_numpy(copy=True),
        "test_indices": test_df.index.to_numpy(copy=True),
        "train_positions": np.asarray(train_pos, dtype=int),
        "test_positions": np.asarray(test_pos, dtype=int),
    }


def _locked_arrays(locked_split: dict[str, Any], features: list[str],
                   target: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract a feature subset from a locked split without re-splitting it."""
    if "train_df" in locked_split and "test_df" in locked_split:
        train_df = locked_split["train_df"]
        test_df = locked_split["test_df"]
        missing = [
            c for c in features + [target]
            if c not in train_df.columns or c not in test_df.columns
        ]
        if missing:
            raise KeyError(
                "Locked split does not contain: " + ", ".join(dict.fromkeys(missing))
            )
        return (
            train_df[features].to_numpy(dtype=float),
            test_df[features].to_numpy(dtype=float),
            train_df[target].to_numpy(dtype=float),
            test_df[target].to_numpy(dtype=float),
        )
    if list(locked_split.get("features", [])) != list(features):
        raise ValueError(
            "This locked split has no row frames, so its feature order must "
            "match exactly."
        )
    return (
        np.asarray(locked_split["X_train"], dtype=float),
        np.asarray(locked_split["X_test"], dtype=float),
        np.asarray(locked_split["y_train"], dtype=float),
        np.asarray(locked_split["y_test"], dtype=float),
    )


def split_xy(df: pd.DataFrame, features: list[str], target: str,
             test_size: float = 0.2, seed: int = RANDOM_STATE):
    """Backward-compatible tuple view of :func:`make_locked_split`."""
    split = make_locked_split(df, features, target, test_size, seed)
    return split["X_train"], split["X_test"], split["y_train"], split["y_test"]


def fit_and_evaluate(model: str, df: pd.DataFrame, features: list[str], target: str,
                     alpha: float = 1.0, degree: int = 2, scale: bool = True,
                     test_size: float = 0.2, seed: int = RANDOM_STATE,
                     locked_split: dict[str, Any] | None = None,
                     max_polynomial_terms: int = DEFAULT_MAX_POLYNOMIAL_TERMS,
                     **params) -> dict:
    """Fit once on a locked training split and evaluate the untouched test rows."""
    if locked_split is None:
        locked_split = make_locked_split(df, features, target, test_size, seed)
    X_tr, X_te, y_tr, y_te = _locked_arrays(locked_split, features, target)
    if model == "Polynomial":
        check_polynomial_budget(
            len(features), degree, n_samples=len(X_tr),
            max_terms=max_polynomial_terms,
        )
    pipe = build_pipeline(model, alpha, degree, scale, **params).fit(X_tr, y_tr)
    pred_train = pipe.predict(X_tr)
    pred_test = pipe.predict(X_te)
    return {
        "pipeline": pipe,
        "train": all_metrics(y_tr, pred_train),
        "test": all_metrics(y_te, pred_test),
        "y_test": y_te,
        "pred_test": pred_test,
        "y_train": y_tr,
        "pred_train": pred_train,
        "X_train": X_tr,
        "X_test": X_te,
        "train_indices": np.asarray(locked_split.get("train_indices", [])),
        "test_indices": np.asarray(locked_split.get("test_indices", [])),
        "locked_split": locked_split,
    }


def expanded_feature_names(pipe: Pipeline, feature_names: list[str]) -> list[str]:
    """Return readable names, including powers/interactions from a poly step."""
    poly = pipe.named_steps.get("poly")
    if poly is not None and hasattr(poly, "get_feature_names_out"):
        return list(poly.get_feature_names_out(feature_names))
    return list(feature_names)


def coefficients(pipe: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    """Extract coefficients with meaningful original or polynomial term names."""
    est = pipe.named_steps.get("model")
    if not hasattr(est, "coef_"):
        return pd.DataFrame(columns=["feature", "coefficient"])
    coef = np.ravel(est.coef_)
    names = expanded_feature_names(pipe, feature_names)
    if len(names) != len(coef):
        names = [f"term {i + 1}" for i in range(len(coef))]
    return pd.DataFrame({"feature": names, "coefficient": coef})


def feature_importance(pipe: Pipeline, feature_names: list[str]):
    """Return (DataFrame[feature, importance], kind) for any fitted model.

    Uses |coefficient| for linear models and feature_importances_ for tree models.
    kind is None when the model exposes neither (e.g. SVM, KNN, mean baseline).
    """
    est = pipe.named_steps.get("model")
    if hasattr(est, "coef_"):
        vals = np.abs(np.ravel(est.coef_))
        kind = "|coefficient|"
    elif hasattr(est, "feature_importances_"):
        vals = np.ravel(est.feature_importances_)
        kind = "importance"
    else:
        return pd.DataFrame(columns=["feature", "importance"]), None
    names = expanded_feature_names(pipe, feature_names)
    if len(vals) != len(names):
        names = [f"term {i + 1}" for i in range(len(vals))]
    df = pd.DataFrame({"feature": names, "importance": vals}).sort_values("importance")
    return df, kind


def predict_curve(pipe: Pipeline, x_min: float, x_max: float, n: int = 200):
    """Predict a smooth 1-D curve (single-feature models) for plotting the fit."""
    xs = np.linspace(x_min, x_max, n)
    return xs, pipe.predict(xs.reshape(-1, 1))


# ===========================================================================
# Cross-validation
# ===========================================================================
def _cross_validate_arrays(pipe: Pipeline, X: np.ndarray, y: np.ndarray,
                           k: int, seed: int) -> dict:
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if k < 2:
        raise ValueError("k must be at least 2.")
    if k > len(y):
        raise ValueError(f"k={k} exceeds the {len(y)} available training rows.")
    if len(y) // k < 2:
        raise ValueError(
            "Each validation fold needs at least two observations to define R²."
        )
    kf = KFold(n_splits=int(k), shuffle=True, random_state=seed)
    scores = cross_validate(
        pipe,
        X,
        y,
        cv=kf,
        scoring={"R2": "r2", "RMSE": "neg_root_mean_squared_error"},
        return_train_score=False,
        error_score="raise",
    )
    r2_folds = np.asarray(scores["test_R2"], dtype=float)
    rmse_folds = -np.asarray(scores["test_RMSE"], dtype=float)
    return {
        "R2": {
            "folds": r2_folds.tolist(),
            "mean": float(np.mean(r2_folds)),
            "std": float(np.std(r2_folds)),
        },
        "RMSE": {
            "folds": rmse_folds.tolist(),
            "mean": float(np.mean(rmse_folds)),
            "std": float(np.std(rmse_folds)),
        },
        "fold_results": pd.DataFrame({
            "fold": np.arange(1, int(k) + 1),
            "R2": r2_folds,
            "RMSE": rmse_folds,
        }),
        "n_splits": int(k),
        "n_samples": int(len(y)),
    }


def cross_validate_metrics(model: str, df: pd.DataFrame,
                           features: list[str], target: str, k: int = 5,
                           alpha: float = 1.0, degree: int = 2,
                           scale: bool = True, seed: int = RANDOM_STATE,
                           test_size: float = 0.2,
                           locked_split: dict[str, Any] | None = None,
                           training_only: bool = True,
                           max_polynomial_terms: int = DEFAULT_MAX_POLYNOMIAL_TERMS,
                           **params) -> dict:
    """Return real fold-level R² and RMSE summaries.

    By default a final test partition is locked away and CV uses only the
    training partition.  Pass the same ``locked_split`` to multiple calls while
    tuning a model so every candidate sees identical folds and the final test
    rows remain unseen.  Set ``training_only=False`` only when no final test
    evaluation is planned.
    """
    if locked_split is None and training_only:
        locked_split = make_locked_split(df, features, target, test_size, seed)
    if locked_split is not None:
        X, _, y, _ = _locked_arrays(locked_split, features, target)
        scope = "locked training partition"
    else:
        clean = df.loc[:, features + [target]].replace(
            [np.inf, -np.inf], np.nan
        ).dropna()
        X = clean[features].to_numpy(dtype=float)
        y = clean[target].to_numpy(dtype=float)
        scope = "all supplied rows"
    if model == "Polynomial":
        check_polynomial_budget(
            len(features), degree, n_samples=len(X),
            max_terms=max_polynomial_terms,
        )
    pipe = build_pipeline(model, alpha, degree, scale, **params)
    result = _cross_validate_arrays(pipe, X, y, k, seed)
    result.update({
        "scope": scope,
        "locked_split": locked_split,
    })
    return result


def cross_validate_r2(model: str, df: pd.DataFrame, features: list[str], target: str,
                      k: int = 5, alpha: float = 1.0, degree: int = 2,
                      scale: bool = True, seed: int = RANDOM_STATE,
                      test_size: float = 0.2,
                      locked_split: dict[str, Any] | None = None,
                      training_only: bool = True,
                      **params) -> dict:
    """Backward-compatible R²-only view of :func:`cross_validate_metrics`."""
    result = cross_validate_metrics(
        model, df, features, target, k=k, alpha=alpha, degree=degree,
        scale=scale, seed=seed, test_size=test_size,
        locked_split=locked_split, training_only=training_only, **params,
    )
    return {
        **result["R2"],
        "RMSE": result["RMSE"],
        "fold_results": result["fold_results"],
        "scope": result["scope"],
        "locked_split": result["locked_split"],
    }


def degree_cv_sweep(df: pd.DataFrame, features: list[str], target: str,
                    degrees: Iterable[int] = range(1, 13), k: int = 5,
                    seed: int = RANDOM_STATE, test_size: float = 0.2,
                    locked_split: dict[str, Any] | None = None,
                    max_polynomial_terms: int = DEFAULT_MAX_POLYNOMIAL_TERMS
                    ) -> pd.DataFrame:
    """Compare polynomial degrees using CV on one locked training partition."""
    if locked_split is None:
        locked_split = make_locked_split(df, features, target, test_size, seed)
    X_train, _, y_train, _ = _locked_arrays(locked_split, features, target)
    rows = []
    for value in degrees:
        degree = int(value)
        if degree != value or degree < 1:
            raise ValueError("Every degree must be a positive integer.")
        n_terms = check_polynomial_budget(
            len(features), degree, n_samples=len(X_train),
            max_terms=max_polynomial_terms,
        )
        cv = _cross_validate_arrays(
            build_pipeline("Polynomial", degree=degree, scale=True),
            X_train, y_train, k, seed,
        )
        fitted = build_pipeline(
            "Polynomial", degree=degree, scale=True
        ).fit(X_train, y_train)
        train_metrics = all_metrics(y_train, fitted.predict(X_train))
        rows.append({
            "degree": degree,
            "n_terms": n_terms,
            "train_R2": train_metrics["R2"],
            "train_RMSE": train_metrics["RMSE"],
            "CV_R2_mean": cv["R2"]["mean"],
            "CV_R2_std": cv["R2"]["std"],
            "CV_RMSE_mean": cv["RMSE"]["mean"],
            "CV_RMSE_std": cv["RMSE"]["std"],
        })
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("Provide at least one degree.")
    result["is_best"] = (
        result["CV_RMSE_mean"] == result["CV_RMSE_mean"].min()
    )
    result.attrs["locked_split"] = locked_split
    return result


def alpha_cv_sweep(model: str, df: pd.DataFrame, features: list[str],
                   target: str,
                   alphas: Iterable[float] = np.logspace(-3, 3, 25),
                   k: int = 5, seed: int = RANDOM_STATE,
                   test_size: float = 0.2,
                   locked_split: dict[str, Any] | None = None
                   ) -> pd.DataFrame:
    """Compare Ridge/Lasso alpha values using training-only CV."""
    if model not in {"Ridge", "Lasso"}:
        raise ValueError("alpha_cv_sweep supports Ridge or Lasso.")
    if locked_split is None:
        locked_split = make_locked_split(df, features, target, test_size, seed)
    X_train, _, y_train, _ = _locked_arrays(locked_split, features, target)
    rows = []
    for value in alphas:
        alpha = float(value)
        if not np.isfinite(alpha) or alpha < 0:
            raise ValueError("Every alpha must be finite and non-negative.")
        cv = _cross_validate_arrays(
            build_pipeline(model, alpha=alpha, scale=True),
            X_train, y_train, k, seed,
        )
        rows.append({
            "alpha": alpha,
            "CV_R2_mean": cv["R2"]["mean"],
            "CV_R2_std": cv["R2"]["std"],
            "CV_RMSE_mean": cv["RMSE"]["mean"],
            "CV_RMSE_std": cv["RMSE"]["std"],
        })
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("Provide at least one alpha.")
    result["is_best"] = (
        result["CV_RMSE_mean"] == result["CV_RMSE_mean"].min()
    )
    result.attrs["locked_split"] = locked_split
    return result


# ===========================================================================
# Gradient descent (single feature) — matches the slide notation
# ===========================================================================
def gradient_descent_1d(x: np.ndarray, y: np.ndarray, alpha: float = 0.05,
                        n_iters: int = 200, theta0: float = 0.0,
                        theta1: float = 0.0, standardize: bool = True) -> dict:
    """
    Batch gradient descent on y ~ theta0 + theta1 * x.

    Cost J = (1/2m) * sum((h - y)^2). Returns the full history so the app can
    animate the line rotating into place and the cost curve dropping.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or len(x) < 2:
        raise ValueError("x and y must be same-length 1-D arrays with at least two rows.")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("x and y must contain only finite values.")
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError("alpha must be finite and non-negative.")
    if isinstance(n_iters, bool) or int(n_iters) != n_iters or n_iters < 0:
        raise ValueError("n_iters must be a non-negative integer.")
    # Standardise x for numerically stable, well-scaled steps (recommended).
    if standardize:
        x_mu, x_sd = x.mean(), x.std() or 1.0
        xs = (x - x_mu) / x_sd
    else:
        x_mu, x_sd, xs = 0.0, 1.0, x
    m = len(xs)
    t0, t1 = float(theta0), float(theta1)
    hist_t0, hist_t1, hist_cost = [], [], []
    for _ in range(int(n_iters) + 1):
        with np.errstate(over="ignore", invalid="ignore"):
            h = t0 + t1 * xs
            err = h - y
            cost = float(np.sum(err ** 2) / (2 * m))
        hist_t0.append(t0)
        hist_t1.append(t1)
        hist_cost.append(cost)
        with np.errstate(over="ignore", invalid="ignore"):
            grad0 = float(np.sum(err) / m)
            grad1 = float(np.sum(err * xs) / m)
        t0 -= alpha * grad0
        t1 -= alpha * grad1
    theta0_arr = np.asarray(hist_t0, dtype=float)
    theta1_arr = np.asarray(hist_t1, dtype=float)
    # Every recorded standardized parameter pair has an exactly equivalent
    # raw-x equation for plotting and interpretation.
    slope_raw_history = theta1_arr / x_sd
    intercept_raw_history = theta0_arr - theta1_arr * x_mu / x_sd
    cost_arr = np.array(hist_cost)
    # Diverged if the cost is non-finite OR grew instead of shrinking (a "too big"
    # learning rate can blow up to a huge-but-finite value, so check both).
    diverged = (not np.isfinite(cost_arr[-1])) or (cost_arr[-1] > cost_arr[0] * 1.0001)
    return {
        "theta0": theta0_arr,
        "theta1": theta1_arr,
        "cost": cost_arr,
        "slope_raw_history": slope_raw_history,
        "intercept_raw_history": intercept_raw_history,
        "slope_raw": float(slope_raw_history[-1]),
        "intercept_raw": float(intercept_raw_history[-1]),
        "x_mu": x_mu,
        "x_sd": x_sd,
        "diverged": bool(diverged),
    }


def ols_1d(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Closed-form ordinary least squares line: returns (slope, intercept)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def sse_mse(x, y, slope, intercept) -> tuple[float, float]:
    """Sum of squared errors and mean squared error for a given line."""
    pred = intercept + slope * np.asarray(x, dtype=float)
    err = np.asarray(y, dtype=float) - pred
    sse = float(np.sum(err ** 2))
    return sse, sse / len(err)


# ===========================================================================
# Regularisation coefficient paths
# ===========================================================================
def coefficient_path(df: pd.DataFrame, features: list[str], target: str,
                     model: str = "Ridge", alphas: np.ndarray | None = None,
                     seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Coefficient value for each feature across a range of alphas (scaled)."""
    if alphas is None:
        alphas = np.logspace(-2, 3, 30)
    X_tr, _, y_tr, _ = split_xy(df, features, target, seed=seed)
    rows = []
    for a in alphas:
        pipe = build_pipeline(model, alpha=float(a), scale=True).fit(X_tr, y_tr)
        coef = np.ravel(pipe.named_steps["model"].coef_)
        for name, c in zip(features, coef):
            rows.append({"alpha": float(a), "feature": name, "coefficient": float(c)})
    return pd.DataFrame(rows)


def n_zeroed(df: pd.DataFrame, features: list[str], target: str, alpha: float,
             model: str = "Lasso", seed: int = RANDOM_STATE) -> int:
    """How many coefficients a model drives to (near) zero at this alpha."""
    X_tr, _, y_tr, _ = split_xy(df, features, target, seed=seed)
    pipe = build_pipeline(model, alpha=alpha, scale=True).fit(X_tr, y_tr)
    coef = np.ravel(pipe.named_steps["model"].coef_)
    return int(np.sum(np.abs(coef) < 1e-6))


# ===========================================================================
# Best feature-combination search (Feature Lab "reveal ranking")
# ===========================================================================
def rank_feature_combos(df: pd.DataFrame, features: list[str], target: str,
                        max_k: int = 3, seed: int = RANDOM_STATE,
                        top: int = 12, k: int = 5,
                        test_size: float = 0.2,
                        locked_split: dict[str, Any] | None = None
                        ) -> pd.DataFrame:
    """Rank feature sets by CV on training rows; never peek at final test rows."""
    if max_k < 1:
        raise ValueError("max_k must be at least 1.")
    if locked_split is None:
        # Split using every candidate column once so all combinations see the
        # same training rows and the same final test rows remain locked away.
        locked_split = make_locked_split(df, features, target, test_size, seed)
    train_df = locked_split["train_df"]
    results = []
    for n_features in range(1, min(max_k, len(features)) + 1):
        for combo in combinations(features, n_features):
            X_train = train_df[list(combo)].to_numpy(dtype=float)
            y_train = train_df[target].to_numpy(dtype=float)
            cv = _cross_validate_arrays(
                build_pipeline("Linear", scale=True),
                X_train, y_train, k, seed,
            )
            results.append({
                "features": ", ".join(combo),
                "n_features": n_features,
                "CV_R2_mean": cv["R2"]["mean"],
                "CV_R2_std": cv["R2"]["std"],
                "CV_RMSE_mean": cv["RMSE"]["mean"],
                "CV_RMSE_std": cv["RMSE"]["std"],
            })
    out = pd.DataFrame(results).sort_values(
        ["CV_R2_mean", "CV_RMSE_mean"], ascending=[False, True]
    ).reset_index(drop=True)
    out = out.head(top).copy()
    out.attrs["locked_split"] = locked_split
    out.attrs["selection_scope"] = "cross-validation on locked training partition"
    return out
