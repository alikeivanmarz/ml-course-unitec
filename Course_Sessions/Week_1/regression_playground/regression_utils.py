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

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

RANDOM_STATE = 42
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
    """UCI Energy Efficiency (ENB2012), columns renamed to readable names."""
    df = pd.read_excel(DATASETS_DIR / "ENB2012_data.xlsx")
    return df.rename(columns=_ENB_NAMES)


ENERGY_FEATURES = ["Relative Compactness", "Surface Area", "Wall Area",
                   "Roof Area", "Overall Height", "Orientation",
                   "Glazing Area", "Glazing Area Distribution"]
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
    """MAPE in %, computed only over non-zero targets (returns nan if none)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.abs(y_true) > 1e-9
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


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


class MeanRegressor:
    """Predicts the training-set mean for every row (the R2 = 0 baseline)."""

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
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
        return LGBMRegressor(n_estimators=int(params.get("n_estimators", 300)),
                             max_depth=md if md else -1,
                             random_state=RANDOM_STATE, verbose=-1)
    if model == "XGBoost" and _HAS_XGB:
        return XGBRegressor(n_estimators=int(params.get("n_estimators", 300)),
                            max_depth=md if md else 6,
                            random_state=RANDOM_STATE, verbosity=0)
    raise ValueError(f"Unknown model: {model}")


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


def split_xy(df: pd.DataFrame, features: list[str], target: str,
             test_size: float = 0.2, seed: int = RANDOM_STATE):
    X = df[features].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)
    return train_test_split(X, y, test_size=test_size, random_state=seed)


def fit_and_evaluate(model: str, df: pd.DataFrame, features: list[str], target: str,
                     alpha: float = 1.0, degree: int = 2, scale: bool = True,
                     test_size: float = 0.2, seed: int = RANDOM_STATE, **params) -> dict:
    """Fit on the training split, report train and test metrics separately."""
    X_tr, X_te, y_tr, y_te = split_xy(df, features, target, test_size, seed)
    pipe = build_pipeline(model, alpha, degree, scale, **params).fit(X_tr, y_tr)
    return {
        "pipeline": pipe,
        "train": all_metrics(y_tr, pipe.predict(X_tr)),
        "test": all_metrics(y_te, pipe.predict(X_te)),
        "y_test": y_te,
        "pred_test": pipe.predict(X_te),
        "y_train": y_tr,
        "pred_train": pipe.predict(X_tr),
    }


def coefficients(pipe: Pipeline, feature_names: list[str]) -> pd.DataFrame:
    """Extract raw coefficients from a fitted linear pipeline (no poly expansion)."""
    est = pipe.named_steps.get("model")
    if not hasattr(est, "coef_"):
        return pd.DataFrame(columns=["feature", "coefficient"])
    coef = np.ravel(est.coef_)
    names = feature_names if len(feature_names) == len(coef) else \
        [f"f{i}" for i in range(len(coef))]
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
    if len(vals) != len(feature_names):  # e.g. polynomial expansion
        names = [f"term {i}" for i in range(len(vals))]
    else:
        names = feature_names
    df = pd.DataFrame({"feature": names, "importance": vals}).sort_values("importance")
    return df, kind


def predict_curve(pipe: Pipeline, x_min: float, x_max: float, n: int = 200):
    """Predict a smooth 1-D curve (single-feature models) for plotting the fit."""
    xs = np.linspace(x_min, x_max, n)
    return xs, pipe.predict(xs.reshape(-1, 1))


# ===========================================================================
# Cross-validation
# ===========================================================================
def cross_validate_r2(model: str, df: pd.DataFrame, features: list[str], target: str,
                      k: int = 5, alpha: float = 1.0, degree: int = 2,
                      scale: bool = True, seed: int = RANDOM_STATE) -> dict:
    """K-fold CV returning per-fold R2 plus mean and std."""
    X = df[features].to_numpy(dtype=float)
    y = df[target].to_numpy(dtype=float)
    pipe = build_pipeline(model, alpha, degree, scale)
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    scores = cross_val_score(pipe, X, y, cv=kf, scoring="r2")
    return {"folds": scores.tolist(), "mean": float(scores.mean()),
            "std": float(scores.std())}


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
        h = t0 + t1 * xs
        err = h - y
        cost = float(np.sum(err ** 2) / (2 * m))
        hist_t0.append(t0)
        hist_t1.append(t1)
        hist_cost.append(cost)
        grad0 = float(np.sum(err) / m)
        grad1 = float(np.sum(err * xs) / m)
        t0 -= alpha * grad0
        t1 -= alpha * grad1
    # Convert final standardized params back to raw-x space for plotting a line.
    slope_raw = t1 / x_sd
    intercept_raw = t0 - t1 * x_mu / x_sd
    cost_arr = np.array(hist_cost)
    # Diverged if the cost is non-finite OR grew instead of shrinking (a "too big"
    # learning rate can blow up to a huge-but-finite value, so check both).
    diverged = (not np.isfinite(cost_arr[-1])) or (cost_arr[-1] > cost_arr[0] * 1.0001)
    return {
        "theta0": np.array(hist_t0),
        "theta1": np.array(hist_t1),
        "cost": cost_arr,
        "slope_raw": slope_raw,
        "intercept_raw": intercept_raw,
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
                        top: int = 12) -> pd.DataFrame:
    """Rank every feature combination (size 1..max_k) by test R2 of a linear model."""
    X_all = df[features + [target]].dropna()
    results = []
    for k in range(1, max_k + 1):
        for combo in combinations(features, k):
            res = fit_and_evaluate("Linear", X_all, list(combo), target, seed=seed)
            results.append({
                "features": ", ".join(combo),
                "n_features": k,
                "test_R2": round(res["test"]["R2"], 4),
                "test_RMSE": round(res["test"]["RMSE"], 3),
            })
    out = pd.DataFrame(results).sort_values("test_R2", ascending=False).reset_index(drop=True)
    return out.head(top)
