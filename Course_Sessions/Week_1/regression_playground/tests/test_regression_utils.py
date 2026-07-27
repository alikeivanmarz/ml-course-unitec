"""Correctness tests for the Regression Playground's modelling utilities."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold


PLAYGROUND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLAYGROUND_DIR))

import regression_utils as ru  # noqa: E402


def _three_feature_data(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    x3 = rng.normal(size=n)
    y = 8 * x1 + 0.2 * x2 + rng.normal(scale=0.25, size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3, "y": y})


def test_safe_mape_is_conservative_about_zero_and_negative_targets():
    assert ru.safe_mape([10, 20], [9, 18]) == pytest.approx(10.0)
    assert np.isnan(ru.safe_mape([0, 20], [1, 18]))
    assert np.isnan(ru.safe_mape([-10, 20], [-9, 18]))
    assert np.isnan(ru.safe_mape([1e-12, 20], [1, 18]))


def test_energy_categories_are_one_hot_numeric_not_ordinal_codes():
    df = ru.load_energy_efficiency()
    assert "Orientation" not in df.columns
    assert "Glazing Area Distribution" not in df.columns
    assert {
        "Orientation code = 3",
        "Orientation code = 4",
        "Orientation code = 5",
        "Glazing distribution code = 1",
        "Glazing distribution code = 2",
        "Glazing distribution code = 3",
        "Glazing distribution code = 4",
        "Glazing distribution code = 5",
    }.issubset(df.columns)
    assert all(pd.api.types.is_numeric_dtype(df[c]) for c in ru.ENERGY_FEATURES)
    assert df[ru.ENERGY_FEATURES].isna().sum().sum() == 0


def test_locked_split_retains_disjoint_indices_and_model_data():
    df = _three_feature_data()
    split = ru.make_locked_split(df, ["x1", "x2", "x3"], "y", test_size=0.25)
    assert len(split["train_indices"]) == 75
    assert len(split["test_indices"]) == 25
    assert set(split["train_indices"]).isdisjoint(split["test_indices"])
    assert set(split["train_indices"]) | set(split["test_indices"]) == set(df.index)

    result = ru.fit_and_evaluate(
        "Linear", df, ["x1"], "y", locked_split=split
    )
    np.testing.assert_array_equal(result["train_indices"], split["train_indices"])
    np.testing.assert_array_equal(result["test_indices"], split["test_indices"])
    assert result["X_train"].shape == (75, 1)
    assert result["X_test"].shape == (25, 1)


def test_cross_validate_metrics_returns_real_positive_fold_rmse():
    df = _three_feature_data()
    split = ru.make_locked_split(df, ["x1", "x2", "x3"], "y")
    result = ru.cross_validate_metrics(
        "Linear", df, ["x1", "x2"], "y", k=5, locked_split=split
    )

    # Independently calculate the same fold RMSE values.
    X = split["train_df"][["x1", "x2"]].to_numpy()
    y = split["train_df"]["y"].to_numpy()
    expected = []
    for train_idx, val_idx in KFold(
        n_splits=5, shuffle=True, random_state=42
    ).split(X):
        model = LinearRegression().fit(X[train_idx], y[train_idx])
        residual = y[val_idx] - model.predict(X[val_idx])
        expected.append(float(np.sqrt(np.mean(residual ** 2))))

    assert result["scope"] == "locked training partition"
    assert len(result["R2"]["folds"]) == 5
    assert len(result["RMSE"]["folds"]) == 5
    assert all(value >= 0 for value in result["RMSE"]["folds"])
    np.testing.assert_allclose(result["RMSE"]["folds"], expected)
    assert result["RMSE"]["mean"] == pytest.approx(np.mean(expected))
    assert result["RMSE"]["std"] == pytest.approx(np.std(expected))


def test_mean_baseline_can_be_cross_validated():
    result = ru.cross_validate_metrics(
        "Mean baseline", _three_feature_data(), ["x1"], "y", k=5
    )
    assert np.isfinite(result["RMSE"]["mean"])
    assert result["RMSE"]["mean"] > 0


def test_gradient_descent_data_patterns_are_reproducible_and_distinct():
    generated = {}
    for label, pattern in ru.GRADIENT_DESCENT_PATTERNS.items():
        first = ru.make_gradient_descent_data(label, n=80, noise=4.0)
        second = ru.make_gradient_descent_data(pattern, n=80, noise=4.0)
        pd.testing.assert_frame_equal(first, second)
        assert first.attrs["pattern"] == pattern
        assert first.attrs["description"]
        assert np.isfinite(first[["x", "y"]]).all().all()
        generated[pattern] = first

    x = generated["linear"]["x"].to_numpy()
    true_line = 4.0 + 2.2 * x
    linear_error = generated["linear"]["y"].to_numpy() - true_line
    noisy_error = generated["noisy"]["y"].to_numpy() - true_line
    assert np.std(noisy_error) == pytest.approx(2 * np.std(linear_error))

    curved_error = generated["curved"]["y"].to_numpy() - true_line
    assert np.corrcoef(curved_error, x ** 2)[0, 1] > 0.9
    assert generated["outlier"]["is_special"].sum() == 1

    funnel_error = generated["funnel"]["y"].to_numpy() - true_line
    assert np.std(funnel_error[-20:]) > 2 * np.std(funnel_error[:20])


def test_gradient_descent_raw_parameter_history_matches_internal_equations():
    df = ru.make_gradient_descent_data("curved", n=60, noise=3.0)
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    result = ru.gradient_descent_1d(x, y, alpha=0.3, n_iters=40)

    assert len(result["slope_raw_history"]) == 41
    assert len(result["intercept_raw_history"]) == 41
    standardized_x = (x - result["x_mu"]) / result["x_sd"]
    for step in (0, 1, 7, 40):
        internal_prediction = (
            result["theta0"][step]
            + result["theta1"][step] * standardized_x
        )
        raw_prediction = (
            result["intercept_raw_history"][step]
            + result["slope_raw_history"][step] * x
        )
        np.testing.assert_allclose(internal_prediction, raw_prediction)
    assert result["slope_raw"] == pytest.approx(result["slope_raw_history"][-1])
    assert result["intercept_raw"] == pytest.approx(
        result["intercept_raw_history"][-1]
    )


@pytest.mark.parametrize("model", ru.MODEL_CHOICES)
def test_every_available_model_choice_fits_and_predicts(model):
    df = ru.make_synthetic(n=80, noise=5.0, curvature=0.3)
    kwargs = {"degree": 3} if model == "Polynomial" else {}
    if model in {"Random Forest", "LightGBM", "XGBoost"}:
        kwargs["n_estimators"] = 20

    result = ru.fit_and_evaluate(model, df, ["x"], "y", **kwargs)

    assert np.isfinite(result["pred_train"]).all()
    assert np.isfinite(result["pred_test"]).all()


def test_feature_ranking_uses_training_cv_and_keeps_test_locked():
    df = _three_feature_data()
    ranking = ru.rank_feature_combos(
        df, ["x1", "x2", "x3"], "y", max_k=2, top=6
    )
    assert "CV_R2_mean" in ranking
    assert "CV_RMSE_mean" in ranking
    assert not any("test" in column.casefold() for column in ranking.columns)
    assert "x1" in ranking.iloc[0]["features"]
    split = ranking.attrs["locked_split"]
    assert set(split["train_indices"]).isdisjoint(split["test_indices"])
    assert ranking.attrs["selection_scope"].startswith("cross-validation")


def test_polynomial_term_count_and_budget_stop_dangerous_expansion():
    assert ru.polynomial_term_count(1, 8) == 8
    assert ru.polynomial_term_count(11, 8) == 75_581
    status = ru.polynomial_budget_status(11, 8, n_samples=1_000)
    assert not status["within_budget"]
    assert status["estimated_matrix_mib"] > 500
    with pytest.raises(ru.PolynomialBudgetError, match="75,581"):
        ru.check_polynomial_budget(11, 8, n_samples=1_000)

    df = _three_feature_data()
    with pytest.raises(ru.PolynomialBudgetError):
        ru.fit_and_evaluate(
            "Polynomial", df, ["x1", "x2", "x3"], "y",
            degree=8, max_polynomial_terms=100,
        )


def test_polynomial_coefficients_have_meaningful_names():
    df = _three_feature_data()
    result = ru.fit_and_evaluate(
        "Polynomial", df, ["x1", "x2"], "y", degree=2
    )
    names = ru.coefficients(
        result["pipeline"], ["x1", "x2"]
    )["feature"].tolist()
    assert names == ["x1", "x2", "x1^2", "x1 x2", "x2^2"]

    importance, kind = ru.feature_importance(
        result["pipeline"], ["x1", "x2"]
    )
    assert kind == "|coefficient|"
    assert set(importance["feature"]) == set(names)


def test_degree_and_alpha_sweeps_reuse_only_locked_training_rows():
    df = _three_feature_data()
    split = ru.make_locked_split(df, ["x1", "x2"], "y")
    degree = ru.degree_cv_sweep(
        df, ["x1", "x2"], "y", degrees=[1, 2, 3],
        locked_split=split,
    )
    alpha = ru.alpha_cv_sweep(
        "Ridge", df, ["x1", "x2"], "y", alphas=[0.01, 1.0, 100.0],
        locked_split=split,
    )
    assert degree["degree"].tolist() == [1, 2, 3]
    assert degree["n_terms"].tolist() == [2, 5, 9]
    assert (degree["train_RMSE"] >= 0).all()
    assert np.isfinite(degree["train_R2"]).all()
    assert alpha["alpha"].tolist() == [0.01, 1.0, 100.0]
    assert degree["is_best"].sum() == 1
    assert alpha["is_best"].sum() == 1
    np.testing.assert_array_equal(
        degree.attrs["locked_split"]["test_indices"], split["test_indices"]
    )
    np.testing.assert_array_equal(
        alpha.attrs["locked_split"]["test_indices"], split["test_indices"]
    )
    assert (degree["CV_RMSE_mean"] >= 0).all()
    assert (alpha["CV_RMSE_mean"] >= 0).all()


@pytest.mark.parametrize("scenario", list(ru.DIAGNOSTIC_SCENARIOS))
def test_diagnostic_scenarios_are_finite_reproducible_and_labelled(scenario):
    first = ru.make_diagnostic_scenario(scenario, n=80)
    second = ru.make_diagnostic_scenario(scenario, n=80)
    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == ["x", "y", "observation", "is_special"]
    assert np.isfinite(first[["x", "y", "observation"]]).all().all()
    assert first.attrs["scenario"] == scenario
    assert first.attrs["label"] == ru.DIAGNOSTIC_SCENARIOS[scenario]


def test_diagnostic_special_points_and_dependent_errors_are_distinct():
    outlier = ru.make_diagnostic_scenario("outlier")
    leverage = ru.make_diagnostic_scenario("High-leverage observation")
    dependent = ru.make_diagnostic_scenario("dependent", n=300)
    assert outlier["is_special"].sum() == 1
    assert leverage["is_special"].sum() == 1
    assert leverage.loc[leverage["is_special"], "x"].iloc[0] == 9.0

    fitted = LinearRegression().fit(dependent[["x"]], dependent["y"])
    residual = dependent["y"] - fitted.predict(dependent[["x"]])
    lag_one = np.corrcoef(residual.iloc[:-1], residual.iloc[1:])[0, 1]
    assert lag_one > 0.6
