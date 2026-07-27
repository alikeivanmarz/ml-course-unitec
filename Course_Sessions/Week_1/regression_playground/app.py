"""
Regression Playground · Fit, Break, Diagnose, Improve
=====================================================
Interactive Week 1, Session 2 laboratory for the Unitec machine-learning course.

The app deliberately separates model selection from final-test evaluation:
students explore with training data and training-only cross-validation, lock a
choice, and only then reveal the final test result.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

# Make sibling modules importable regardless of the launch directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import regression_utils as ru
import theme


st.set_page_config(
    page_title="Regression Playground",
    page_icon="=",
    layout="wide",
    initial_sidebar_state="auto",
)


WORKSPACES = [
    "Fit the Line",
    "Gradient Descent",
    "Feature Lab",
    "Model Arena",
    "Generalisation & Regularisation",
    "Diagnose",
]

FRIENDLY_NAMES = {
    "x": "Input x",
    "y": "Target y",
    "ENGINESIZE": "Engine size (L)",
    "CYLINDERS": "Cylinders",
    "FUELCONSUMPTION_CITY": "City fuel use (L/100 km)",
    "FUELCONSUMPTION_HWY": "Highway fuel use (L/100 km)",
    "FUELCONSUMPTION_COMB": "Combined fuel use (L/100 km)",
    "CO2EMISSIONS": "CO₂ emissions (g/km)",
    "R&D Spend": "R&D spend",
    "Administration": "Administration spend",
    "Marketing Spend": "Marketing spend",
    "Profit": "Profit",
    "Avg_Daily_Usage_Hours": "Daily social-media use (hours)",
    "Addicted_Score": "Self-reported addiction score",
    "alcohol": "Alcohol",
    "quality": "Wine quality score",
    "target": "Disease progression",
}

FIT_EXAMPLES = {
    "Engine size vs CO₂": ("FuelConsumption CO2", "ENGINESIZE", "CO2EMISSIONS"),
    "Fuel use vs CO₂": ("FuelConsumption CO2", "FUELCONSUMPTION_COMB", "CO2EMISSIONS"),
    "R&D spend vs profit": ("Startup Profit", "R&D Spend", "Profit"),
    "Daily use vs self-reported addiction score": (
        "Student wellbeing",
        "Avg_Daily_Usage_Hours",
        "Addicted_Score",
    ),
    "Alcohol vs wine quality score": ("Wine Quality (red)", "alcohol", "quality"),
}

DIAGNOSTIC_SCENARIOS = {
    "Mystery A": "clean",
    "Mystery B": "curved",
    "Mystery C": "funnel",
    "Mystery D": "outlier",
    "Mystery E": "high_leverage",
    "Mystery F": "skewed",
    "Mystery G": "dependent",
}
DIAGNOSTIC_CLUES = {
    "clean": "No strong visual warning",
    "curved": "Non-linearity",
    "funnel": "Non-constant variance",
    "outlier": "An unusual response",
    "high_leverage": "A high-leverage or influential point",
    "skewed": "Non-normal residual shape",
    "dependent": "Dependence across observation order",
}


# ---------------------------------------------------------------------------
# Cached data and computations
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_dataset(name: str):
    return ru.get_named_dataset(name)


@st.cache_data(show_spinner=False)
def fit_example_xy(label: str, max_points: int = 150):
    name, xcol, ycol = FIT_EXAMPLES[label]
    df = ru.get_named_dataset(name)[0][[xcol, ycol]].dropna()
    if len(df) > max_points:
        df = df.sample(max_points, random_state=ru.RANDOM_STATE)
    return df[xcol].to_numpy(float), df[ycol].to_numpy(float), xcol, ycol


@st.cache_data(show_spinner=False)
def cached_combo_ranking(dataset_name: str, max_k: int) -> pd.DataFrame:
    df, features, target = get_dataset(dataset_name)
    return ru.rank_feature_combos(
        df,
        features,
        target,
        max_k=max_k,
        top=10_000,
    )


@st.cache_data(show_spinner=False)
def cached_coefficient_path(dataset_name: str, model: str) -> pd.DataFrame:
    df, features, target = get_dataset(dataset_name)
    return ru.coefficient_path(df, features, target, model=model)


@st.cache_data(show_spinner=False)
def cached_degree_sweep(n: int, noise: float) -> pd.DataFrame:
    df = ru.make_synthetic(n=n, noise=noise, curvature=0.6)
    return ru.degree_cv_sweep(df, ["x"], "y", degrees=range(1, 13), k=5)


@st.cache_data(show_spinner=False)
def cached_alpha_sweep(dataset_name: str, model: str) -> pd.DataFrame:
    df, features, target = get_dataset(dataset_name)
    return ru.alpha_cv_sweep(
        model,
        df,
        features,
        target,
        alphas=np.logspace(-2, 3, 25),
        k=5,
    )


# ---------------------------------------------------------------------------
# Shared presentation helpers
# ---------------------------------------------------------------------------
def friendly(name: str) -> str:
    if name in FRIENDLY_NAMES:
        return FRIENDLY_NAMES[name]
    return (
        str(name)
        .replace("Orientation_", "Orientation code ")
        .replace("Glazing Area Distribution_", "Glazing distribution code ")
        .replace("_", " ")
    )


def compact_number(value: float) -> str:
    value = float(value)
    if not np.isfinite(value):
        return "∞"
    if value != 0 and (abs(value) >= 1_000_000 or abs(value) < 0.001):
        return f"{value:.2e}"
    return f"{value:,.2f}"


def slider_grid_value(low: float, high: float, desired: float, step: float) -> float:
    """Snap a floating-point default to a valid Streamlit slider step."""
    if step <= 0:
        raise ValueError("Slider step must be positive.")
    snapped = low + round((desired - low) / step) * step
    return float(np.clip(snapped, low, high))


def state_signature(*parts: Any) -> str:
    return "|".join(str(part) for part in parts)


def reveal_gate(
    state_key: str,
    signature: str,
    label: str,
) -> bool:
    if st.button(label, key=f"{state_key}_button", type="primary"):
        st.session_state[state_key] = signature
    return st.session_state.get(state_key) == signature


def remember_button(state_key: str, signature: str, label: str) -> bool:
    if st.button(label, key=f"{state_key}_button", type="primary"):
        st.session_state[state_key] = signature
    return st.session_state.get(state_key) == signature


def metric_value(metrics: dict, key: str) -> str:
    value = metrics.get(key, float("nan"))
    if value != value:
        return "n/a"
    if key == "R2":
        return f"{value:.3f}"
    if key == "MAPE":
        return f"{value:.1f}%"
    return f"{value:,.2f}"


def primary_metric_cards(metrics: dict, prefix: str, include_mse: bool = False) -> None:
    keys = ["R2", "RMSE", "MAE"]
    if include_mse:
        keys.insert(1, "MSE")
    labels = {"R2": "R²", "RMSE": "RMSE", "MAE": "MAE", "MSE": "MSE"}
    help_text = {
        "R2": "1 is perfect; 0 matches a mean baseline; negative is possible.",
        "RMSE": "Root mean squared error, in the target's units.",
        "MAE": "Mean absolute error; less sensitive to a large error than MSE.",
        "MSE": "Mean squared error; large errors receive extra weight.",
    }
    columns = st.columns(len(keys))
    for column, key in zip(columns, keys):
        column.metric(labels[key], metric_value(metrics, key), help=help_text[key])


def cv_block(cv_result: dict, metric: str) -> dict:
    for key in (metric, metric.upper(), metric.lower()):
        if key in cv_result:
            return cv_result[key]
    raise KeyError(f"CV result has no {metric} block: {list(cv_result)}")


def cv_text(cv_result: dict, metric: str, digits: int = 3) -> str:
    block = cv_block(cv_result, metric)
    return f"{block['mean']:.{digits}f} ± {block['std']:.{digits}f}"


def log_experiment(row: dict) -> None:
    st.session_state.setdefault("history", []).append(row)


def scatter_line_fig(
    x,
    y,
    lines: list[dict],
    *,
    height: int = 360,
    x_title: str = "x",
    y_title: str = "y",
    residual_from: dict | None = None,
    y_range: list[float] | None = None,
):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    fig = go.Figure()
    if residual_from is not None:
        prediction = residual_from["intercept"] + residual_from["slope"] * x
        rx, ry = [], []
        for xi, yi, pi in zip(x, y, prediction):
            rx.extend([xi, xi, None])
            ry.extend([yi, pi, None])
        fig.add_trace(
            go.Scatter(
                x=rx,
                y=ry,
                mode="lines",
                line=dict(
                    color=residual_from.get("color", theme.ERROR),
                    width=1,
                    dash="dot",
                ),
                name="residuals",
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name="observations",
            marker=dict(
                color=theme.DATA,
                size=8,
                line=dict(color=theme.PALETTE["data_edge"], width=1),
            ),
        )
    )
    xs = np.linspace(float(x.min()), float(x.max()), 200)
    for line in lines:
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=line["intercept"] + line["slope"] * xs,
                mode="lines",
                name=line["name"],
                line=dict(
                    color=line["color"],
                    width=line.get("width", 3),
                    dash=line.get("dash", "solid"),
                ),
            )
        )
    theme.style_fig(fig, height=height)
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        margin=dict(l=10, r=10, t=58, b=10),
    )
    fig.update_xaxes(title=x_title)
    fig.update_yaxes(title=y_title, range=y_range)
    return fig


def actual_vs_pred_fig(y_true, y_pred, *, height=320, color=None, title=""):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    color = color or theme.MODEL
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            name="perfect prediction",
            line=dict(color=theme.INK_SOFT, width=1.5, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=y_pred,
            mode="markers",
            name="predictions",
            marker=dict(
                color=color,
                size=7,
                opacity=0.72,
                line=dict(color=theme.PALETTE["data_edge"], width=0.5),
            ),
        )
    )
    theme.style_fig(fig, height=height, title=title)
    fig.update_xaxes(title="Actual")
    fig.update_yaxes(title="Predicted")
    return fig


def model_curve_figure(
    df: pd.DataFrame,
    feature: str,
    target: str,
    result_a: dict,
    result_b: dict,
    name_a: str,
    name_b: str,
    reveal_test: bool,
):
    fig = go.Figure()
    x_train = np.asarray(result_a["X_train"])[:, 0]
    y_train = np.asarray(result_a["y_train"])
    fig.add_trace(
        go.Scatter(
            x=x_train,
            y=y_train,
            mode="markers",
            name="training observations",
            marker=dict(color=theme.DATA, size=7, symbol="circle", opacity=0.68),
        )
    )
    if reveal_test:
        x_test = np.asarray(result_a["X_test"])[:, 0]
        y_test = np.asarray(result_a["y_test"])
        fig.add_trace(
            go.Scatter(
                x=x_test,
                y=y_test,
                mode="markers",
                name="final-test observations",
                marker=dict(
                    color=theme.GOOD,
                    size=9,
                    symbol="diamond-open",
                    line=dict(width=2),
                ),
            )
        )
    x_all = df[feature].to_numpy(dtype=float)
    for result, name, color, dash in [
        (result_a, name_a, theme.MODEL_A, "solid"),
        (result_b, name_b, theme.CB_BLUE, "solid"),
    ]:
        xs, curve = ru.predict_curve(
            result["pipeline"],
            float(x_all.min()),
            float(x_all.max()),
        )
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=curve,
                mode="lines",
                name=name,
                line=dict(color=color, width=3, dash=dash),
            )
        )
    theme.style_fig(fig, height=390, title=f"{friendly(target)} vs {friendly(feature)}")
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="left",
            x=0,
        ),
        margin=dict(l=10, r=10, t=46, b=78),
    )
    fig.update_xaxes(title=friendly(feature))
    fig.update_yaxes(title=friendly(target))
    return fig


def page_intro(title: str, subtitle: str, challenge: str | None = None) -> None:
    theme.header(title, subtitle)
    if challenge:
        theme.challenge(challenge)


# ---------------------------------------------------------------------------
# Workspace 1: Fit the Line
# ---------------------------------------------------------------------------
def workspace_fit_line() -> None:
    page_intro(
        "Fit the Line",
        "Fit a line by hand, then compare it with ordinary least squares",
        "Make the squared errors as small as you can before revealing the best fit.",
    )
    st.latex(r"\hat{y} = \theta_0 + \theta_1x")

    controls, visual = st.columns([1, 2], gap="large")
    with controls:
        choice = st.selectbox(
            "Example",
            ["Practice data (make your own)", *FIT_EXAMPLES, "Anscombe's Quartet"],
            key="fit_choice",
        )
        x_name, y_name = "x", "y"
        data_parts: list[Any] = [choice]
        if choice.startswith("Practice"):
            n = st.slider("Number of observations", 10, 200, 40, key="fit_n")
            noise = st.slider("Noise", 0.0, 20.0, 6.0, 0.5, key="fit_noise")
            seed = st.number_input("Random seed", 0, 9999, 42, key="fit_seed")
            add_outlier = st.checkbox("Add one unusual observation", key="fit_outlier")
            df = ru.make_synthetic(n=n, noise=noise, seed=int(seed))
            if add_outlier:
                df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            {
                                "x": [df["x"].max()],
                                "y": [df["y"].min() - 3 * noise - 10],
                            }
                        ),
                    ],
                    ignore_index=True,
                )
            x = df["x"].to_numpy()
            y = df["y"].to_numpy()
            data_parts.extend([n, noise, seed, add_outlier])
        elif choice == "Anscombe's Quartet":
            quartet = st.selectbox("Quartet", ["I", "II", "III", "IV"], key="fit_anscombe")
            df = ru.make_anscombe(quartet)
            x, y = df["x"].to_numpy(), df["y"].to_numpy()
            data_parts.append(quartet)
            st.caption("Similar summary statistics can hide very different data patterns.")
        else:
            x, y, x_name, y_name = fit_example_xy(choice)

        ols_slope, ols_intercept = ru.ols_1d(x, y)
        slope_limit = max(abs(ols_slope) * 3.0, 1.0)
        y_span = float(y.max() - y.min()) or 1.0
        intercept_low = float(y.min() - y_span)
        intercept_high = float(y.max() + y_span)
        intercept_step = y_span / 100
        intercept_default = slider_grid_value(
            intercept_low,
            intercept_high,
            float(y.mean()),
            intercept_step,
        )
        slope = st.slider(
            "Slope θ₁",
            -slope_limit,
            slope_limit,
            0.0,
            step=slope_limit / 100,
            key=f"fit_slope_{choice}",
        )
        intercept = st.slider(
            "Intercept θ₀",
            intercept_low,
            intercept_high,
            intercept_default,
            step=intercept_step,
            key=f"fit_intercept_{choice}",
        )

    with visual:
        data_signature = state_signature(*data_parts)
        show_best = st.checkbox(
            "Reveal the least-squares line",
            key=f"fit_reveal_{data_signature}",
            help="Fit by eye first, then reveal the computer's solution.",
        )
        student_prediction = intercept + slope * x
        student_metrics = ru.all_metrics(y, student_prediction)
        student_sse, student_mse = ru.sse_mse(x, y, slope, intercept)
        best_prediction = ols_intercept + ols_slope * x
        best_sse, best_mse = ru.sse_mse(x, y, ols_slope, ols_intercept)
        lines = [
            {
                "slope": slope,
                "intercept": intercept,
                "name": "your line",
                "color": theme.MODEL,
            }
        ]
        if show_best:
            lines.append(
                {
                    "slope": ols_slope,
                    "intercept": ols_intercept,
                    "name": "least-squares line",
                    "color": theme.GOOD,
                    "dash": "dash",
                }
            )
        fig = scatter_line_fig(
            x,
            y,
            lines,
            x_title=friendly(x_name),
            y_title=friendly(y_name),
            residual_from={"slope": slope, "intercept": intercept, "color": theme.ERROR},
        )
        st.plotly_chart(fig, width="stretch", key="fit_figure", config={"displayModeBar": False})
        st.caption("Dotted segments are residuals. OLS minimises their squared lengths.")

        cards = st.columns(4)
        cards[0].metric("SSE", f"{student_sse:,.1f}", help="Total squared error")
        cards[1].metric("MSE", f"{student_mse:,.2f}", help="Average squared error")
        cards[2].metric("RMSE", metric_value(student_metrics, "RMSE"))
        cards[3].metric("R²", metric_value(student_metrics, "R2"))

        if show_best:
            ratio = student_mse / best_mse if best_mse > 0 else 1.0
            st.success(
                f"Least-squares line: slope {ols_slope:.3g}, intercept {ols_intercept:.3g}. "
                f"Its MSE is {best_mse:.2f}; your MSE is {ratio:.2f}× as large."
            )

        predict_col, result_col = st.columns([3, 1])
        predict_low = float(x.min())
        predict_high = float(x.max())
        predict_step = (predict_high - predict_low) / 100
        predict_default = slider_grid_value(
            predict_low,
            predict_high,
            float(np.median(x)),
            predict_step,
        )
        predict_x = predict_col.slider(
            f"Predict {friendly(y_name)} when {friendly(x_name)} is",
            predict_low,
            predict_high,
            predict_default,
            step=predict_step,
            key=f"fit_predict_{choice}",
        )
        result_col.metric("Your prediction", f"{intercept + slope * predict_x:,.2f}")

        if st.button("Log this attempt", key="fit_log"):
            log_experiment(
                {
                    "workspace": "Fit the Line",
                    "example": choice,
                    "slope": round(slope, 4),
                    "intercept": round(intercept, 4),
                    "MSE": round(student_mse, 4),
                    "best_MSE": round(best_mse, 4),
                }
            )
            st.success("Attempt added to My Experiments.")


# ---------------------------------------------------------------------------
# Workspace 2: Gradient Descent
# ---------------------------------------------------------------------------
def workspace_gradient_descent() -> None:
    page_intro(
        "Gradient Descent",
        "Watch the machine approach the same least-squares solution",
        "Hold the data fixed and compare how several learning rates move through the same cost surface.",
    )
    with st.expander("Show the maths"):
        left, right = st.columns(2)
        left.markdown("**Update rule**")
        left.latex(r"\theta \leftarrow \theta - \alpha\frac{\partial J}{\partial\theta}")
        right.markdown("**Half-MSE cost**")
        right.latex(
            r"J(\theta)=\frac{1}{2m}\sum_{i=1}^{m}"
            r"\left(h_\theta(x^{(i)})-y^{(i)}\right)^2"
        )

    data_controls = st.columns([1.5, 1, 1])
    pattern_label = data_controls[0].selectbox(
        "Data pattern",
        list(ru.GRADIENT_DESCENT_PATTERNS),
        key="gd_pattern",
    )
    n = data_controls[1].slider("Observations", 20, 200, 60, key="gd_n")
    noise = data_controls[2].slider(
        "Base noise σ",
        0.0,
        20.0,
        8.0,
        0.5,
        key="gd_noise",
    )
    st.caption(
        "Full-batch gradient descent uses every row for each update. More rows can make "
        "one update cost more computation, but more rows alone do not necessarily require "
        "more iterations to converge."
    )

    optimiser_controls = st.columns([1.5, 1])
    preset = optimiser_controls[0].selectbox(
        "Learning rate α",
        ["Too small · 0.01", "Useful · 0.30", "Too large · 2.50", "Custom"],
        index=1,
        key="gd_preset",
    )
    iterations = optimiser_controls[1].slider(
        "Iterations",
        10,
        500,
        150,
        key="gd_iterations",
    )
    preset_values = {
        "Too small · 0.01": 0.01,
        "Useful · 0.30": 0.30,
        "Too large · 2.50": 2.50,
    }
    alpha = (
        st.slider("Custom learning rate α", 0.001, 3.0, 0.3, 0.001, key="gd_alpha")
        if preset == "Custom"
        else preset_values[preset]
    )

    pattern = ru.GRADIENT_DESCENT_PATTERNS[pattern_label]
    df = ru.make_gradient_descent_data(pattern, n=n, noise=noise)
    st.caption(df.attrs["description"])
    x, y = df["x"].to_numpy(), df["y"].to_numpy()
    gd = ru.gradient_descent_1d(x, y, alpha=alpha, n_iters=iterations)
    ols_slope, ols_intercept = ru.ols_1d(x, y)
    step = st.slider(
        "Scrub through iterations",
        0,
        len(gd["cost"]) - 1,
        0,
        key=f"gd_step_{iterations}_{preset}",
    )

    theta0_internal = gd["theta0"][step]
    theta1_internal = gd["theta1"][step]
    slope_now = gd["slope_raw_history"][step]
    intercept_now = gd["intercept_raw_history"][step]
    cost = np.asarray(gd["cost"], dtype=float)

    if gd["diverged"]:
        status = "Diverging"
        st.warning("The steps are too large: cost is growing rather than shrinking.")
    elif step == 0:
        status = "Not started"
    elif step > 2 and cost[step] > cost[0] * 0.98:
        status = "Stalled or oscillating"
        st.info("Cost is not falling meaningfully. Try a different learning rate.")
    else:
        status = "Converging"

    comparison_specs = [
        ("α = 0.01 · too small", 0.01, theme.NEUTRAL, "dot"),
        ("α = 0.30 · useful", 0.30, theme.GOOD, "solid"),
    ]
    # Keep the comparison readable: divergent rates remain available as a
    # separate warning/demo, but they are not drawn because they overwhelm the
    # scale of the useful convergence curves.
    if (
        preset == "Custom"
        and not gd["diverged"]
        and not any(np.isclose(alpha, rate) for _, rate, _, _ in comparison_specs)
    ):
        comparison_specs.append(
            (f"α = {alpha:.3g} · custom", alpha, theme.MODEL, "dashdot")
        )
    comparison_runs = [
        (
            label,
            rate,
            color,
            dash,
            ru.gradient_descent_1d(x, y, alpha=rate, n_iters=iterations),
        )
        for label, rate, color, dash in comparison_specs
    ]

    fit_col, cost_col = st.columns(2, gap="large")
    with fit_col:
        y_padding = max(float(np.ptp(y)) * 0.15, 1.0)
        fit_fig = scatter_line_fig(
            x,
            y,
            [
                {
                    "slope": slope_now,
                    "intercept": intercept_now,
                    "name": f"gradient descent · step {step}",
                    "color": theme.MODEL,
                },
                {
                    "slope": ols_slope,
                    "intercept": ols_intercept,
                    "name": "OLS target",
                    "color": theme.GOOD,
                    "dash": "dash",
                },
            ],
            height=340,
            y_range=[float(y.min() - y_padding), float(y.max() + y_padding)],
        )
        st.plotly_chart(
            fit_fig,
            width="stretch",
            key="gd_fit",
            config={"displayModeBar": False},
        )
        parameter_cols = st.columns(2)
        parameter_cols[0].metric(
            "θ₀ · raw-x intercept",
            compact_number(intercept_now),
        )
        parameter_cols[1].metric(
            "θ₁ · raw-x coefficient",
            compact_number(slope_now),
        )
        equation_sign = "+" if slope_now >= 0 else "−"
        st.caption(
            f"Plotted equation at step {step}: ŷ = {compact_number(intercept_now)} "
            f"{equation_sign} {compact_number(abs(slope_now))}x. Internally, x is "
            f"standardised, so the "
            f"optimised parameters are θ₀* = {compact_number(theta0_internal)} and "
            f"θ₁* = {compact_number(theta1_internal)} before conversion back to raw x."
        )
        if pattern == "curved":
            st.info(
                "Convergence only finds the best straight line. It cannot make a linear "
                "model represent the curved data-generating pattern."
            )
    with cost_col:
        cost_fig = go.Figure()
        displayed_positive = []
        selected_rate_plotted = False
        for label, rate, color, dash, run in comparison_runs:
            run_cost = np.asarray(run["cost"], dtype=float)
            valid = np.isfinite(run_cost) & (run_cost > 0)
            plot_cost = np.where(valid, run_cost, np.nan)
            displayed_positive.extend(
                plot_cost[np.isfinite(plot_cost) & (plot_cost > 0)].tolist()
            )
            selected_rate = bool(np.isclose(rate, alpha))
            selected_rate_plotted = selected_rate_plotted or selected_rate
            cost_fig.add_trace(
                go.Scatter(
                    x=np.arange(len(plot_cost)),
                    y=plot_cost,
                    mode="lines",
                    name=("Selected · " if selected_rate else "") + label,
                    line=dict(
                        color=color,
                        width=4.0 if selected_rate else 2.0,
                        dash=dash,
                    ),
                    opacity=1.0 if selected_rate else 0.7,
                )
            )
        if selected_rate_plotted:
            current_cost = cost[step] if step < len(cost) else np.nan
            cost_fig.add_trace(
                go.Scatter(
                    x=[step],
                    y=[current_cost],
                    mode="markers",
                    name=f"selected α · step {step}",
                    marker=dict(
                        color=theme.MODEL,
                        size=12,
                        line=dict(color="white", width=1.5),
                        symbol="circle",
                    ),
                )
            )
        theme.style_fig(
            cost_fig,
            height=410,
            title="Learning-rate comparison · half-MSE cost J(θ)",
        )
        cost_fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.2,
                xanchor="left",
                x=0,
            ),
            margin=dict(l=10, r=10, t=46, b=82),
        )
        cost_fig.update_xaxes(title="Iteration")
        cost_fig.update_yaxes(title="Cost J")
        positive = np.asarray(displayed_positive, dtype=float)
        use_log_scale = (
            len(positive) > 1
            and float(positive.max() / positive.min()) > 50
        )
        if use_log_scale:
            cost_fig.update_yaxes(type="log")
        st.plotly_chart(
            cost_fig,
            width="stretch",
            key="gd_cost",
            config={"displayModeBar": False},
        )
        st.caption(
            "The reference curves use the same observations and start from θ₀ = θ₁ = 0; "
            "only α changes."
        )
        if selected_rate_plotted:
            st.caption(
                "The selected plotted α is drawn thicker and its current step is marked."
            )
        else:
            st.caption(
                "The selected divergent α is intentionally omitted from this comparison "
                "so the useful curves remain readable; the warning and metrics still show "
                "its behaviour."
            )
        if use_log_scale:
            st.caption(
                "The vertical axis uses a log scale when the plotted costs span a wide range."
            )

    metrics = st.columns(4)
    metrics[0].metric("Status", status)
    metrics[1].metric("Start cost", compact_number(cost[0]))
    metrics[2].metric("Current cost", compact_number(cost[step]))
    metrics[3].metric(
        "Raw θ₁ · GD / OLS",
        f"{compact_number(slope_now)} / {compact_number(ols_slope)}",
    )


# ---------------------------------------------------------------------------
# Workspace 3: Feature Lab
# ---------------------------------------------------------------------------
def workspace_feature_lab() -> None:
    page_intro(
        "Feature Lab",
        "Find features that help a linear model predict",
        "Build the strongest model using no more than three features, then compare it safely.",
    )
    dataset_name = st.selectbox("Dataset", ru.MULTI_FEATURE_DATASETS, key="feature_dataset")
    df, features, target = get_dataset(dataset_name)
    st.caption(
        f"{len(df):,} observations · predicting **{friendly(target)}** "
        f"from {len(features)} available numeric/encoded features."
    )

    correlation_col, selection_col = st.columns([1, 1], gap="large")
    with correlation_col:
        correlations = (
            df[features + [target]]
            .corr(numeric_only=True)[target]
            .drop(target)
            .sort_values()
        )
        corr_frame = pd.DataFrame(
            {
                "Feature": [friendly(name) for name in correlations.index],
                "Correlation": correlations.values,
            }
        )
        corr_fig = px.bar(
            corr_frame,
            x="Correlation",
            y="Feature",
            orientation="h",
            color="Correlation",
            color_continuous_scale=getattr(theme, "CORRELATION_SCALE", "RdBu_r"),
            range_color=[-1, 1],
            title=f"Correlation with {friendly(target)}",
        )
        corr_fig.update_layout(coloraxis_showscale=False)
        theme.style_fig(corr_fig, height=max(310, 28 * len(features)))
        st.plotly_chart(
            corr_fig,
            width="stretch",
            key="feature_target_corr",
            config={"displayModeBar": False},
        )
        with st.expander("Show full correlation matrix"):
            corr_matrix = df[features + [target]].corr(numeric_only=True)
            corr_matrix.index = [friendly(name) for name in corr_matrix.index]
            corr_matrix.columns = [friendly(name) for name in corr_matrix.columns]
            matrix_fig = px.imshow(
                corr_matrix,
                text_auto=".2f",
                zmin=-1,
                zmax=1,
                color_continuous_scale=getattr(theme, "CORRELATION_SCALE", "RdBu_r"),
                aspect="auto",
            )
            theme.style_fig(matrix_fig, height=max(380, 34 * len(corr_matrix)))
            st.plotly_chart(
                matrix_fig,
                width="stretch",
                key="feature_corr_matrix",
                config={"displayModeBar": False},
            )

    with selection_col:
        chosen = st.multiselect(
            "Features · choose one to three",
            features,
            default=[features[0]],
            max_selections=3,
            format_func=friendly,
            key=f"feature_choice_{dataset_name}",
        )
        if not chosen:
            st.info("Select at least one feature.")
            return
        # Lock rows using every candidate feature so changing the selected
        # subset cannot silently change which observations form the final test.
        locked_split = ru.make_locked_split(df, features, target)
        result = ru.fit_and_evaluate(
            "Linear",
            df,
            chosen,
            target,
            locked_split=locked_split,
        )
        cv = ru.cross_validate_metrics(
            "Linear",
            df,
            chosen,
            target,
            k=5,
            locked_split=locked_split,
            training_only=True,
        )
        if len(chosen) > 1:
            selected_corr = df[chosen].corr().abs()
            np.fill_diagonal(selected_corr.values, 0)
            if selected_corr.to_numpy().max() > 0.85:
                st.warning(
                    "Some selected predictors are strongly correlated. Coefficients may be unstable "
                    "even after standardisation."
                )

    cv_r2 = cv_block(cv, "R2")
    cv_rmse = cv_block(cv, "RMSE")
    summary = st.columns(3)
    summary[0].metric("Training R²", f"{result['train']['R2']:.3f}")
    summary[1].metric(
        "CV R²",
        f"{cv_r2['mean']:.3f}",
        delta=f"± {cv_r2['std']:.3f} SD",
        delta_color="off",
        help="Mean cross-validation R²; the second line is the fold-to-fold standard deviation.",
    )
    summary[2].metric(
        "CV RMSE",
        f"{cv_rmse['mean']:.2f}",
        delta=f"± {cv_rmse['std']:.2f} SD",
        delta_color="off",
        help="Mean cross-validation RMSE; the second line is the fold-to-fold standard deviation.",
    )
    st.caption("Cross-validation uses only the locked training partition.")

    st.markdown("### Coefficients")
    raw_result = ru.fit_and_evaluate(
        "Linear",
        df,
        chosen,
        target,
        scale=False,
        locked_split=locked_split,
    )
    standard_result = ru.fit_and_evaluate(
        "Linear",
        df,
        chosen,
        target,
        scale=True,
        locked_split=locked_split,
    )
    raw_coef = ru.coefficients(raw_result["pipeline"], chosen).rename(
        columns={"coefficient": "Raw"}
    )
    std_coef = ru.coefficients(standard_result["pipeline"], chosen).rename(
        columns={"coefficient": "Standardised"}
    )
    coefficient_frame = raw_coef.merge(std_coef, on="feature")
    coefficient_frame["feature"] = coefficient_frame["feature"].map(friendly)
    if len(chosen) == 1:
        cards = st.columns(2)
        cards[0].metric("Raw coefficient", f"{coefficient_frame['Raw'].iloc[0]:.3g}")
        cards[1].metric(
            "Per 1-SD increase",
            f"{coefficient_frame['Standardised'].iloc[0]:.3g}",
        )
    else:
        long_coef = coefficient_frame.melt(
            id_vars="feature",
            value_vars=["Raw", "Standardised"],
            var_name="Coefficient scale",
            value_name="Coefficient",
        )
        coefficient_fig = px.bar(
            long_coef,
            x="Coefficient",
            y="feature",
            color="Coefficient scale",
            barmode="group",
            orientation="h",
            color_discrete_map={
                "Raw": theme.NEUTRAL,
                "Standardised": theme.MODEL,
            },
            title="Raw and standardised coefficients",
        )
        theme.style_fig(coefficient_fig, height=max(260, 60 * len(chosen)))
        st.plotly_chart(
            coefficient_fig,
            width="stretch",
            key="feature_coefficients",
            config={"displayModeBar": False},
        )
    st.caption(
        "Standardising makes units comparable, but correlated predictors can still make "
        "coefficients unstable. Association is not causation."
    )

    ranking_signature = dataset_name
    if st.button("Reveal cross-validated feature ranking", key="feature_ranking_button", type="primary"):
        st.session_state["feature_ranking"] = ranking_signature
    if st.session_state.get("feature_ranking") == ranking_signature:
        with st.spinner("Comparing feature combinations using training-only cross-validation..."):
            ranking = cached_combo_ranking(dataset_name, 3)
        ranking_display = ranking.copy()
        # The modelling frame stores its locked split in attrs for Python
        # callers. Streamlit only needs the visible values and cannot serialise
        # the nested train/test DataFrames held in that metadata.
        ranking_display.attrs = {}
        rename_map = {
            "CV_R2_mean": "CV R² mean",
            "CV_R2_std": "CV R² SD",
            "CV_RMSE_mean": "CV RMSE mean",
            "CV_RMSE_std": "CV RMSE SD",
        }
        ranking_display = ranking_display.rename(columns=rename_map)
        st.dataframe(ranking_display.head(12), width="stretch", hide_index=True)
        chosen_set = set(chosen)
        matches = ranking.index[
            ranking["features"].apply(lambda text: set(str(text).split(", ")) == chosen_set)
        ].tolist()
        if matches:
            rank = matches[0] + 1
            best = float(ranking.iloc[0]["CV_R2_mean"])
            own = float(ranking.iloc[matches[0]]["CV_R2_mean"])
            st.info(
                f"Your choice ranks {rank} of {len(ranking)} combinations; "
                f"its CV R² is {best - own:.3f} below the top-ranked combination."
            )

    final_signature = state_signature(dataset_name, *sorted(chosen))
    final_reveal = reveal_gate(
        "feature_final_reveal",
        final_signature,
        "Lock these features and reveal the final test",
    )
    if final_reveal:
        st.markdown("### Final locked-test result")
        primary_metric_cards(result["test"], "feature_final")
        st.plotly_chart(
            actual_vs_pred_fig(
                result["y_test"],
                result["pred_test"],
                title="Final-test predictions",
            ),
            width="stretch",
            key="feature_final_plot",
            config={"displayModeBar": False},
        )


# ---------------------------------------------------------------------------
# Workspace 4: Model Arena
# ---------------------------------------------------------------------------
def model_controls(letter: str, n_features: int):
    options = list(ru.MODEL_CHOICES)
    state_key = f"arena_model_{letter}"
    current = st.session_state.get(state_key)
    default_name = "Linear" if letter == "A" else "Ridge"
    if current not in options:
        st.session_state.pop(state_key, None)
        model = st.selectbox(
            f"Model {letter}",
            options,
            index=options.index(default_name),
            key=state_key,
        )
    else:
        model = st.selectbox(f"Model {letter}", options, key=state_key)
    params: dict[str, Any] = {}
    if model == "Polynomial":
        max_degree = 8 if n_features == 1 else 3 if n_features <= 3 else 2
        degree = st.slider(
            f"Polynomial degree · {letter}",
            1,
            max_degree,
            min(2, max_degree),
            key=f"arena_degree_{letter}",
        )
        params["degree"] = degree
        terms = ru.polynomial_term_count(n_features, degree)
        st.caption(f"{terms:,} expanded polynomial terms")
        ru.check_polynomial_budget(n_features, degree)
    if model in ("Ridge", "Lasso"):
        exponent = st.slider(
            f"log₁₀ α · {letter}",
            -2.0,
            3.0,
            0.0,
            0.5,
            key=f"arena_alpha_{letter}",
        )
        params["alpha"] = float(10**exponent)
    if model in ("Decision Tree", "Random Forest", "LightGBM", "XGBoost"):
        limit_text = "0 = model default" if model == "XGBoost" else "0 = no limit"
        params["max_depth"] = st.slider(
            f"Maximum tree depth · {letter} ({limit_text})",
            0,
            14,
            6,
            key=f"arena_depth_{letter}",
        )
    if model in ("Random Forest", "LightGBM", "XGBoost"):
        params["n_estimators"] = st.slider(
            f"Number of trees · {letter}",
            50,
            500,
            200,
            50,
            key=f"arena_trees_{letter}",
        )
    if model == "SVM (RBF)":
        params["C"] = st.slider(
            f"SVM C · {letter}",
            0.1,
            100.0,
            10.0,
            key=f"arena_c_{letter}",
        )
    if model == "K-Nearest Neighbours":
        params["n_neighbors"] = st.slider(
            f"Neighbours k · {letter}",
            1,
            30,
            5,
            key=f"arena_k_{letter}",
        )
    return model, params


def model_comparison_table(
    model_a: str,
    model_b: str,
    result_a: dict,
    result_b: dict,
    cv_a: dict,
    cv_b: dict,
    reveal_test: bool,
) -> pd.DataFrame:
    rows = []
    for metric, label in [("R2", "R²"), ("RMSE", "RMSE"), ("MAE", "MAE")]:
        row = {
            "Metric": label,
            f"{model_a} · train": metric_value(result_a["train"], metric),
            f"{model_a} · CV": (
                cv_text(cv_a, metric, 2 if metric != "R2" else 3)
                if metric in ("R2", "RMSE")
                else "—"
            ),
            f"{model_b} · train": metric_value(result_b["train"], metric),
            f"{model_b} · CV": (
                cv_text(cv_b, metric, 2 if metric != "R2" else 3)
                if metric in ("R2", "RMSE")
                else "—"
            ),
        }
        if reveal_test:
            row[f"{model_a} · final test"] = metric_value(result_a["test"], metric)
            row[f"{model_b} · final test"] = metric_value(result_b["test"], metric)
        rows.append(row)
    return pd.DataFrame(rows)


def workspace_model_arena() -> None:
    page_intro(
        "Model Arena",
        "Compare two models using one locked data split",
        "Does the model that fits training data best also perform consistently in cross-validation?",
    )
    dataset_name = st.selectbox("Dataset", ru.ARENA_DATASETS, key="arena_dataset")
    df, features, target = get_dataset(dataset_name)
    chosen = st.multiselect(
        "Features",
        features,
        default=[features[0]],
        format_func=friendly,
        key=f"arena_features_{dataset_name}",
    )
    if not chosen:
        st.info("Select at least one feature.")
        return
    st.caption(
        f"All {len(ru.MODEL_CHOICES)} models available in this environment "
        "are listed directly in both selectors."
    )
    model_columns = st.columns(2, gap="large")
    try:
        with model_columns[0]:
            model_a, params_a = model_controls("A", len(chosen))
        with model_columns[1]:
            model_b, params_b = model_controls("B", len(chosen))
    except ru.PolynomialBudgetError as exc:
        st.error(str(exc))
        return

    try:
        # Keep the final-test rows fixed even when students change the feature
        # subset while comparing model configurations.
        locked_split = ru.make_locked_split(df, features, target)
        result_a = ru.fit_and_evaluate(
            model_a,
            df,
            chosen,
            target,
            locked_split=locked_split,
            **params_a,
        )
        result_b = ru.fit_and_evaluate(
            model_b,
            df,
            chosen,
            target,
            locked_split=locked_split,
            **params_b,
        )
        cv_a = ru.cross_validate_metrics(
            model_a,
            df,
            chosen,
            target,
            k=5,
            locked_split=locked_split,
            training_only=True,
            **params_a,
        )
        cv_b = ru.cross_validate_metrics(
            model_b,
            df,
            chosen,
            target,
            k=5,
            locked_split=locked_split,
            training_only=True,
            **params_b,
        )
    except (ValueError, ru.PolynomialBudgetError) as exc:
        st.error(f"This configuration cannot be fitted safely: {exc}")
        return

    signature = state_signature(
        dataset_name,
        *chosen,
        model_a,
        sorted(params_a.items()),
        model_b,
        sorted(params_b.items()),
    )
    reveal_test = reveal_gate(
        "arena_final_reveal",
        signature,
        "Lock both models and reveal the final test",
    )

    if len(chosen) == 1:
        curve_fig = model_curve_figure(
            df,
            chosen[0],
            target,
            result_a,
            result_b,
            model_a,
            model_b,
            reveal_test,
        )
        st.plotly_chart(
            curve_fig,
            width="stretch",
            key="arena_curve",
            config={"displayModeBar": False},
        )
        if not reveal_test:
            st.caption("Final-test observations remain hidden until both models are locked.")
    elif reveal_test:
        prediction_columns = st.columns(2)
        prediction_columns[0].plotly_chart(
            actual_vs_pred_fig(
                result_a["y_test"],
                result_a["pred_test"],
                color=theme.MODEL,
                title=f"{model_a} · final test",
            ),
            width="stretch",
            key="arena_actual_a",
            config={"displayModeBar": False},
        )
        prediction_columns[1].plotly_chart(
            actual_vs_pred_fig(
                result_b["y_test"],
                result_b["pred_test"],
                color=theme.NEUTRAL,
                title=f"{model_b} · final test",
            ),
            width="stretch",
            key="arena_actual_b",
            config={"displayModeBar": False},
        )

    comparison = model_comparison_table(
        model_a,
        model_b,
        result_a,
        result_b,
        cv_a,
        cv_b,
        reveal_test,
    )
    st.dataframe(comparison, width="stretch", hide_index=True)
    with st.expander("More metrics"):
        extra = pd.DataFrame(
            {
                "Model / split": [
                    f"{model_a} · train",
                    f"{model_b} · train",
                    *(
                        [f"{model_a} · final test", f"{model_b} · final test"]
                        if reveal_test
                        else []
                    ),
                ],
                "MSE": [
                    result_a["train"]["MSE"],
                    result_b["train"]["MSE"],
                    *(
                        [result_a["test"]["MSE"], result_b["test"]["MSE"]]
                        if reveal_test
                        else []
                    ),
                ],
                "MAPE": [
                    result_a["train"]["MAPE"],
                    result_b["train"]["MAPE"],
                    *(
                        [result_a["test"]["MAPE"], result_b["test"]["MAPE"]]
                        if reveal_test
                        else []
                    ),
                ],
            }
        )
        st.dataframe(extra, width="stretch", hide_index=True)

    cv_difference = cv_block(cv_b, "R2")["mean"] - cv_block(cv_a, "R2")["mean"]
    combined_sd = cv_block(cv_a, "R2")["std"] + cv_block(cv_b, "R2")["std"]
    if abs(cv_difference) <= combined_sd:
        st.info("The models' cross-validation ranges overlap; there is no clear winner yet.")
    else:
        better = model_b if cv_difference > 0 else model_a
        st.info(f"{better} has the higher mean CV R², with variability shown in the table.")
    if reveal_test:
        test_difference = result_b["test"]["R2"] - result_a["test"]["R2"]
        better = model_b if test_difference > 0 else model_a
        st.success(
            f"{better} achieved the higher R² on this one locked final-test set. "
            "This is evidence for this split, not a universal ranking."
        )

    if len(chosen) > 1:
        with st.expander("Inspect model coefficients or feature importance"):
            importance_columns = st.columns(2)
            for column, result, model, color, key in [
                (importance_columns[0], result_a, model_a, theme.MODEL, "arena_importance_a"),
                (importance_columns[1], result_b, model_b, theme.NEUTRAL, "arena_importance_b"),
            ]:
                importance, kind = ru.feature_importance(result["pipeline"], chosen)
                if kind:
                    importance["feature"] = importance["feature"].map(friendly)
                    fig = px.bar(
                        importance,
                        x="importance",
                        y="feature",
                        orientation="h",
                        title=f"{model} · {kind}",
                    )
                    fig.update_traces(marker_color=color)
                    theme.style_fig(fig, height=max(260, 32 * len(importance)))
                    column.plotly_chart(
                        fig,
                        width="stretch",
                        key=key,
                        config={"displayModeBar": False},
                    )
                else:
                    column.caption(f"{model} does not expose a simple importance measure.")

    if st.button("Log this comparison", key="arena_log"):
        log_experiment(
            {
                "workspace": "Model Arena",
                "dataset": dataset_name,
                "model_A": model_a,
                "model_B": model_b,
                "A_CV_R2": round(cv_block(cv_a, "R2")["mean"], 4),
                "B_CV_R2": round(cv_block(cv_b, "R2")["mean"], 4),
            }
        )
        st.success("Comparison added to My Experiments.")


# ---------------------------------------------------------------------------
# Workspace 5: Generalisation and regularisation
# ---------------------------------------------------------------------------
def complexity_lab() -> None:
    theme.challenge("Predict whether the chosen degree underfits, generalises or overfits.")
    controls, fit_column = st.columns([1, 2], gap="large")
    with controls:
        n = st.slider("Observations", 20, 150, 40, key="complexity_n")
        noise = st.slider("Noise", 1.0, 12.0, 4.0, 0.5, key="complexity_noise")
        degree = st.slider("Polynomial degree", 1, 12, 1, key="complexity_degree")
        prediction = st.radio(
            "My prediction",
            ["Underfitting", "About right", "Overfitting"],
            key="complexity_prediction",
        )

    df = ru.make_synthetic(n=n, noise=noise, curvature=0.6)
    result = ru.fit_and_evaluate("Polynomial", df, ["x"], "y", degree=degree)
    signature = state_signature(n, noise, degree, prediction)
    reveal = reveal_gate(
        "complexity_reveal",
        signature,
        "Commit prediction and reveal CV evidence",
    )
    with fit_column:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=np.asarray(result["X_train"])[:, 0],
                y=result["y_train"],
                mode="markers",
                name="training observations",
                marker=dict(color=theme.DATA, symbol="circle", size=8),
            )
        )
        if reveal:
            fig.add_trace(
                go.Scatter(
                    x=np.asarray(result["X_test"])[:, 0],
                    y=result["y_test"],
                    mode="markers",
                    name="locked final-test observations",
                    marker=dict(
                        color=theme.GOOD,
                        symbol="diamond-open",
                        size=9,
                        line=dict(width=2),
                    ),
                )
            )
        xs = np.linspace(float(df["x"].min()), float(df["x"].max()), 300)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=result["pipeline"].predict(xs.reshape(-1, 1)),
                mode="lines",
                name=f"degree {degree}",
                line=dict(color=theme.MODEL, width=3),
            )
        )
        theme.style_fig(fig, height=350, title=f"Polynomial degree {degree}")
        fig.update_xaxes(title="Input x")
        fig.update_yaxes(title="Target y")
        st.plotly_chart(
            fig,
            width="stretch",
            key="complexity_fit",
            config={"displayModeBar": False},
        )

    if not reveal:
        st.info("Cross-validation and final-test observations are hidden until you commit.")
        return

    sweep = cached_degree_sweep(n, noise)
    error_fig = go.Figure()
    error_fig.add_trace(
        go.Scatter(
            x=sweep["degree"],
            y=sweep["train_RMSE"],
            mode="lines+markers",
            name="training RMSE",
            line=dict(color=theme.DATA),
        )
    )
    cv_mean = sweep["CV_RMSE_mean"].to_numpy()
    cv_std = sweep["CV_RMSE_std"].to_numpy()
    degrees = sweep["degree"].to_numpy()
    cv_lower = np.maximum(cv_mean - cv_std, 0)
    error_fig.add_trace(
        go.Scatter(
            x=np.concatenate([degrees, degrees[::-1]]),
            y=np.concatenate([cv_mean + cv_std, cv_lower[::-1]]),
            fill="toself",
            fillcolor=getattr(theme, "GOOD_FADE", "rgba(62,112,99,0.16)"),
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            name="CV ± 1 SD",
        )
    )
    error_fig.add_trace(
        go.Scatter(
            x=degrees,
            y=cv_mean,
            mode="lines+markers",
            name="CV RMSE",
            line=dict(color=theme.GOOD, width=3, dash="dash"),
        )
    )
    error_fig.add_vline(x=degree, line=dict(color=theme.MODEL, dash="dot"))
    theme.style_fig(error_fig, height=340, title="Training and cross-validation error")
    error_fig.update_xaxes(title="Polynomial degree", dtick=1)
    error_fig.update_yaxes(title="RMSE")
    st.plotly_chart(
        error_fig,
        width="stretch",
        key="complexity_error",
        config={"displayModeBar": False},
    )
    selected_row = sweep.loc[sweep["degree"] == degree].iloc[0]
    summary = st.columns(3)
    summary[0].metric("Training RMSE", f"{selected_row['train_RMSE']:.2f}")
    summary[1].metric(
        "CV RMSE",
        f"{selected_row['CV_RMSE_mean']:.2f} ± {selected_row['CV_RMSE_std']:.2f}",
    )
    summary[2].metric("Locked final-test RMSE", f"{result['test']['RMSE']:.2f}")
    st.caption(
        "The sweep uses training-only cross-validation. The final-test value is shown only "
        "for the degree you committed."
    )


def regularisation_lab() -> None:
    dataset_name = st.selectbox(
        "Dataset",
        ru.MULTI_FEATURE_DATASETS,
        key="regularisation_dataset",
    )
    df, features, target = get_dataset(dataset_name)
    model = st.radio(
        "Model",
        ["Lasso", "Ridge"],
        horizontal=True,
        key="regularisation_model",
    )
    if model == "Lasso":
        theme.challenge(
            "Raise α until some Lasso coefficients reach zero. Does cross-validation improve?"
        )
    else:
        theme.challenge(
            "Raise α and watch Ridge shrink coefficients smoothly. Where is CV performance strongest?"
        )
    exponent = st.slider(
        "log₁₀ α",
        -2.0,
        3.0,
        0.0,
        0.25,
        key="regularisation_alpha",
    )
    alpha = float(10**exponent)
    zeroed = ru.n_zeroed(df, features, target, alpha, model=model)
    st.metric(
        "Coefficients effectively at zero",
        f"{zeroed} / {len(features)}",
        help="Lasso can set coefficients to zero; Ridge usually shrinks without zeroing.",
    )

    path = cached_coefficient_path(dataset_name, model)
    path["feature"] = path["feature"].map(friendly)
    path_fig = px.line(
        path,
        x="alpha",
        y="coefficient",
        color="feature",
        color_discrete_sequence=getattr(
            theme,
            "CHART_SEQUENCE",
            [theme.DATA, theme.MODEL, theme.GOOD, theme.NEUTRAL, theme.ERROR],
        ),
        title=f"{model} coefficient paths",
    )
    path_fig.update_xaxes(type="log", title="α · log scale")
    path_fig.add_vline(x=alpha, line=dict(color=theme.INK_SOFT, dash="dot"))
    theme.style_fig(path_fig, height=340)
    st.plotly_chart(
        path_fig,
        width="stretch",
        key="regularisation_path",
        config={"displayModeBar": False},
    )

    sweep = cached_alpha_sweep(dataset_name, model)
    cv_fig = go.Figure()
    cv_mean = sweep["CV_RMSE_mean"].to_numpy()
    cv_std = sweep["CV_RMSE_std"].to_numpy()
    alphas = sweep["alpha"].to_numpy()
    cv_lower = np.maximum(cv_mean - cv_std, 0)
    cv_fig.add_trace(
        go.Scatter(
            x=np.concatenate([alphas, alphas[::-1]]),
            y=np.concatenate([cv_mean + cv_std, cv_lower[::-1]]),
            fill="toself",
            fillcolor=getattr(theme, "GOOD_FADE", "rgba(62,112,99,0.16)"),
            line=dict(color="rgba(0,0,0,0)"),
            hoverinfo="skip",
            name="CV ± 1 SD",
        )
    )
    cv_fig.add_trace(
        go.Scatter(
            x=alphas,
            y=cv_mean,
            mode="lines+markers",
            name="CV RMSE",
            line=dict(color=theme.GOOD, width=3),
        )
    )
    cv_fig.add_vline(x=alpha, line=dict(color=theme.MODEL, dash="dot"))
    theme.style_fig(cv_fig, height=310, title="Training-only cross-validation across α")
    cv_fig.update_xaxes(type="log", title="α · log scale")
    cv_fig.update_yaxes(title="CV RMSE")
    st.plotly_chart(
        cv_fig,
        width="stretch",
        key="regularisation_cv",
        config={"displayModeBar": False},
    )

    locked_split = ru.make_locked_split(df, features, target)
    result = ru.fit_and_evaluate(
        model,
        df,
        features,
        target,
        alpha=alpha,
        locked_split=locked_split,
    )
    cv = ru.cross_validate_metrics(
        model,
        df,
        features,
        target,
        k=5,
        locked_split=locked_split,
        training_only=True,
        alpha=alpha,
    )
    summary = st.columns(3)
    summary[0].metric("Training R²", f"{result['train']['R2']:.3f}")
    summary[1].metric("CV R²", cv_text(cv, "R2"))
    summary[2].metric("CV RMSE", cv_text(cv, "RMSE", digits=2))

    signature = state_signature(dataset_name, model, alpha)
    reveal = reveal_gate(
        "regularisation_final_reveal",
        signature,
        "Lock α and reveal the final test",
    )
    if reveal:
        st.success(
            f"Locked final test · R² {result['test']['R2']:.3f} · "
            f"RMSE {result['test']['RMSE']:.2f}"
        )


def workspace_generalisation() -> None:
    page_intro(
        "Generalisation & Regularisation",
        "Control model complexity without learning the noise",
    )
    section = st.radio(
        "Choose investigation",
        ["Polynomial complexity", "Ridge & Lasso"],
        horizontal=True,
        key="generalisation_section",
    )
    if section == "Polynomial complexity":
        complexity_lab()
    else:
        regularisation_lab()


# ---------------------------------------------------------------------------
# Workspace 6: Diagnostics
# ---------------------------------------------------------------------------
def workspace_diagnose() -> None:
    page_intro(
        "Diagnose",
        "Use residual patterns as visual clues, not automatic pass/fail tests",
        "Inspect all linked views before deciding which assumption deserves attention.",
    )
    selected_label = st.selectbox(
        "Mystery dataset",
        list(DIAGNOSTIC_SCENARIOS),
        key="diagnostic_mystery",
    )
    scenario = DIAGNOSTIC_SCENARIOS[selected_label]
    prediction = st.selectbox(
        "Most important visual clue",
        [
            "No strong visual warning",
            "Non-linearity",
            "Non-constant variance",
            "An unusual response",
            "A high-leverage or influential point",
            "Non-normal residual shape",
            "Dependence across observation order",
        ],
        key="diagnostic_prediction",
    )
    df = ru.make_diagnostic_scenario(scenario, n=100, seed=ru.RANDOM_STATE)
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    slope, intercept = ru.ols_1d(x, y)
    fitted = intercept + slope * x
    residual = y - fitted

    chart_columns = st.columns(2, gap="large")
    with chart_columns[0]:
        relationship_fig = go.Figure()
        regular = ~df["is_special"].to_numpy(dtype=bool)
        relationship_fig.add_trace(
            go.Scatter(
                x=x[regular],
                y=y[regular],
                mode="markers",
                marker=dict(color=theme.DATA, size=7, opacity=0.72),
                name="observations",
            )
        )
        if (~regular).any():
            relationship_fig.add_trace(
                go.Scatter(
                    x=x[~regular],
                    y=y[~regular],
                    mode="markers",
                    marker=dict(
                        color=theme.ERROR,
                        symbol="x",
                        size=12,
                        line=dict(width=2),
                    ),
                    name="highlighted observation",
                )
            )
        x_line = np.linspace(float(x.min()), float(x.max()), 200)
        relationship_fig.add_trace(
            go.Scatter(
                x=x_line,
                y=intercept + slope * x_line,
                mode="lines",
                line=dict(color=theme.MODEL, width=3),
                name="linear fit",
            )
        )
        theme.style_fig(relationship_fig, height=300, title="Observed relationship")
        relationship_fig.update_xaxes(title="Input x")
        relationship_fig.update_yaxes(title="Target y")
        st.plotly_chart(
            relationship_fig,
            width="stretch",
            key="diagnostic_relationship",
            config={"displayModeBar": False},
        )
    with chart_columns[1]:
        residual_fig = go.Figure(
            go.Scatter(
                x=fitted,
                y=residual,
                mode="markers",
                marker=dict(color=theme.DATA, size=7, opacity=0.72),
                name="residuals",
            )
        )
        residual_fig.add_hline(y=0, line=dict(color=theme.ERROR, dash="dash"))
        theme.style_fig(residual_fig, height=300, title="Residuals versus predicted")
        residual_fig.update_xaxes(title="Predicted")
        residual_fig.update_yaxes(title="Residual")
        st.plotly_chart(
            residual_fig,
            width="stretch",
            key="diagnostic_residual",
            config={"displayModeBar": False},
        )

    distribution_columns = st.columns(3, gap="large")
    with distribution_columns[0]:
        histogram = px.histogram(
            x=residual,
            nbins=18,
            labels={"x": "Residual"},
            title="Residual distribution",
        )
        histogram.update_traces(marker_color=theme.MODEL)
        theme.style_fig(histogram, height=280)
        st.plotly_chart(
            histogram,
            width="stretch",
            key="diagnostic_histogram",
            config={"displayModeBar": False},
        )
    with distribution_columns[1]:
        theoretical, ordered = stats.probplot(residual, dist="norm", fit=False)
        qq_fig = go.Figure(
            go.Scatter(
                x=theoretical,
                y=ordered,
                mode="markers",
                marker=dict(color=theme.NEUTRAL, size=6),
                name="residuals",
            )
        )
        line_fit = np.polyfit(theoretical, ordered, 1)
        xx = np.asarray([min(theoretical), max(theoretical)])
        qq_fig.add_trace(
            go.Scatter(
                x=xx,
                y=line_fit[0] * xx + line_fit[1],
                mode="lines",
                line=dict(color=theme.INK_SOFT, dash="dash"),
                name="reference",
            )
        )
        theme.style_fig(qq_fig, height=280, title="Normal Q–Q view")
        qq_fig.update_xaxes(title="Theoretical quantile")
        qq_fig.update_yaxes(title="Ordered residual")
        st.plotly_chart(
            qq_fig,
            width="stretch",
            key="diagnostic_qq",
            config={"displayModeBar": False},
        )
    with distribution_columns[2]:
        order_fig = go.Figure(
            go.Scatter(
                x=np.arange(1, len(residual) + 1),
                y=residual,
                mode="lines+markers",
                marker=dict(color=theme.DATA, size=5),
                line=dict(color=theme.DATA, width=1),
                name="residuals",
            )
        )
        order_fig.add_hline(y=0, line=dict(color=theme.ERROR, dash="dash"))
        theme.style_fig(order_fig, height=280, title="Residuals by observation order")
        order_fig.update_xaxes(title="Observation order")
        order_fig.update_yaxes(title="Residual")
        st.plotly_chart(
            order_fig,
            width="stretch",
            key="diagnostic_order",
            config={"displayModeBar": False},
        )

    signature = state_signature(scenario, prediction)
    reveal = reveal_gate(
        "diagnostic_reveal",
        signature,
        "Lock diagnosis and reveal the intended clue",
    )
    if reveal:
        expected = DIAGNOSTIC_CLUES[scenario]
        if prediction == expected:
            st.success(f"Good diagnosis: **{expected}** is the intended primary clue.")
        else:
            st.info(
                f"The intended primary clue is **{expected}**. Your choice may still describe "
                "a secondary pattern; diagnostics are evidence, not proof."
            )
        clue_explanations = {
            "clean": {
                "why": (
                    "The target was generated from a straight-line relationship with "
                    "roughly constant, symmetric random noise."
                ),
                "figures": (
                    "**Residuals versus predicted** and **Residuals by observation order**"
                ),
                "inspect": (
                    "Look for an unstructured band scattered around zero: no curve, no "
                    "widening funnel, and no long runs on one side of zero. In **Normal Q–Q "
                    "view**, most points should also stay reasonably close to the reference line."
                ),
            },
            "curved": {
                "why": (
                    "A quadratic component was added to the straight-line signal, so one "
                    "linear fit cannot follow the centre of the data across the full x range."
                ),
                "figures": "**Observed relationship** and **Residuals versus predicted**",
                "inspect": (
                    "In **Observed relationship**, compare the bowed point cloud with the "
                    "straight fitted line. In **Residuals versus predicted**, trace the clear "
                    "U-shaped pattern: positive residuals near both ends and negative residuals "
                    "through the middle."
                ),
            },
            "funnel": {
                "why": (
                    "The noise spread was deliberately increased from low x to high x, while "
                    "the average relationship remained linear."
                ),
                "figures": "**Residuals versus predicted**",
                "inspect": (
                    "Read the plot from left to right. The residuals form a narrow vertical "
                    "band at low predictions and a much wider band at high predictions—the "
                    "funnel is about changing spread, not a curved average."
                ),
            },
            "outlier": {
                "why": (
                    "One observation near the middle of the x range was moved far upward from "
                    "the otherwise linear data."
                ),
                "figures": "**Observed relationship** and **Residuals versus predicted**",
                "inspect": (
                    "Find the highlighted red × near x = 0 in **Observed relationship**; it is "
                    "well above the fitted line. The matching point is the single very large "
                    "positive residual in **Residuals versus predicted**."
                ),
            },
            "high_leverage": {
                "why": (
                    "One observation was placed at x = 9, far beyond the main x range, with "
                    "a target of −10 that conflicts with the main linear trend. Its unusual x "
                    "position gives it leverage to pull the fitted line."
                ),
                "figures": "**Observed relationship**",
                "inspect": (
                    "Find the highlighted red × isolated at the far right. Check both its large "
                    "horizontal separation from every other point and the way the fitted line "
                    "tilts toward it; a large residual alone would describe an outlier, but the "
                    "extreme x position is what makes this point high leverage."
                ),
            },
            "skewed": {
                "why": (
                    "The errors were generated with a centred exponential distribution, which "
                    "keeps a long right tail instead of a symmetric normal shape."
                ),
                "figures": "**Residual distribution** and **Normal Q–Q view**",
                "inspect": (
                    "In **Residual distribution**, look for many residuals near the centre and "
                    "a sparse tail stretching much farther to the right. In **Normal Q–Q view**, "
                    "the upper-quantile points bend above and away from the reference line."
                ),
            },
            "dependent": {
                "why": (
                    "Each error was generated to resemble the preceding error, so neighbouring "
                    "residuals are related even though the x values are sampled independently "
                    "rather than arranged in observation order."
                ),
                "figures": "**Residuals by observation order**",
                "inspect": (
                    "Follow adjacent points rather than individual extremes. Look for smooth "
                    "stretches and long runs above or below zero; independent noise would switch "
                    "direction and cross zero more irregularly."
                ),
            },
        }
        explanation = clue_explanations[scenario]
        st.markdown("#### What the generated data is showing")
        st.markdown(f"**Why this clue fits:** {explanation['why']}")
        st.markdown(f"**Strongest figure(s):** {explanation['figures']}")
        st.markdown(f"**What to inspect:** {explanation['inspect']}")
        st.caption(
            "A plot cannot prove that an assumption holds. It helps identify patterns worth "
            "investigating alongside subject knowledge and additional tests."
        )


# ---------------------------------------------------------------------------
# Navigation, experiment history and main entry point
# ---------------------------------------------------------------------------
def render_sidebar() -> None:
    sidebar = st.sidebar
    sidebar.markdown("### My Experiments")
    history = st.session_state.get("history", [])
    sidebar.caption(f"{len(history)} logged result{'s' if len(history) != 1 else ''}")
    if history:
        frame = pd.DataFrame(history)
        with sidebar.expander("View history", expanded=False):
            sidebar.dataframe(frame, width="stretch", hide_index=True, height=260)
        sidebar.download_button(
            "Download CSV",
            frame.to_csv(index=False).encode(),
            "regression_experiments.csv",
            "text/csv",
            key="history_download",
        )
        if sidebar.button("Clear history", key="history_clear"):
            st.session_state["history"] = []
            st.rerun()
    else:
        sidebar.caption("Log results from Fit the Line or Model Arena.")
    sidebar.divider()
    sidebar.caption(
        "Final test is locked. Model and feature choices use training-only cross-validation."
    )


def main() -> None:
    theme.inject_css()
    st.sidebar.markdown("## Workspaces")
    workspace = st.sidebar.radio(
        "Choose a workspace",
        WORKSPACES,
        key="workspace",
        label_visibility="collapsed",
    )
    st.sidebar.caption("Move between investigations at any time.")
    st.sidebar.divider()

    st.markdown(
        """
        <div class="rp-app-kicker">UNITEC MACHINE LEARNING · WEEK 1 · SESSION 2</div>
        <div class="rp-app-name">Regression Playground</div>
        <div class="rp-app-tagline">Fit · Break · Diagnose · Improve</div>
        """,
        unsafe_allow_html=True,
    )

    workspace_functions = {
        WORKSPACES[0]: workspace_fit_line,
        WORKSPACES[1]: workspace_gradient_descent,
        WORKSPACES[2]: workspace_feature_lab,
        WORKSPACES[3]: workspace_model_arena,
        WORKSPACES[4]: workspace_generalisation,
        WORKSPACES[5]: workspace_diagnose,
    }
    workspace_functions[workspace]()
    render_sidebar()
    st.markdown(
        '<div class="rp-footer">Regression Playground · Unitec ML Course · '
        "training-only model selection · locked final test</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
