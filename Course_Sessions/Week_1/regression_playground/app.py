"""
Regression Playground  ·  Fit, Break, Diagnose, Improve
=======================================================
An interactive laboratory for Week 1 Session 2 (Regression Techniques and Model
Evaluation), Unitec ML Course.

Run:
    streamlit run app.py

Six workspaces:
  1. Fit the Line          - hand-fit a line, then reveal ordinary least squares
  2. Gradient Descent      - watch the machine fit the line automatically
  3. Metrics Under Pressure- MAE / MSE / RMSE / R2 / MAPE and how outliers move them
  4. Feature Lab           - the FuelConsumption CO2 activity, made interactive
  5. Model Arena           - compare two models on the same locked split
  6. Overfitting & Regularisation - complexity, cross-validation, Ridge / Lasso

Design, notation (theta / alpha / J(theta)) and palette follow the lecture slides.
House style: no emojis.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Make sibling modules importable no matter how the app is launched
# (streamlit run adds the script dir automatically; this covers other launchers).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import regression_utils as ru
import theme

st.set_page_config(page_title="Regression Playground", page_icon="=",
                   layout="wide", initial_sidebar_state="expanded")

METRIC_HELP = {
    "MAE": "mean(|y - y_hat|)  ·  robust to outliers  ·  target units",
    "MSE": "mean((y - y_hat)^2)  ·  punishes big errors  ·  squared units",
    "RMSE": "sqrt(MSE)  ·  same units as the target",
    "R2": "1 - SSres/SStot  ·  1 is perfect, 0 = mean baseline, can go negative",
    "MAPE": "mean(|(y - y_hat)/y|) * 100  ·  percent, scale-free (watch zeros)",
}
CORR_SCALE = [[0.0, "#2C4E79"], [0.25, "#A8BFD6"], [0.5, "#EDE8DA"],
              [0.75, "#D4A886"], [1.0, "#8B3A14"]]


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_fuel() -> pd.DataFrame:
    return ru.load_fuel_consumption()


@st.cache_data(show_spinner=False)
def load_energy() -> pd.DataFrame:
    return ru.load_energy_efficiency()


@st.cache_data(show_spinner=False)
def cached_combo_ranking(features_key: tuple, max_k: int) -> pd.DataFrame:
    df = load_fuel()
    return ru.rank_feature_combos(df, list(features_key), ru.FUEL_TARGET, max_k=max_k)


@st.cache_data(show_spinner=False)
def cached_degree_sweep(n: int, noise: float):
    """Train / test / CV error across polynomial degrees 1..12 (deterministic)."""
    df = ru.make_synthetic(n=n, noise=noise, curvature=0.6)
    degs = list(range(1, 13))
    tr, te, cv = [], [], []
    for d in degs:
        r = ru.fit_and_evaluate("Polynomial", df, ["x"], "y", degree=d)
        tr.append(r["train"]["RMSE"])
        te.append(r["test"]["RMSE"])
        cv.append(-ru.cross_validate_r2("Polynomial", df, ["x"], "y", k=5, degree=d)["mean"])
    return degs, tr, te, cv


@st.cache_data(show_spinner=False)
def cached_coef_path(ds_name: str, model: str) -> pd.DataFrame:
    df, feats, target = get_multi_dataset(ds_name)
    return ru.coefficient_path(df, feats, target, model=model)


def get_multi_dataset(name: str):
    """Return (df, features, target) for the multi-dataset workspaces."""
    if name == "FuelConsumption CO2":
        return load_fuel(), ru.FUEL_FEATURES, ru.FUEL_TARGET
    if name == "Energy Efficiency":
        return load_energy(), ru.ENERGY_FEATURES, ru.ENERGY_TARGET
    df = ru.make_synthetic(n=120, noise=10, curvature=0.4)
    return df, ["x"], "y"


MULTI_DATASETS = ["FuelConsumption CO2", "Energy Efficiency", "Synthetic sandbox"]


# ---------------------------------------------------------------------------
# Shared UI helpers
# ---------------------------------------------------------------------------
def metric_cards(metrics: dict, prefix: str) -> None:
    order = ["MAE", "MSE", "RMSE", "R2", "MAPE"]
    cols = st.columns(5)
    for c, key in zip(cols, order):
        val = metrics.get(key, float("nan"))
        if key == "R2":
            txt = f"{val:.3f}"
        elif key == "MAPE":
            txt = "n/a" if (val != val) else f"{val:.1f}%"
        else:
            txt = f"{val:,.2f}"
        label = "R²" if key == "R2" else key
        c.metric(label, txt, help=METRIC_HELP[key])


def reveal(label: str, guided: bool):
    """An expander that starts collapsed in Guided mode, open in Playground."""
    return st.expander(label, expanded=not guided)


def scatter_line_fig(x, y, lines: list[dict], height=380,
                     xtitle="x", ytitle="y", residual_from=None):
    """Scatter of (x,y) plus any number of overlaid lines.

    lines: list of {"slope","intercept","name","color"}.
    residual_from: optional {"slope","intercept","color"} to draw drop-lines.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    fig = go.Figure()
    if residual_from is not None:
        pred = residual_from["intercept"] + residual_from["slope"] * x
        rx, ry = [], []
        for xi, yi, pi in zip(x, y, pred):
            rx += [xi, xi, None]
            ry += [yi, pi, None]
        fig.add_trace(go.Scatter(x=rx, y=ry, mode="lines",
                                 line=dict(color=residual_from["color"], width=1, dash="dot"),
                                 name="residuals", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name="data",
                             marker=dict(color=theme.DATA, size=8,
                                         line=dict(color=theme.PALETTE["data_edge"], width=1))))
    xs = np.linspace(x.min(), x.max(), 100)
    for ln in lines:
        fig.add_trace(go.Scatter(x=xs, y=ln["intercept"] + ln["slope"] * xs,
                                 mode="lines", name=ln["name"],
                                 line=dict(color=ln["color"], width=3)))
    theme.style_fig(fig, height=height)
    fig.update_xaxes(title=xtitle)
    fig.update_yaxes(title=ytitle)
    return fig


def actual_vs_pred_fig(y_true, y_pred, height=360, color=None):
    color = color or theme.MODEL
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="perfect",
                             line=dict(color=theme.INK_SOFT, width=1.5, dash="dash")))
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers", name="predictions",
                             marker=dict(color=color, size=7, opacity=0.7,
                                         line=dict(color=theme.PALETTE["data_edge"], width=0.5))))
    theme.style_fig(fig, height=height)
    fig.update_xaxes(title="actual")
    fig.update_yaxes(title="predicted")
    return fig


def log_experiment(row: dict) -> None:
    st.session_state.setdefault("history", []).append(row)


# ===========================================================================
# Tab 1 - Fit the Line
# ===========================================================================
def tab_fit_line(guided: bool):
    theme.header("Fit the Line",
                 "Move the line by hand to make the errors small, then reveal the best-fit line")
    st.caption("This tab uses a simple dataset with one input and one output, so you can "
               "see a single straight line. Pick your data on the left.")
    theme.challenge("Move the two sliders so the line goes through the middle of the dots. "
                    "Then reveal the computer's best-fit line and compare.")

    st.latex(r"\hat{y} = \theta_0 + \theta_1\,x \qquad\text{(intercept } \theta_0"
             r"\text{, slope } \theta_1\text{)}")

    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        source = st.radio("Choose your data",
                          ["Practice data (you control it)", "Famous demo (Anscombe)"],
                          key="fl_src")
        if source.startswith("Practice"):
            n = st.slider("How many dots", 10, 200, 40, key="fl_n")
            noise = st.slider("How scattered (noise)", 0.0, 20.0, 6.0, 0.5, key="fl_noise")
            seed = st.number_input("Shuffle (seed)", 0, 9999, 42, key="fl_seed",
                                   help="Change this to get a different random scatter.")
            df = ru.make_synthetic(n=n, noise=noise, seed=int(seed))
            add_outlier = st.checkbox("Add one odd point (outlier)", key="fl_out")
            if add_outlier:
                df = pd.concat([df, pd.DataFrame({"x": [df["x"].max()],
                                                  "y": [df["y"].min() - 3 * noise - 10]})],
                               ignore_index=True)
        else:
            which = st.selectbox("Which of the four sets", ["I", "II", "III", "IV"],
                                 key="fl_ans")
            df = ru.make_anscombe(which)
            st.markdown('<span class="rp-note">The four "Anscombe" sets are a famous demo: '
                        'they have almost identical statistics but look completely different '
                        'once you plot them.</span>', unsafe_allow_html=True)

        x, y = df["x"].to_numpy(), df["y"].to_numpy()
        ols_slope, ols_intercept = ru.ols_1d(x, y)
        span = float(y.max() - y.min()) or 1.0
        slope = st.slider("Slope (how steep the line is)", -5.0, 5.0, 1.0, 0.05,
                          key="fl_slope", help="Called theta-one in the equation.")
        # Fixed range so a stored value can never fall outside the bounds when the
        # underlying data changes (which would raise a Streamlit slider error).
        intercept = st.slider("Intercept (where the line starts)", -60.0, 60.0,
                              float(np.clip(y.mean(), -60, 60)), 0.1, key="fl_intercept",
                              help="Where the line crosses the vertical axis. "
                                   "Called theta-zero in the equation.")

    with c2:
        student_sse, student_mse = ru.sse_mse(x, y, slope, intercept)
        lines = [{"slope": slope, "intercept": intercept, "name": "your line",
                  "color": theme.MODEL}]
        show_ols = reveal("Reveal the best-fit line (the computer's exact answer)", guided)
        with show_ols:
            lines.append({"slope": ols_slope, "intercept": ols_intercept,
                          "name": "best-fit line", "color": theme.GOOD})
            ols_sse, ols_mse = ru.sse_mse(x, y, ols_slope, ols_intercept)
            matched = abs(student_mse - ols_mse) < 0.01 * ols_mse + 1e-9
            st.markdown(
                f"The best-fit line has slope **{ols_slope:.3f}** and intercept "
                f"**{ols_intercept:.3f}**, giving an average error of **{ols_mse:.2f}**. "
                f"Your line's average error is **{student_mse:.2f}** — "
                f"{'you matched it!' if matched else f'about {student_mse/ols_mse:.1f} times larger.'}"
            )
            resid = np.abs(y - (ols_intercept + ols_slope * x))
            k = int(np.argmax(resid))
            st.markdown(f'<span class="rp-note">The dot furthest from the line '
                        f'(x={x[k]:.2f}, y={y[k]:.2f}) pulls it the hardest.</span>',
                        unsafe_allow_html=True)

        fig = scatter_line_fig(x, y, lines,
                               residual_from={"slope": slope, "intercept": intercept,
                                              "color": theme.ERROR})
        st.plotly_chart(fig, use_container_width=True, key="fl_fig",
                        config={"displayModeBar": False})
        st.caption("The red dotted lines are the errors: the gap from each dot up or down "
                   "to your line. The goal is to make them as small as possible overall.")

        m1, m2, m3 = st.columns(3)
        m1.metric("Your total error", f"{student_sse:,.1f}",
                  help="Sum of the squared gaps (SSE). Lower is better.")
        m2.metric("Your average error", f"{student_mse:,.2f}",
                  help="Mean squared error (MSE) — the total shared over all dots.")
        m3.metric("Best-fit average error", f"{ru.sse_mse(x, y, ols_slope, ols_intercept)[1]:,.2f}",
                  help="The smallest average error possible for a straight line.")

        pc1, pc2 = st.columns([2, 1])
        px_val = pc1.slider("Predict the output when the input =",
                            float(x.min()), float(x.max()),
                            float(np.median(x)), key=f"fl_predx_{source}")
        pc2.metric("Your line predicts", f"{intercept + slope * px_val:.2f}")

        if st.button("Log this attempt", key="fl_log"):
            log_experiment({"tab": "Fit the Line", "slope": round(slope, 3),
                            "intercept": round(intercept, 3),
                            "your_MSE": round(student_mse, 3),
                            "OLS_MSE": round(ru.sse_mse(x, y, ols_slope, ols_intercept)[1], 3)})
            st.success("Logged to experiment history.")


# ===========================================================================
# Tab 2 - Gradient Descent
# ===========================================================================
def tab_gradient_descent(guided: bool):
    theme.header("Gradient Descent",
                 "The same line - but now the machine finds it by walking downhill on the cost")
    st.caption("Uses made-up practice data (set the amount and scatter below). The computer "
               "starts from a random line and improves it step by step.")
    theme.challenge("Predict what happens with a tiny, a good, and a huge learning rate α. "
                    "Then run each and watch the cost curve.")
    lc, rc = st.columns(2)
    with lc:
        st.markdown("**Update rule** (take a step downhill):")
        st.latex(r"\theta \;\leftarrow\; \theta \;-\; \alpha\,\frac{\partial J}{\partial \theta}")
    with rc:
        st.markdown("**Cost** (half the mean squared error):")
        st.latex(r"J(\theta) = \frac{1}{2m}\sum_{i=1}^{m}\bigl(h_\theta(x^{(i)}) - y^{(i)}\bigr)^2")

    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        n = st.slider("Points", 20, 200, 60, key="gd_n")
        noise = st.slider("Noise", 0.0, 20.0, 8.0, 0.5, key="gd_noise")
        df = ru.make_synthetic(n=n, noise=noise)
        x, y = df["x"].to_numpy(), df["y"].to_numpy()

        preset = st.radio("Learning rate α", ["Too small (0.01)", "Just right (0.3)",
                                             "Too big (2.5)", "Custom"], index=1, key="gd_preset")
        preset_alpha = {"Too small (0.01)": 0.01, "Just right (0.3)": 0.3,
                        "Too big (2.5)": 2.5}
        if preset == "Custom":
            alpha = st.slider("α (learning rate)", 0.001, 3.0, 0.3, 0.001, key="gd_alpha")
        else:
            alpha = preset_alpha[preset]
        n_iters = st.slider("Iterations", 10, 500, 150, key="gd_iters")

    gd = ru.gradient_descent_1d(x, y, alpha=alpha, n_iters=n_iters)
    ols_slope, ols_intercept = ru.ols_1d(x, y)

    with c2:
        if gd["diverged"]:
            st.warning("This learning rate is too large - the cost is growing, not "
                       "shrinking. Gradient descent is diverging. Try a smaller α.")
        it = st.slider("Scrub through iterations", 0, len(gd["cost"]) - 1,
                       len(gd["cost"]) - 1, key=f"gd_scrub_{n_iters}")
        # line at iteration `it`, converted back to raw-x space
        t0, t1 = gd["theta0"][it], gd["theta1"][it]
        slope_it = t1 / gd["x_sd"]
        intercept_it = t0 - t1 * gd["x_mu"] / gd["x_sd"]
        lines = [{"slope": slope_it, "intercept": intercept_it,
                  "name": f"GD step {it}", "color": theme.MODEL},
                 {"slope": ols_slope, "intercept": ols_intercept,
                  "name": "OLS target", "color": theme.GOOD}]
        st.plotly_chart(scatter_line_fig(x, y, lines), use_container_width=True,
                        key="gd_line", config={"displayModeBar": False})

    # cost curve
    cost = np.array(gd["cost"], dtype=float)
    cost_plot = np.where(np.isfinite(cost), cost, np.nan)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=cost_plot, mode="lines", name="cost J",
                             line=dict(color=theme.DATA, width=2.5)))
    fig.add_trace(go.Scatter(x=[it], y=[cost_plot[it] if np.isfinite(cost_plot[it]) else None],
                             mode="markers", name="current",
                             marker=dict(color=theme.ERROR, size=11)))
    theme.style_fig(fig, height=280, title="Cost J(θ) vs iteration")
    fig.update_xaxes(title="iteration")
    fig.update_yaxes(title="cost J")
    if np.isfinite(cost_plot).sum() > 1 and np.nanmax(cost_plot) / max(np.nanmin(cost_plot), 1e-9) > 50:
        fig.update_yaxes(type="log")
    st.plotly_chart(fig, use_container_width=True, key="gd_cost",
                    config={"displayModeBar": False})
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Start cost", f"{cost[0]:.2f}")
    cc2.metric("Current cost", f"{cost[it]:.2f}" if np.isfinite(cost[it]) else "inf")
    cc3.metric("GD slope vs OLS", f"{slope_it:.3f} / {ols_slope:.3f}")


# ===========================================================================
# Tab 3 - Metrics Under Pressure
# ===========================================================================
def tab_metrics(guided: bool):
    theme.header("Metrics Under Pressure",
                 "Same predictions, different lenses - and what one outlier does to each metric")
    st.caption("Uses made-up practice data. We fit the best line for you, then you can add "
               "one bad point and watch how each error score reacts.")
    theme.challenge("Inject one extreme error. Predict which metric will move the most, "
                    "then check. (Hint: think about squaring.)")

    with st.expander("Show the formulas", expanded=False):
        fc1, fc2 = st.columns(2)
        with fc1:
            st.latex(r"\mathrm{MAE} = \frac{1}{n}\sum_{i=1}^{n}\lvert y_i - \hat{y}_i\rvert")
            st.latex(r"\mathrm{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2")
            st.latex(r"\mathrm{RMSE} = \sqrt{\mathrm{MSE}}")
        with fc2:
            st.latex(r"R^2 = 1 - \frac{\sum_i (y_i-\hat{y}_i)^2}{\sum_i (y_i-\bar{y})^2}")
            st.latex(r"\mathrm{MAPE} = \frac{100}{n}\sum_{i=1}^{n}"
                     r"\left\lvert\frac{y_i-\hat{y}_i}{y_i}\right\rvert")

    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        n = st.slider("Points", 20, 200, 60, key="mt_n")
        noise = st.slider("Noise", 1.0, 20.0, 7.0, 0.5, key="mt_noise")
        df = ru.make_synthetic(n=n, noise=noise)
        x, y = df["x"].to_numpy(), df["y"].to_numpy()
        slope, intercept = ru.ols_1d(x, y)
        pred = intercept + slope * x
        inject = st.checkbox("Inject one extreme error", key="mt_inject")
        strength = st.slider("Outlier strength", 1.0, 8.0, 4.0, 0.5, key="mt_strength",
                             disabled=not inject)

    y_dirty = y.copy()
    if inject:
        k = int(np.argmax(np.abs(x - np.median(x))))  # a point out on the edge
        y_dirty[k] = y[k] + strength * (y.max() - y.min())

    base = ru.all_metrics(y, pred)
    dirty = ru.all_metrics(y_dirty, pred)
    mean_pred = np.full_like(y, y.mean())
    baseline = ru.all_metrics(y_dirty, mean_pred)

    with c2:
        st.markdown("**Metrics on the clean data**")
        metric_cards(base, "clean")
        if inject:
            st.markdown("**After injecting the outlier** (change vs clean)")
            cols = st.columns(5)
            for col, key in zip(cols, ["MAE", "MSE", "RMSE", "R2", "MAPE"]):
                v, b = dirty[key], base[key]
                if key == "MAPE" and (v != v or b != b):
                    col.metric(key, "n/a")
                    continue
                delta = v - b
                label = "R²" if key == "R2" else key
                txt = f"{v:.3f}" if key == "R2" else f"{v:,.2f}"
                col.metric(label, txt, delta=f"{delta:+,.2f}",
                           delta_color="inverse" if key != "R2" else "normal")
            st.markdown('<span class="rp-note">MSE and RMSE jump the most because the '
                        'error is squared; MAE barely moves. That is why MAE is called '
                        'robust to outliers.</span>', unsafe_allow_html=True)

        with reveal("Mean-prediction baseline and negative R-squared", guided):
            st.markdown(
                f"If the model just predicted the mean every time, R-squared would be "
                f"**0** by definition. Here the mean baseline scores R-squared = "
                f"**{baseline['R2']:.3f}**. A value **below 0** means a model is doing "
                f"*worse than guessing the mean*."
            )

    # error contribution chart
    contrib = (y_dirty - pred) ** 2
    order = np.argsort(contrib)[::-1][:12]
    fig = go.Figure(go.Bar(x=[f"pt {i}" for i in order], y=contrib[order],
                           marker_color=theme.PALETTE["error"]))
    theme.style_fig(fig, height=260, title="Top squared-error contributors")
    fig.update_yaxes(title="squared error")
    st.plotly_chart(fig, use_container_width=True, key="mt_contrib",
                    config={"displayModeBar": False})


# ===========================================================================
# Tab 4 - Feature Lab
# ===========================================================================
def tab_feature_lab(guided: bool):
    theme.header("Feature Lab",
                 "Explore FuelConsumption CO2 - which features actually drive the prediction")
    st.caption("This activity uses the real FuelConsumption CO2 dataset (fixed for this "
               "exercise): car engine details used to predict a car's CO2 emissions.")
    theme.challenge("Build the best CO2 model you can using no more than three features. "
                    "Commit your choice, then reveal the ranking of all combinations.")
    df = load_fuel()
    feats = ru.FUEL_FEATURES
    target = ru.FUEL_TARGET

    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        corr = df[feats + [target]].corr()
        fig = px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1,
                        color_continuous_scale=CORR_SCALE, aspect="auto")
        theme.style_fig(fig, height=380, title="Correlation matrix (r)")
        st.plotly_chart(fig, use_container_width=True, key="flab_corr",
                        config={"displayModeBar": False})

    with c2:
        chosen = st.multiselect("Features (choose 1 to 3)", feats,
                                default=["ENGINESIZE"], key="flab_feats",
                                max_selections=3)
        if not chosen:
            st.info("Select at least one feature.")
            return
        single = ru.fit_and_evaluate("Linear", df, [chosen[0]], target)
        multi = ru.fit_and_evaluate("Linear", df, chosen, target)
        b1, b2 = st.columns(2)
        b1.metric(f"1 feature ({chosen[0]})", f"R² {single['test']['R2']:.3f}",
                  help="single-feature baseline (test)")
        b2.metric(f"{len(chosen)} features", f"R² {multi['test']['R2']:.3f}",
                  delta=f"{multi['test']['R2'] - single['test']['R2']:+.3f}")
        st.plotly_chart(actual_vs_pred_fig(multi["y_test"], multi["pred_test"], height=300),
                        use_container_width=True, key="flab_avp",
                        config={"displayModeBar": False})

    # coefficients: raw vs standardised
    st.markdown("**Coefficients - why raw values can mislead**")
    raw = ru.fit_and_evaluate("Linear", df, chosen, target, scale=False)
    std = ru.fit_and_evaluate("Linear", df, chosen, target, scale=True)
    raw_c = ru.coefficients(raw["pipeline"], chosen)
    std_c = ru.coefficients(std["pipeline"], chosen)
    cc1, cc2 = st.columns(2)
    for col, dfc, title, colr, key in [
        (cc1, raw_c, "Raw coefficients (different units)", theme.NEUTRAL, "flab_raw"),
        (cc2, std_c, "Standardised coefficients (comparable)", theme.MODEL, "flab_std")]:
        fig = go.Figure(go.Bar(x=dfc["coefficient"], y=dfc["feature"], orientation="h",
                               marker_color=colr))
        theme.style_fig(fig, height=240, title=title)
        col.plotly_chart(fig, use_container_width=True, key=key,
                         config={"displayModeBar": False})
    st.markdown('<span class="rp-note">Raw coefficients are not comparable when features '
                'use different units. Standardised coefficients put them on the same scale. '
                'A large coefficient is an association, not proof of causation.</span>',
                unsafe_allow_html=True)

    if st.button("Reveal ranking of all combinations (up to 3 features)", key="flab_reveal"):
        ranking = cached_combo_ranking(tuple(feats), 3)
        st.dataframe(ranking, use_container_width=True, hide_index=True)
        best = ranking.iloc[0]
        st.success(f"Best by test R-squared: {best['features']} "
                   f"(R² = {best['test_R2']}).")


# ===========================================================================
# Tab 5 - Model Arena
# ===========================================================================
def _model_controls(letter: str):
    default = 1 if letter == "A" else 6  # A -> Linear, B -> Random Forest
    model = st.selectbox(f"Model {letter}", ru.MODEL_CHOICES, index=default,
                         key=f"arena_m{letter}")
    kw = {}
    if model == "Polynomial":
        kw["degree"] = st.slider(f"Polynomial degree · {letter}", 1, 8, 2,
                                 key=f"arena_deg{letter}")
    if model in ("Ridge", "Lasso"):
        exp = st.slider(f"log₁₀ α · {letter}", -2.0, 3.0, 0.0, 0.5,
                        key=f"arena_a{letter}", help="Regularisation strength α.")
        kw["alpha"] = float(10 ** exp)
    if model in ("Decision Tree", "Random Forest", "LightGBM", "XGBoost"):
        kw["max_depth"] = st.slider(f"Max tree depth · {letter} (0 = no limit)",
                                    0, 14, 6, key=f"arena_md{letter}")
    if model in ("Random Forest", "LightGBM", "XGBoost"):
        kw["n_estimators"] = st.slider(f"Number of trees · {letter}", 50, 500, 200, 50,
                                       key=f"arena_ne{letter}")
    if model == "SVM (RBF)":
        kw["C"] = st.slider(f"SVM C · {letter}", 0.1, 100.0, 10.0, key=f"arena_c{letter}")
    if model == "K-Nearest Neighbours":
        kw["n_neighbors"] = st.slider(f"Neighbours k · {letter}", 1, 30, 5,
                                      key=f"arena_k{letter}")
    return model, kw


def _verdict(a_test, b_test, na, nb):
    diff = b_test["R2"] - a_test["R2"]
    if abs(diff) < 0.01:
        return ("The two models perform about the same on unseen data - the difference "
                "is too small to trust from one split.")
    better = nb if diff > 0 else na
    other = na if diff > 0 else nb
    return f"**{better}** generalises better (higher test R-squared) than {other} on this split."


def tab_model_arena(guided: bool):
    theme.header("Model Arena",
                 "Pick a dataset and two models - see them fit the data, then who wins on unseen data")
    theme.challenge("Put a flexible model (Random Forest, XGBoost) against a simple one "
                    "(Linear). Does training-set skill translate to the test set?")
    ds = st.selectbox("Dataset to compare on", MULTI_DATASETS, key="arena_ds",
                      help="Pick which data both models are trained and tested on.")
    df, feats, target = get_multi_dataset(ds)
    st.caption(f"{len(df):,} rows · predicting **{target}** from up to {len(feats)} feature"
               f"{'s' if len(feats) != 1 else ''}.")
    chosen = st.multiselect("Features to use", feats, default=[feats[0]], key="arena_feats",
                            help="One feature shows the fitted curve on a scatter; "
                                 "several build a full multi-feature model.")
    if not chosen:
        st.info("Pick at least one feature.")
        return

    cA, cB = st.columns(2, gap="large")
    with cA:
        mA, kwA = _model_controls("A")
    with cB:
        mB, kwB = _model_controls("B")

    rA = ru.fit_and_evaluate(mA, df, chosen, target, **kwA)
    rB = ru.fit_and_evaluate(mB, df, chosen, target, **kwB)

    # See the data and the fitted models
    if len(chosen) == 1:
        xcol = chosen[0]
        xv = df[xcol].to_numpy(dtype=float)
        yv = df[target].to_numpy(dtype=float)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xv, y=yv, mode="markers", name="data",
                                 marker=dict(color=theme.DATA, size=6, opacity=0.55,
                                             line=dict(color=theme.PALETTE["data_edge"], width=0.5))))
        for res, name, colr in [(rA, mA, theme.MODEL), (rB, mB, theme.NEUTRAL)]:
            xs, curve = ru.predict_curve(res["pipeline"], float(xv.min()), float(xv.max()))
            fig.add_trace(go.Scatter(x=xs, y=curve, mode="lines", name=name,
                                     line=dict(color=colr, width=3)))
        theme.style_fig(fig, height=400,
                        title=f"{target} vs {xcol} - the data with both fitted models")
        fig.update_xaxes(title=xcol)
        fig.update_yaxes(title=target)
        st.plotly_chart(fig, use_container_width=True, key="arena_fit",
                        config={"displayModeBar": False})
        st.caption("With one feature you can see each model's shape directly: a straight "
                   "line, a smooth curve, or the step-like fit of a tree model.")
    else:
        vc1, vc2 = st.columns(2)
        vc1.plotly_chart(actual_vs_pred_fig(rA["y_test"], rA["pred_test"], 300, theme.MODEL),
                         use_container_width=True, key="arena_avpA",
                         config={"displayModeBar": False})
        vc1.caption(f"{mA}: predicted vs actual (test)")
        vc2.plotly_chart(actual_vs_pred_fig(rB["y_test"], rB["pred_test"], 300, theme.NEUTRAL),
                         use_container_width=True, key="arena_avpB",
                         config={"displayModeBar": False})
        vc2.caption(f"{mB}: predicted vs actual (test)")

    mc1, mc2 = st.columns(2, gap="large")
    with mc1:
        st.markdown(f"**{mA}**")
        st.caption("Training set"); metric_cards(rA["train"], "arena_a_tr")
        st.caption("Test set"); metric_cards(rA["test"], "arena_a_te")
    with mc2:
        st.markdown(f"**{mB}**")
        st.caption("Training set"); metric_cards(rB["train"], "arena_b_tr")
        st.caption("Test set"); metric_cards(rB["test"], "arena_b_te")

    # Feature importance (multi-feature only, when the model exposes it)
    if len(chosen) > 1:
        ic1, ic2 = st.columns(2, gap="large")
        for col, res, name, colr, key in [
            (ic1, rA, mA, theme.MODEL, "arena_fiA"),
            (ic2, rB, mB, theme.NEUTRAL, "arena_fiB")]:
            fi, kind = ru.feature_importance(res["pipeline"], chosen)
            if kind:
                fig = go.Figure(go.Bar(x=fi["importance"], y=fi["feature"],
                                       orientation="h", marker_color=colr))
                theme.style_fig(fig, height=240, title=f"{name}: {kind}")
                col.plotly_chart(fig, use_container_width=True, key=key,
                                 config={"displayModeBar": False})
            else:
                col.caption(f"{name} does not expose a feature-importance measure.")

    st.markdown("#### Verdict")
    st.info(_verdict(rA["test"], rB["test"], mA, mB))
    gap_a = rA["train"]["R2"] - rA["test"]["R2"]
    gap_b = rB["train"]["R2"] - rB["test"]["R2"]
    st.markdown(f'<span class="rp-note">Train-minus-test R-squared gap - '
                f'{mA}: {gap_a:+.3f} · {mB}: {gap_b:+.3f}. '
                f'A large positive gap is a sign of overfitting.</span>',
                unsafe_allow_html=True)
    if st.button("Log this comparison", key="arena_log"):
        log_experiment({"tab": "Model Arena", "dataset": ds,
                        "A": mA, "A_test_R2": round(rA["test"]["R2"], 3),
                        "B": mB, "B_test_R2": round(rB["test"]["R2"], 3)})
        st.success("Logged to experiment history.")


# ===========================================================================
# Tab 6 - Overfitting & Regularisation
# ===========================================================================
def tab_overfitting(guided: bool):
    theme.header("Overfitting & Regularisation",
                 "Control complexity - then tame it with Ridge and Lasso")

    left, right = st.tabs(["Complexity (polynomial degree)", "Regularisation (Ridge / Lasso)"])

    # --- polynomial complexity ---
    with left:
        st.caption("Uses made-up practice data with a gentle curve in it.")
        theme.challenge("Which degree is too simple, about right, and too flexible? "
                        "Predict before you look at the test error.")
        n = st.slider("Points", 20, 150, 40, key="of_n")
        noise = st.slider("Noise", 1.0, 20.0, 8.0, 0.5, key="of_noise")
        df = ru.make_synthetic(n=n, noise=noise, curvature=0.6)
        degree = st.slider("Polynomial degree", 1, 12, 1, key="of_deg")

        res = ru.fit_and_evaluate("Polynomial", df, ["x"], "y", degree=degree)
        xs = np.linspace(df["x"].min(), df["x"].max(), 200)
        curve = res["pipeline"].predict(xs.reshape(-1, 1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["x"], y=df["y"], mode="markers", name="data",
                                 marker=dict(color=theme.DATA, size=7)))
        fig.add_trace(go.Scatter(x=xs, y=curve, mode="lines", name=f"degree {degree}",
                                 line=dict(color=theme.MODEL, width=3)))
        theme.style_fig(fig, height=320)
        st.plotly_chart(fig, use_container_width=True, key="of_curve",
                        config={"displayModeBar": False})

        # train/test/CV across degrees (cached - deterministic in n, noise)
        degs, tr, te, cv = cached_degree_sweep(n, noise)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=degs, y=tr, mode="lines+markers", name="train RMSE",
                                  line=dict(color=theme.DATA), marker=dict(symbol="circle")))
        fig2.add_trace(go.Scatter(x=degs, y=te, mode="lines+markers", name="test RMSE",
                                  line=dict(color=theme.ERROR), marker=dict(symbol="square")))
        fig2.add_trace(go.Scatter(x=degs, y=cv, mode="lines+markers", name="CV RMSE (5-fold)",
                                  line=dict(color=theme.GOOD, dash="dot"),
                                  marker=dict(symbol="diamond")))
        theme.style_fig(fig2, height=300, title="Train vs test error as complexity grows")
        fig2.update_xaxes(title="polynomial degree"); fig2.update_yaxes(title="RMSE")
        fig2.add_vline(x=degree, line=dict(color=theme.GOOD, dash="dot"))
        st.plotly_chart(fig2, use_container_width=True, key="of_curves",
                        config={"displayModeBar": False})
        st.markdown('<span class="rp-note">Underfitting: both errors high. Overfitting: '
                    'train keeps dropping while test turns back up. The sweet spot is the '
                    'lowest test error.</span>', unsafe_allow_html=True)

    # --- regularisation ---
    with right:
        theme.challenge("Increase α until Lasso removes at least two features. "
                        "Did test performance improve?")
        ds = st.selectbox("Dataset", ["FuelConsumption CO2", "Energy Efficiency"],
                          key="of_reg_ds")
        df2, feats, target = get_multi_dataset(ds)
        model = st.radio("Model", ["Ridge", "Lasso"], key="of_reg_model", horizontal=True)
        exp = st.slider("log₁₀ α", -2.0, 3.0, 0.0, 0.25, key="of_reg_alpha",
                        help="Regularisation strength α on a log scale.")
        alpha = float(10 ** exp)

        path = cached_coef_path(ds, model)
        fig = px.line(path, x="alpha", y="coefficient", color="feature")
        fig.update_xaxes(type="log", title="α (log scale)")
        theme.style_fig(fig, height=320, title=f"{model} coefficient paths")
        fig.update_layout(showlegend=True)
        fig.add_vline(x=alpha, line=dict(color=theme.INK_SOFT, dash="dot"))
        st.plotly_chart(fig, use_container_width=True, key="of_path",
                        config={"displayModeBar": False})

        res = ru.fit_and_evaluate(model, df2, feats, target, alpha=alpha)
        cv = ru.cross_validate_r2(model, df2, feats, target, k=5, alpha=alpha)
        zeroed = ru.n_zeroed(df2, feats, target, alpha, model=model)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Train R²", f"{res['train']['R2']:.3f}")
        k2.metric("Test R²", f"{res['test']['R2']:.3f}")
        k3.metric("CV R² (5-fold)", f"{cv['mean']:.3f}")
        k4.metric("Coefficients at ~0", f"{zeroed} / {len(feats)}")
        st.markdown('<span class="rp-note">Ridge shrinks coefficients smoothly; Lasso can '
                    'push them exactly to zero, which is a form of automatic feature '
                    'selection. Cross-validation helps choose alpha.</span>',
                    unsafe_allow_html=True)


# ===========================================================================
# Main
# ===========================================================================
def main():
    theme.inject_css()
    st.sidebar.markdown("### Regression Playground")
    mode = st.sidebar.radio("Teaching mode", ["Guided Lab", "Open Playground"],
                            key="app_mode",
                            help="Guided hides answers behind Reveal buttons; "
                                 "Playground shows everything.")
    guided = mode == "Guided Lab"
    st.sidebar.markdown('<span class="rp-note">Week 1 · Session 2 · Regression Techniques '
                        'and Model Evaluation</span>', unsafe_allow_html=True)

    with st.sidebar.expander("How to use", expanded=False):
        st.markdown(
            "- Work left to right through the tabs.\n"
            "- In **Guided Lab**, make a prediction before clicking Reveal.\n"
            "- Everything uses a fixed split (`random_state=42`).\n"
            "- Scaling is fitted on the training split only (no leakage).\n"
            "- Log interesting results and download them below."
        )

    st.info("**Where do I pick the dataset?** Each tab has its own data control near the "
            "top-left, because the tabs need different kinds of data. The first tabs use a "
            "simple made-up dataset so you can see one line; **Feature Lab** uses the real "
            "FuelConsumption data; **Model Arena** and **Regularisation** let you choose.")

    tabs = st.tabs(["Fit the Line", "Gradient Descent", "Metrics Under Pressure",
                    "Feature Lab", "Model Arena", "Overfitting & Regularisation"])
    with tabs[0]:
        tab_fit_line(guided)
    with tabs[1]:
        tab_gradient_descent(guided)
    with tabs[2]:
        tab_metrics(guided)
    with tabs[3]:
        tab_feature_lab(guided)
    with tabs[4]:
        tab_model_arena(guided)
    with tabs[5]:
        tab_overfitting(guided)

    # Experiment history
    hist = st.session_state.get("history", [])
    with st.sidebar.expander(f"Experiment history ({len(hist)})", expanded=False):
        if hist:
            hdf = pd.DataFrame(hist)
            st.dataframe(hdf, use_container_width=True, hide_index=True)
            st.download_button("Download CSV", hdf.to_csv(index=False).encode(),
                               "regression_experiments.csv", "text/csv", key="hist_dl")
            if st.button("Clear history", key="hist_clear"):
                st.session_state["history"] = []
        else:
            st.caption("No experiments logged yet.")

    st.markdown('<div class="rp-footer">Regression Playground · Unitec ML Course · '
                'Week 1 Session 2 · notation θ / α / J(θ) follows the lecture '
                'slides</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
