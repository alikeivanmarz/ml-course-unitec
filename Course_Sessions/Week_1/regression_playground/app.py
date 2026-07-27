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
def get_dataset(name: str):
    """Return (df, features, target) for a named dataset (cached)."""
    return ru.get_named_dataset(name)


@st.cache_data(show_spinner=False)
def cached_combo_ranking(ds_name: str, max_k: int) -> pd.DataFrame:
    df, feats, target = get_dataset(ds_name)
    return ru.rank_feature_combos(df, feats, target, max_k=max_k)


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
    df, feats, target = get_dataset(ds_name)
    return ru.coefficient_path(df, feats, target, model=model)


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
# Single-feature real examples: label -> (dataset name, x column, y column)
FL_EXAMPLES = {
    "Engine size → CO₂": ("FuelConsumption CO2", "ENGINESIZE", "CO2EMISSIONS"),
    "Fuel use → CO₂": ("FuelConsumption CO2", "FUELCONSUMPTION_COMB", "CO2EMISSIONS"),
    "R&D spend → Profit": ("Startup Profit", "R&D Spend", "Profit"),
    "Social-media use → Addiction": ("Student wellbeing", "Avg_Daily_Usage_Hours", "Addicted_Score"),
    "Alcohol → Wine quality": ("Wine Quality (red)", "alcohol", "quality"),
}


@st.cache_data(show_spinner=False)
def fl_example_xy(label: str, max_points: int = 150):
    """Return (x, y) for a real single-feature example, sampled for a clean scatter."""
    name, xcol, ycol = FL_EXAMPLES[label]
    df = ru.get_named_dataset(name)[0][[xcol, ycol]].dropna()
    if len(df) > max_points:
        df = df.sample(max_points, random_state=ru.RANDOM_STATE)
    return df[xcol].to_numpy(float), df[ycol].to_numpy(float), xcol, ycol


def tab_fit_line():
    theme.header("Fit the Line", "Fit a line by hand, then compare with least squares")
    theme.challenge("Move the sliders to raise your R², then reveal the best-fit line.")
    st.latex(r"\hat{y} = \theta_0 + \theta_1\,x")

    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        choice = st.selectbox(
            "Example", ["Practice data (make your own)", *FL_EXAMPLES,
                        "Anscombe's Quartet"], key="fl_choice")
        xlabel, ylabel = "x", "y"
        if choice.startswith("Practice"):
            n = st.slider("How many dots", 10, 200, 40, key="fl_n")
            noise = st.slider("How scattered (noise)", 0.0, 20.0, 6.0, 0.5, key="fl_noise")
            seed = st.number_input("Shuffle (seed)", 0, 9999, 42, key="fl_seed")
            df = ru.make_synthetic(n=n, noise=noise, seed=int(seed))
            if st.checkbox("Add one odd point (outlier)", key="fl_out"):
                df = pd.concat([df, pd.DataFrame({"x": [df["x"].max()],
                                                  "y": [df["y"].min() - 3 * noise - 10]})],
                               ignore_index=True)
            x, y = df["x"].to_numpy(), df["y"].to_numpy()
        elif choice == "Anscombe's Quartet":
            which = st.selectbox("Which of the four sets", ["I", "II", "III", "IV"],
                                 key="fl_ans")
            df = ru.make_anscombe(which)
            x, y = df["x"].to_numpy(), df["y"].to_numpy()
            st.markdown('<span class="rp-note">Four sets with near-identical statistics '
                        'that look completely different once plotted.</span>',
                        unsafe_allow_html=True)
        else:
            x, y, xlabel, ylabel = fl_example_xy(choice)

        ols_slope, ols_intercept = ru.ols_1d(x, y)
        # Adaptive ranges so real-unit data (e.g. CO2) is reachable; keyed per
        # example so switching resets the sliders (avoids out-of-range errors).
        s_hi = max(abs(ols_slope) * 3.0, 1.0)
        span = float(y.max() - y.min()) or 1.0
        i_lo, i_hi = float(y.min() - span), float(y.max() + span)
        slope = st.slider("Slope (steepness)", -s_hi, s_hi, 0.0, step=s_hi / 100,
                          key=f"fl_slope_{choice}", help="θ₁ in the equation.")
        intercept = st.slider("Intercept (where it starts)", i_lo, i_hi,
                              float(np.clip(y.mean(), i_lo, i_hi)), step=span / 100,
                              key=f"fl_intercept_{choice}", help="θ₀ in the equation.")

    with c2:
        student = ru.all_metrics(y, intercept + slope * x)
        ols_r2 = ru.all_metrics(y, ols_intercept + ols_slope * x)["R2"]
        show_best = st.checkbox("Show the best-fit line", key="fl_show_best",
                                help="Fit by eye first, then tick this to compare.")
        lines = [{"slope": slope, "intercept": intercept, "name": "your line",
                  "color": theme.MODEL}]
        if show_best:
            lines.append({"slope": ols_slope, "intercept": ols_intercept,
                          "name": "best-fit line", "color": theme.GOOD})

        fig = scatter_line_fig(x, y, lines, xtitle=xlabel, ytitle=ylabel,
                               residual_from={"slope": slope, "intercept": intercept,
                                              "color": theme.ERROR})
        st.plotly_chart(fig, use_container_width=True, key="fl_fig",
                        config={"displayModeBar": False})
        st.caption("Red dotted lines are the errors — make them small overall.")
        if show_best:
            st.markdown(
                f"Best-fit line: slope **{ols_slope:.3g}**, intercept **{ols_intercept:.3g}**, "
                f"**R² {ols_r2:.3f}** — the highest possible for a straight line. "
                f"Your R² is **{student['R2']:.3f}**.")

        m1, m2, m3 = st.columns(3)
        m1.metric("Your R²", f"{student['R2']:.3f}",
                  help="1 = perfect · 0 = no better than a flat line at the mean · "
                       "can be negative")
        m2.metric("Your RMSE", f"{student['RMSE']:,.2f}", help=f"typical error, in {ylabel}")
        m3.metric("Your MAE", f"{student['MAE']:,.2f}")

        pc1, pc2 = st.columns([2, 1])
        px_val = pc1.slider(f"Predict {ylabel} when {xlabel} =",
                            float(x.min()), float(x.max()), float(np.median(x)),
                            key=f"fl_predx_{choice}")
        pc2.metric("Your line predicts", f"{intercept + slope * px_val:.2f}")

        if st.button("Log this attempt", key="fl_log"):
            log_experiment({"tab": "Fit the Line", "example": choice,
                            "slope": round(slope, 3), "intercept": round(intercept, 3),
                            "your_R2": round(student["R2"], 3),
                            "best_R2": round(ols_r2, 3)})
            st.success("Logged to experiment history.")


# ===========================================================================
# Tab 2 - Gradient Descent
# ===========================================================================
def tab_gradient_descent():
    theme.header("Gradient Descent", "How the machine finds the line by minimising cost")
    theme.challenge("Try a tiny, a good, and a huge learning rate α. Watch the cost curve.")
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
# Tab 4 - Feature Lab
# ===========================================================================
def tab_feature_lab():
    theme.header("Feature Lab", "Which features actually drive the prediction")
    theme.challenge("Build the best model with three features or fewer, then reveal the ranking.")

    ds = st.selectbox("Dataset", ru.MULTI_FEATURE_DATASETS, key="flab_ds")
    df, feats, target = get_dataset(ds)
    st.caption(f"{len(df):,} rows · predicting **{target}** from {len(feats)} features.")

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
                                default=[feats[0]], key=f"flab_feats_{ds}",
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
    st.markdown("**Coefficients: raw vs standardised**")
    raw = ru.fit_and_evaluate("Linear", df, chosen, target, scale=False)
    std = ru.fit_and_evaluate("Linear", df, chosen, target, scale=True)
    raw_c = ru.coefficients(raw["pipeline"], chosen)
    std_c = ru.coefficients(std["pipeline"], chosen)
    cc1, cc2 = st.columns(2)
    for col, dfc, title, colr, key in [
        (cc1, raw_c, "Raw (different units)", theme.NEUTRAL, "flab_raw"),
        (cc2, std_c, "Standardised (comparable)", theme.MODEL, "flab_std")]:
        fig = go.Figure(go.Bar(x=dfc["coefficient"], y=dfc["feature"], orientation="h",
                               marker_color=colr))
        theme.style_fig(fig, height=240, title=title)
        col.plotly_chart(fig, use_container_width=True, key=key,
                         config={"displayModeBar": False})
    st.markdown('<span class="rp-note">Raw coefficients aren\'t comparable across different '
                'units; standardised ones are. Association is not causation.</span>',
                unsafe_allow_html=True)

    if st.button("Reveal best feature combination", key="flab_reveal"):
        ranking = cached_combo_ranking(ds, 3)
        st.dataframe(ranking, use_container_width=True, hide_index=True)
        best = ranking.iloc[0]
        st.success(f"Best (test R²): {best['features']} — R² {best['test_R2']}")


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


def tab_model_arena():
    theme.header("Model Arena", "Compare two models on the same data")
    theme.challenge("Put a flexible model against a simple one. Does train skill hold on the test set?")
    ds = st.selectbox("Dataset", ru.ARENA_DATASETS, key="arena_ds")
    df, feats, target = get_dataset(ds)
    st.caption(f"{len(df):,} rows · predicting **{target}** from up to {len(feats)} feature"
               f"{'s' if len(feats) != 1 else ''}.")
    chosen = st.multiselect("Features", feats, default=[feats[0]], key=f"arena_feats_{ds}",
                            help="One feature draws the fitted curve; several build a "
                                 "multi-feature model.")
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
        theme.style_fig(fig, height=400, title=f"{target} vs {xcol}")
        fig.update_xaxes(title=xcol)
        fig.update_yaxes(title=target)
        st.plotly_chart(fig, use_container_width=True, key="arena_fit",
                        config={"displayModeBar": False})
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

    st.info(_verdict(rA["test"], rB["test"], mA, mB))
    gap_a = rA["train"]["R2"] - rA["test"]["R2"]
    gap_b = rB["train"]["R2"] - rB["test"]["R2"]
    st.markdown(f'<span class="rp-note">Train−test R² gap · {mA}: {gap_a:+.3f} · '
                f'{mB}: {gap_b:+.3f}. A large gap signals overfitting.</span>',
                unsafe_allow_html=True)
    if st.button("Log this comparison", key="arena_log"):
        log_experiment({"tab": "Model Arena", "dataset": ds,
                        "A": mA, "A_test_R2": round(rA["test"]["R2"], 3),
                        "B": mB, "B_test_R2": round(rB["test"]["R2"], 3)})
        st.success("Logged to experiment history.")


# ===========================================================================
# Tab 6 - Overfitting & Regularisation
# ===========================================================================
def tab_overfitting():
    theme.header("Overfitting & Regularisation",
                 "Control complexity, then tame it with Ridge and Lasso")

    left, right = st.tabs(["Complexity (polynomial degree)", "Regularisation (Ridge / Lasso)"])

    # --- polynomial complexity ---
    with left:
        theme.challenge("Which degree is too simple, about right, and too flexible?")
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
                    'train drops while test turns back up.</span>', unsafe_allow_html=True)

    # --- regularisation ---
    with right:
        theme.challenge("Raise α until Lasso zeros out features. Does the test score improve?")
        ds = st.selectbox("Dataset", ru.MULTI_FEATURE_DATASETS, key="of_reg_ds")
        df2, feats, target = get_dataset(ds)
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
                    'push them to zero (automatic feature selection).</span>',
                    unsafe_allow_html=True)


# ===========================================================================
# Main
# ===========================================================================
def sidebar():
    sb = st.sidebar
    sb.markdown("### Regression Playground")
    sb.caption("Week 1 · Session 2 — Regression")
    sb.markdown(
        '<div class="rp-note" style="line-height:1.9">'
        '<span style="color:#1C3D5A">■</span> data &nbsp;&nbsp;'
        '<span style="color:#A8622D">■</span> your line / model A<br>'
        '<span style="color:#3E7063">■</span> best fit / target &nbsp;&nbsp;'
        '<span style="color:#6B4E71">■</span> model B</div>',
        unsafe_allow_html=True)
    sb.divider()

    hist = st.session_state.get("history", [])
    sb.markdown(f"**Experiment log** ({len(hist)})")
    if hist:
        hdf = pd.DataFrame(hist)
        sb.dataframe(hdf, use_container_width=True, hide_index=True, height=200)
        sb.download_button("Download CSV", hdf.to_csv(index=False).encode(),
                           "regression_experiments.csv", "text/csv", key="hist_dl")
        if sb.button("Clear", key="hist_clear"):
            st.session_state["history"] = []
    else:
        sb.caption("Log results from Fit the Line and Model Arena to collect them here.")
    sb.divider()
    sb.caption("Fixed split (random_state = 42). Scaling is fitted on the training data only.")


def main():
    theme.inject_css()
    sidebar()

    tabs = st.tabs(["Fit the Line", "Gradient Descent", "Feature Lab",
                    "Model Arena", "Overfitting & Regularisation"])
    with tabs[0]:
        tab_fit_line()
    with tabs[1]:
        tab_gradient_descent()
    with tabs[2]:
        tab_feature_lab()
    with tabs[3]:
        tab_model_arena()
    with tabs[4]:
        tab_overfitting()

    st.markdown('<div class="rp-footer">Regression Playground · Unitec ML Course · '
                'Week 1 · Session 2 · notation θ / α / J(θ)</div>',
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()
