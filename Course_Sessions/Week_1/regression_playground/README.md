# Regression Playground

An interactive laboratory for **Week 1 · Session 2 — Regression Techniques and
Model Evaluation** (Unitec ML Course). Students make a prediction, change
something, watch the consequence, and explain what happened — rather than read a
passive dashboard.

It is the regression companion to the Week 1 `yolo_dashboard`, and it reuses the
lecture slides' visual identity (cream / terracotta palette, serif titles) and
notation (`theta`, `alpha`, `J(theta)`).

---

## The six workspaces

| Tab | What students do | Concepts |
|---|---|---|
| **Fit the Line** | Drag slope/intercept to minimise error by hand, then reveal OLS | line of best fit, residuals, SSE/MSE, least squares |
| **Gradient Descent** | Watch the machine find the same line by walking downhill | cost `J(theta)`, learning rate, convergence vs divergence |
| **Metrics Under Pressure** | Inject an outlier, see which metric moves most | MAE · MSE · RMSE · R² · MAPE, robustness, negative R² |
| **Feature Lab** | Explore FuelConsumption CO₂, find the best ≤3-feature model | correlation, single vs multiple regression, coefficient interpretation |
| **Model Arena** | Pick a dataset + two models; see the data with each model's fit, then compare | train vs test, generalisation, feature importance |
| **Overfitting & Regularisation** | Push polynomial degree, then tame it with Ridge/Lasso | bias–variance, cross-validation, L1/L2, coefficient paths |

Two modes (sidebar): **Guided Lab** hides answers behind *Reveal* buttons so
students predict first; **Open Playground** shows everything for free exploration.

**Models available in Model Arena:** Mean baseline, Linear, Polynomial, Ridge,
Lasso, Decision Tree, Random Forest, SVM (RBF), K-Nearest Neighbours, and
LightGBM / XGBoost when installed. With a single feature selected you see each
model's fitted curve drawn straight onto the data scatter (a straight line, a
smooth curve, or a tree's step-like fit); with several features you get
predicted-vs-actual plus feature-importance bars. All equations render as proper
maths (θ, α, `J(θ)`), not plain text.

---

## How to run

```bash
# 1. Activate the course environment
conda activate mlcourse

# 2. Go to this folder
cd "Course_Sessions/Week_1/regression_playground"

# 3. (first time only) install anything missing
pip install -r requirements.txt

# 4. Launch
streamlit run app.py
```

Opens at **http://localhost:8501**. Stop with `Ctrl + C`.
No conda? Any Python 3.10+ environment works — `pip install -r requirements.txt`.

---

## Datasets

- **Synthetic sandbox** — controllable noise/curvature; used for fitting, gradient
  descent, metrics and overfitting (no files needed).
- **FuelConsumption CO₂** — the Session 2 class-activity dataset (`ENGINESIZE`,
  `CYLINDERS`, `FUELCONSUMPTION_*` → `CO2EMISSIONS`), read from `../../../Datasets/`.
- **Energy Efficiency (ENB2012)** — 8 building features → Heating Load.
- **Anscombe's Quartet** — four datasets with near-identical statistics but very
  different shapes (built in).

---

## Teaching guarantees (baked in)

- Fixed `random_state = 42`; the train/test split is **locked** when comparing models.
- Scaling/polynomial features are fitted **inside a pipeline on the training split
  only** — no data leakage.
- A **mean-prediction baseline** is always available, and **R² is allowed to go
  negative** (worse than the mean).
- Diagnostics are framed as visual evidence, not pass/fail tests.
- No universal "best model" is declared; a coefficient is an association, not a cause.

---

## A 60–75 minute class flow

1. Fit a line by hand, then reveal OLS.
2. Watch gradient descent find the same line; try too-small / just-right / too-big α.
3. Inject an outlier — compare MAE with RMSE.
4. Feature Lab: build the best CO₂ model with ≤3 features, then reveal the ranking.
5. Model Arena: a flexible model vs a simple one — who wins on the test set?
6. Overfitting: predict which polynomial degree overfits, then check with CV.
7. Tune Ridge / Lasso; watch coefficients shrink and zero out.
8. Exit question: *what would you change in the data, features, or model — and why?*
   Log results and download the experiment history as evidence.

---

## How it's organised (easy to modify)

```
regression_playground/
├── app.py               # Streamlit UI (six workspaces)
├── regression_utils.py  # datasets, pipelines, metrics, gradient descent, CV  <- edit to extend
├── theme.py             # slide palette + CSS + Plotly styling  <- edit to rebrand
├── requirements.txt
├── README.md
└── .streamlit/config.toml
```

Add a dataset in `regression_utils.py`; change a colour in `theme.py`.
