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
| **Fit the Line** | Fit a line by hand, then reveal least squares | line of best fit, residuals, SSE/MSE |
| **Gradient Descent** | Watch the machine find the same line by walking downhill | cost `J(θ)`, learning rate α, convergence vs divergence |
| **Feature Lab** | Pick a dataset, find the best ≤3-feature model | correlation, single vs multiple regression, coefficient interpretation |
| **Model Arena** | Pick a dataset + two models; see the data with each model's fit, then compare | train vs test, generalisation, feature importance |
| **Overfitting & Regularisation** | Push polynomial degree, then tame it with Ridge/Lasso | bias–variance, cross-validation, L1/L2, coefficient paths |

The *Reveal* boxes stay collapsed so students can predict before seeing the answer.

**Models (Model Arena):** Mean baseline, Linear, Polynomial, Ridge, Lasso,
Decision Tree, Random Forest, SVM (RBF), K-Nearest Neighbours, and LightGBM /
XGBoost when installed. With one feature selected you see each model's fitted
curve on the data scatter (line, smooth curve, or a tree's step-like fit); with
several features you get predicted-vs-actual plus feature-importance bars. All
equations render as proper maths (θ, α, `J(θ)`).

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

Feature Lab and Model Arena share a dataset picker:

- **FuelConsumption CO₂** — the Session 2 class-activity dataset (`ENGINESIZE`,
  `CYLINDERS`, `FUELCONSUMPTION_*` → `CO2EMISSIONS`).
- **Energy Efficiency (ENB2012)** — 8 building features → Heating Load.
- **Wine Quality (red)** — 11 physico-chemical features → quality.
- **Diabetes** — 10 features → disease progression (scikit-learn, bundled).
- **Startup Profit** — R&D / Admin / Marketing spend → Profit.
- **Student wellbeing** — usage, sleep, mental-health → addiction score.

Model Arena also offers a **Synthetic sandbox** (one feature) so you can see each
model's fitted curve directly. Fit the Line, Gradient Descent and the complexity
demo use their own synthetic data (plus **Anscombe's Quartet** in Fit the Line).

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

1. Fit a line by hand, then reveal least squares.
2. Watch gradient descent find the same line; try too-small / just-right / too-big α.
3. Feature Lab: build the best model with ≤3 features, then reveal the ranking.
4. Model Arena: a flexible model vs a simple one — who wins on the test set?
5. Overfitting: predict which polynomial degree overfits, then check with CV.
6. Tune Ridge / Lasso; watch coefficients shrink and zero out.
7. Exit question: *what would you change in the data, features, or model — and why?*
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
