# Regression Playground

An interactive laboratory for **Week 1 · Session 2 — Regression Techniques and
Model Evaluation** in the Unitec Machine Learning course.

Students are asked to predict, change something, inspect the evidence, and then
explain what happened. The app complements the Session 2 notebook; it is not a
replacement for writing and interpreting regression code.

The interface retains the course slides' warm cream/terracotta identity, with
readable sans-serif body text and serif headings.

---

## The six workspaces

| Workspace | Student activity | Main concepts |
|---|---|---|
| **Fit the Line** | Adjust slope and intercept, inspect residuals and compare with least squares | line of best fit, residuals, SSE, MSE, R² |
| **Gradient Descent** | Change the data shape, scrub through the fitted `θ` values and compare several learning-rate cost curves on the same observations | cost `J(θ)`, learning rate `α`, convergence, divergence and model mismatch |
| **Feature Lab** | Choose a small feature set, inspect associations and compare candidate sets fairly | correlation, simple vs multiple regression, standardised coefficients, multicollinearity |
| **Model Arena** | Compare two models on the same split, make a choice, then reveal held-out performance | baselines, training vs validation/test, generalisation, model comparison |
| **Generalisation & Regularisation** | Change polynomial degree, inspect fold scores, then tune Ridge or Lasso | underfit/overfit, cross-validation, bias–variance, L1/L2, coefficient paths |
| **Diagnose** | Identify deliberately generated residual patterns before revealing the explanation | linearity, constant variance, residual normality, independence, outliers and leverage |

The workspaces use a vertical tab rail in the left sidebar. It stays visible on
wide classroom screens and collapses behind Streamlit's standard menu control on
narrow screens.

Each workspace begins with a short challenge. Relevant outcomes stay hidden
until the student makes or locks a prediction; explanations are revealed after
the observation, not before it.

---

## Fair model selection and the locked test

Activities that select features or tune a model follow this sequence:

1. The data split and random seed are held constant for a fair comparison.
2. Candidate features and hyperparameters are compared with cross-validation on
   the training data only.
3. The student records or locks a choice.
4. The untouched test result is then revealed once as a final check.
5. Changing the choice locks the test result again.

This prevents the test set from becoming another tuning set. Cross-validation
charts report fold-level scores (including mean and spread), and RMSE is always
shown as a non-negative error in the target's original units.

---

## Models

Both Model Arena selectors expose the complete model list directly:

- Mean-prediction baseline
- Ordinary Linear Regression
- Polynomial Regression
- Ridge Regression
- Lasso Regression
- Decision Tree
- Random Forest
- Support Vector Regression (RBF)
- K-Nearest Neighbours
- LightGBM and XGBoost when those optional packages are available

There is no separate basic/advanced mode.

### Safe polynomial experiments

Polynomial expansion grows very quickly when degree and feature count are both
increased. Before fitting, the playground calculates the expanded term count.
Unsafe combinations are blocked with a suggestion to reduce the degree or
number of features. High polynomial degrees are intended for one-feature
underfitting/overfitting experiments; multivariable experiments use conservative
limits. Polynomial terms are labelled with their feature names rather than
anonymous term numbers.

---

## Datasets

| Dataset | Main use in the playground |
|---|---|
| **Synthetic sandbox** | noise, outliers, assumptions, polynomial complexity and controlled comparisons |
| **Anscombe's Quartet** | why a scatter plot matters even when summary statistics look similar |
| **FuelConsumption CO₂** | the Session 2 class activity: simple/multiple regression and feature choice |
| **Energy Efficiency (ENB2012)** | multiple regression, polynomial terms, scaling and regularisation |
| **Wine Quality (red)** | a larger multivariable comparison |
| **Diabetes** | a bundled scikit-learn regression example |
| **Startup Profit** | business-feature regression and coefficient interpretation |
| **Student wellbeing** | discussion of association, prediction and the limits of causal claims |

The data loaders resolve files from the repository, so the app does not depend
on the terminal's current folder or download data at launch.

---

## Teaching safeguards

- Reproducible examples use `random_state=42` by default.
- Compared models use the same observations and split.
- Scaling and polynomial expansion are fitted inside a pipeline on training
  folds only.
- A mean-prediction baseline is available, and R² is allowed to be negative
  when a model performs worse than that baseline.
- MAE and RMSE retain target units; MSE is identified as using squared units.
- Repeated model/feature selection uses training-only cross-validation.
- Coefficients are described as conditional associations, not causes.
- Assumption displays are described as **visual clues**, not automatic
  pass/fail tests.
- Train, validation/test and CV series use both colour and distinct
  markers/line styles.

---

## Suggested 60–75 minute class route

1. **Fit the Line:** fit by eye and predict what least squares will change.
2. **Gradient Descent:** compare learning-rate curves on linear, curved and deliberately difficult data, then inspect `θ₀` and `θ₁` at a chosen iteration.
3. **Feature Lab:** build a CO₂ model using no more than three features and lock
   the feature choice.
4. **Model Arena:** compare a baseline/simple model with a more flexible model;
   choose before revealing the held-out result.
5. **Generalisation & Regularisation:** spot an overfit polynomial and use
   Ridge/Lasso to control complexity.
6. **Diagnose:** identify one residual pattern and finish with:
   *What would you change in the data, features or model—and why?*

For a shorter session, Gradient Descent or the coefficient-path portion can be
an instructor-led demonstration.

---

## Launch with `mlcourse`

From the repository root:

```bash
conda run -n mlcourse streamlit run \
  Course_Sessions/Week_1/regression_playground/app.py
```

Or activate the environment first:

```bash
conda activate mlcourse
cd Course_Sessions/Week_1/regression_playground
streamlit run app.py
```

Streamlit normally opens `http://localhost:8501`. Stop the server with
`Ctrl+C`. The playground does not need a webcam, model weights, a GPU, or an
internet connection.

---

## Verification

From this folder, run the modelling and Streamlit interaction tests with:

```bash
conda run -n mlcourse python -m pytest -q
```

The suite checks the locked-split/CV rules, metrics, polynomial safety, encoded
Energy features, all diagnostic scenarios, every workspace and the staged
reveal paths.

---

## Project structure

```text
regression_playground/
├── app.py               # Streamlit UI and the six workspaces
├── regression_utils.py  # datasets, pipelines, metrics, CV and diagnostics
├── theme.py             # accessible UI palette, CSS and Plotly styling
├── requirements.txt     # direct runtime dependencies
├── tests/                # modelling and Streamlit interaction checks
├── README.md
└── .streamlit/
    └── config.toml
```

Add or adjust datasets and modelling logic in `regression_utils.py`; keep visual
tokens and shared chart styling in `theme.py`.
