# Model Interpretability

Interpretability methods explain what a trained model has learned and why it produces a given prediction. They serve three distinct purposes: debugging the model and its training data, communicating behaviour to stakeholders, and meeting regulatory or audit requirements. This guide covers the dominant techniques, the model classes each supports, and the trade-offs between them.

**Table of Contents**

1. [Global vs Local Explanations](#1-global-vs-local-explanations)
2. [Built-In Feature Importance](#2-built-in-feature-importance)
3. [Permutation Importance](#3-permutation-importance)
4. [Partial Dependence and ICE Plots](#4-partial-dependence-and-ice-plots)
5. [SHAP — Unified Attribution](#5-shap--unified-attribution)
6. [LIME — Local Surrogate Models](#6-lime--local-surrogate-models)
7. [Counterfactual Explanations](#7-counterfactual-explanations)
8. [Model Cards and Reporting](#8-model-cards-and-reporting)
9. [Method Selection](#9-method-selection)
10. [Resources](#10-resources)

---

## 1. Global vs Local Explanations

Interpretability methods partition along two axes: scope (global or local) and applicability (model-specific or model-agnostic).

| Scope | Question answered | Examples |
|-------|-------------------|----------|
| Global | "What does the model rely on overall?" | Built-in feature importance, permutation importance, partial dependence |
| Local | "Why this particular prediction?" | SHAP values, LIME, counterfactuals |

| Applicability | Constraint | Examples |
|---------------|------------|----------|
| Model-specific | Requires access to model internals | Tree feature importance, linear coefficients, attention weights |
| Model-agnostic | Treats the model as a black box | Permutation importance, SHAP (KernelExplainer), LIME |

Choosing between them depends on whether the explanation is for a single decision or for the model's general behaviour, and on whether internals are accessible.

---

## 2. Built-In Feature Importance

Many model classes expose feature importance directly.

### 2.1 Tree-Based Models

Tree ensembles compute importance from how often a feature is used in splits and how much each split reduces the loss.

```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(random_state=0).fit(X_train, y_train)
importances = model.feature_importances_

import pandas as pd
pd.Series(importances, index=feature_names).sort_values(ascending=False)
```

Caveats:
- Biased toward high-cardinality features.
- Correlated features split importance arbitrarily between them.
- Captures training-set behaviour; does not reflect held-out generalization.

### 2.2 Linear Models

For linear models, the magnitude of standardized coefficients indicates feature influence.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

pipe = make_pipeline(StandardScaler(), LogisticRegression()).fit(X_train, y_train)
coefs = pipe.named_steps["logisticregression"].coef_[0]
```

Coefficients are interpretable as marginal effects only when features are scaled and approximately uncorrelated.

---

## 3. Permutation Importance

Permutation importance measures how much a model's score drops when a single feature's values are randomly shuffled. It is model-agnostic and computed on held-out data, addressing two of the limitations of built-in tree importance.

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(
    model, X_val, y_val,
    n_repeats=10, random_state=0, n_jobs=-1,
)

import pandas as pd
pd.DataFrame({
    "feature": feature_names,
    "mean_drop": result.importances_mean,
    "std": result.importances_std,
}).sort_values("mean_drop", ascending=False)
```

### 3.1 Caveats

- Correlated features mask each other: shuffling one barely hurts performance because another carries the same information. Group correlated features and permute jointly.
- Computationally expensive: cost is `n_repeats × n_features × inference_time`.
- Reflects the chosen scoring metric; importance for accuracy may differ from importance for log-loss.

---

## 4. Partial Dependence and ICE Plots

### 4.1 Partial Dependence Plots (PDP)

A partial dependence plot shows the average predicted output as a function of a single feature, marginalizing over the others. It reveals the average direction and shape of a feature's effect.

```python
from sklearn.inspection import PartialDependenceDisplay

PartialDependenceDisplay.from_estimator(
    model, X_val, features=["age", "income"], kind="average",
)
```

PDPs assume features are independent. Strong correlations produce unrealistic averaging — combinations may be plotted that never occur in the data.

### 4.2 Individual Conditional Expectation (ICE)

ICE plots show one curve per sample instead of an average. Heterogeneity between curves indicates interactions; parallel curves indicate an additive effect.

```python
PartialDependenceDisplay.from_estimator(
    model, X_val, features=["age"], kind="both",   # average + per-sample
)
```

### 4.3 PDP / ICE Decision

| Pattern | Interpretation |
|---------|----------------|
| Flat PDP, flat ICEs | Feature has no effect |
| Sloped PDP, parallel ICEs | Feature has a consistent main effect |
| Sloped PDP, fanning ICEs | Feature has interaction effects with others |
| Flat PDP, fanning ICEs | Effects cancel on average; per-sample matters |

---

## 5. SHAP — Unified Attribution

SHAP (SHapley Additive exPlanations) attributes a single prediction to its input features by computing each feature's contribution to the deviation from the average prediction. The values are derived from cooperative game theory and satisfy desirable properties (additivity, symmetry, dummy).

### 5.1 Tree-Based Models

For tree ensembles, `TreeExplainer` computes exact SHAP values efficiently.

```python
# pip install shap
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)

# Local explanation for one prediction
shap.plots.waterfall(shap.Explanation(
    values=shap_values[0],
    base_values=explainer.expected_value,
    data=X_val.iloc[0],
    feature_names=feature_names,
))

# Global summary
shap.summary_plot(shap_values, X_val)
```

### 5.2 Model-Agnostic SHAP

For arbitrary models, `KernelExplainer` and `Explainer` provide model-agnostic estimates. Both are slower than tree-specific methods.

```python
explainer = shap.Explainer(model.predict, X_train_summary)
shap_values = explainer(X_val[:100])    # subsample for speed
```

### 5.3 Reading SHAP Outputs

| Plot | Reads as |
|------|----------|
| Waterfall | Per-prediction breakdown: which features pushed the prediction up or down |
| Force | Compact horizontal version of the waterfall |
| Summary (beeswarm) | Global feature ranking, with per-sample value distribution |
| Dependence | A feature's SHAP value vs its raw value, coloured by an interacting feature |

SHAP values sum to the difference between a single prediction and the model's expected output — making them additive and locally accurate.

---

## 6. LIME — Local Surrogate Models

LIME (Local Interpretable Model-agnostic Explanations) explains an individual prediction by fitting a simple, interpretable model (typically a linear regression) to a local neighbourhood of perturbed samples.

```python
# pip install lime
from lime.lime_tabular import LimeTabularExplainer

explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=feature_names,
    class_names=["0", "1"],
    mode="classification",
)

exp = explainer.explain_instance(
    X_val.iloc[0].values, model.predict_proba, num_features=10,
)
exp.show_in_notebook()
```

### 6.1 LIME vs SHAP

| Property | LIME | SHAP |
|----------|------|------|
| Theoretical basis | Local linear approximation | Cooperative game theory |
| Stability across runs | Lower (relies on random sampling) | Higher (deterministic for tree models) |
| Speed | Faster per instance for tabular | Slower (KernelExplainer); fast for trees |
| Global aggregation | Limited | Direct (mean absolute SHAP per feature) |
| Image / text support | Native | Available; less mature than tabular |

LIME is appropriate for fast, single-instance explanations; SHAP is the default when stability and global aggregation matter.

---

## 7. Counterfactual Explanations

A counterfactual explanation answers "what would need to change for the prediction to flip?" It is intuitive for non-technical audiences and aligns with how legal and regulatory frameworks frame "right to explanation".

```python
# pip install dice-ml
import dice_ml

data_dice = dice_ml.Data(
    dataframe=df, continuous_features=["age", "income"], outcome_name="approved",
)
model_dice = dice_ml.Model(model=model, backend="sklearn")
exp = dice_ml.Dice(data_dice, model_dice, method="random")

cf = exp.generate_counterfactuals(
    X_val.iloc[[0]], total_CFs=3, desired_class="opposite",
)
cf.visualize_as_dataframe()
```

Useful counterfactuals are: minimal (few features changed), realistic (within the data manifold), and actionable (only mutable features changed). Generic counterfactual generators do not enforce all three by default; constraints must be specified explicitly.

---

## 8. Model Cards and Reporting

Interpretability outputs benefit from a structured report that accompanies a deployed model. The model card framework (Mitchell et al.) standardizes the document.

Standard sections:

- **Model details** — type, version, training date, owners.
- **Intended use** — task, intended users, out-of-scope uses.
- **Factors** — relevant demographic or environmental factors that may affect performance.
- **Metrics** — overall and disaggregated performance metrics.
- **Evaluation data** — composition, motivation, preprocessing.
- **Training data** — same as above for the training set.
- **Quantitative analyses** — performance broken down by relevant subgroups.
- **Ethical considerations** — risks, mitigations, and known limitations.

A model card converts ad-hoc interpretability artefacts (importance plots, fairness metrics, performance tables) into a single durable record.

---

## 9. Method Selection

| Goal | Recommended method |
|------|--------------------|
| Quick global ranking on a tree model | Built-in `feature_importances_`, then validate with permutation importance |
| Robust global ranking on any model | Permutation importance on held-out data |
| Understanding the shape of a feature's effect | PDP for average; ICE for heterogeneity |
| Per-prediction explanation, tree model | SHAP TreeExplainer |
| Per-prediction explanation, any model | SHAP `Explainer` (slow) or LIME (faster, less stable) |
| Explanation for non-technical stakeholder | Counterfactual explanation |
| Regulated context requiring documentation | Model card with disaggregated metrics + global SHAP summary |

Multiple methods used together typically produce more reliable conclusions than any single method alone.

---

## 10. Resources

- [Molnar, *Interpretable Machine Learning*](https://christophm.github.io/interpretable-ml-book/) — open-access reference covering every major technique.
- [SHAP documentation](https://shap.readthedocs.io/) — explainers, plots, and worked examples.
- [LIME repository](https://github.com/marcotcr/lime) — tabular, text, and image explainers.
- [DiCE](https://github.com/interpretml/DiCE) — counterfactual explanations.
- [scikit-learn — Inspection module](https://scikit-learn.org/stable/inspection.html) — permutation importance, partial dependence, ICE.
- [Lundberg and Lee, *A Unified Approach to Interpreting Model Predictions* (2017)](https://arxiv.org/abs/1705.07874) — original SHAP paper.
- [Ribeiro, Singh, Guestrin, *"Why Should I Trust You?": Explaining the Predictions of Any Classifier* (2016)](https://arxiv.org/abs/1602.04938) — original LIME paper.
- [Mitchell et al., *Model Cards for Model Reporting* (2019)](https://arxiv.org/abs/1810.03993) — model card framework.

---

[← Previous: Testing ML Code](16_TESTING_ML_CODE_GUIDE.md) | [Index](README.md) | [Next: Unsupervised Learning →](18_UNSUPERVISED_LEARNING_GUIDE.md)
