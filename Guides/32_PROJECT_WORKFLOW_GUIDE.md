# End-to-End ML Project Workflow

This guide walks through the steps of a complete machine-learning project, from problem definition to final delivery. It is generalised — applicable to any supervised learning project — and is meant as a checklist of *what to do, in what order*. For technical detail on any step, follow the linked guide.

The steps are presented in the order they should be performed. Skipping a step or doing them out of order is the most common cause of projects that produce inflated results, fail in unexpected ways, or are difficult to communicate.

**Table of Contents**

1. [Define the Problem](#1-define-the-problem)
2. [Get the Data](#2-get-the-data)
3. [Explore the Data (EDA)](#3-explore-the-data-eda)
4. [Split into Train, Validation, and Test](#4-split-into-train-validation-and-test)
5. [Clean and Prepare the Data](#5-clean-and-prepare-the-data)
6. [Build a Simple Baseline](#6-build-a-simple-baseline)
7. [Train Your Main Model](#7-train-your-main-model)
8. [Evaluate on Validation](#8-evaluate-on-validation)
9. [Tune the Model](#9-tune-the-model)
10. [Look at the Mistakes](#10-look-at-the-mistakes)
11. [Final Test on Held-Out Test Set](#11-final-test-on-held-out-test-set)
12. [Save and Document](#12-save-and-document)
13. [Write the Report and Slides](#13-write-the-report-and-slides)
14. [The Three-Set Workflow at a Glance](#14-the-three-set-workflow-at-a-glance)
15. [Resources](#15-resources)

---

## 1. Define the Problem

Before any code, write down:

- **What you are predicting** — the target variable.
- **What you are predicting it from** — the input features.
- **The task type** — regression, binary classification, multi-class classification, etc.
- **The single metric** that decides whether the model is good — accuracy, F1, RMSE, AUC, etc.
- **The trivial baseline** — what score you would get by predicting the majority class or the mean.

A one-sentence problem statement is enough at this stage: *"Given X, predict Y, measured by Z."*

### Tips

- Pick the metric **before** you look at any results. Picking afterwards lets you rationalise.
- Write down the trivial baseline number. Your model has to clearly beat it.
- If you cannot describe the task in one sentence, the scope is probably too wide.

---

## 2. Get the Data

Find the dataset, download it, and look at the first few rows. Record:

- **Source** — where it came from (URL, paper, internal system).
- **License** — what you are allowed to do with it.
- **Shape** — number of rows and columns.
- **Schema** — column names, types, units.

### Tips

- Keep the raw file untouched. Make all changes in code, not by editing the file.
- If the dataset is large, work on a small sample first. Switch to full data once your pipeline runs end-to-end.

For sourcing patterns and file formats, see [10_DATASETS_GUIDE.md](10_DATASETS_GUIDE.md).

---

## 3. Explore the Data (EDA)

Inspect the data before doing anything to it. Look at:

- Distribution of each feature and the target.
- Missing values per column.
- Class balance (for classification).
- Suspicious correlations (a feature too correlated with the target may be leakage).
- Sample rows of raw data to check that values make sense.

### Tips

- Just **observe and record** findings — do not start cleaning yet. Cleaning belongs in step 5, after splitting.
- If one feature correlates almost perfectly with the target, suspect leakage and investigate.
- Save your EDA plots; you will reuse them in the report.

For a full EDA workflow, see [11_EDA_GUIDE.md](11_EDA_GUIDE.md).

---

## 4. Split into Train, Validation, and Test

Split the data into **three sets** before doing any preprocessing:

| Set | Typical proportion | Purpose |
|-----|--------------------|---------|
| Train | 60–70% | Fit the model |
| Validation | 15–20% | Tune hyperparameters; compare candidate models |
| Test | 15–20% | Final, single-shot evaluation at the end |

A two-step split is the simplest way to do this:

```python
from sklearn.model_selection import train_test_split

# First split: separate the test set
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Second split: separate validation from training
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=0.2, random_state=42, stratify=y_trainval
)
```

### Tips

- **Split before preprocessing.** Cleaning, scaling, or encoding on the whole dataset leaks information from validation and test into training.
- **Lock the test set away** until step 11. Do not look at it, do not score on it, do not tune to it.
- Use `stratify=y` for classification to keep class proportions consistent across the three sets.
- For time-series data, split by time, not randomly.
- For repeated subjects (same patient/user in multiple rows), use group-aware splitting.

For split strategies, see [10_DATASETS_GUIDE.md](10_DATASETS_GUIDE.md).

---

## 5. Clean and Prepare the Data

Now that the three sets exist, build a preprocessing pipeline that applies the same transformations consistently:

- **Missing values** — drop, impute, or flag.
- **Categorical encoding** — one-hot for nominal, ordinal encoding when there is a natural order.
- **Numeric scaling** — `StandardScaler` or `MinMaxScaler` when the model needs it.
- **Feature engineering** — derived columns relevant to the task.

Wrap these steps in a scikit-learn `Pipeline` or `ColumnTransformer` so they are tied to the model and applied identically to training, validation, and test data.

### Tips

- **Fit on training data only.** Call `fit_transform` on the training set; call `transform` on validation and test.
- A `Pipeline` makes this leak-proof by construction — prefer it to manual steps.
- Tree-based models (Random Forest, XGBoost) do not need scaling; neural networks and distance-based models do.

For preprocessing techniques and pipeline patterns, see [12_DATA_PREPROCESSING_GUIDE.md](12_DATA_PREPROCESSING_GUIDE.md).

---

## 6. Build a Simple Baseline

Before training your main model, get a **simple model** running end-to-end on the same pipeline. Good baselines:

- Predict the majority class (classification) or mean (regression) — the trivial baseline from step 1.
- Logistic regression or linear regression.
- A small decision tree.

Fit the baseline on the training set, score it on the validation set, and record the number.

### Tips

- This baseline is what your main model has to beat. If your final model is only one percentage point better than logistic regression, you do not have a strong case.
- Building the baseline first also tests the pipeline. If something is broken (shape errors, leakage, wrong metric), you find it now, not after a long training run.
- Keep the baseline code and its score — both will appear in the final report.

For pipeline construction and worked examples, see [13_ML_PIPELINE_GUIDE.md](13_ML_PIPELINE_GUIDE.md).

---

## 7. Train Your Main Model

Now train the model the project is built around — typically a neural network, gradient-boosting ensemble, transformer, or similar modern method. Use the same pipeline as the baseline so results are directly comparable.

Run a **small sanity check first**: a few epochs, a small subset of the data. Confirm that the loss decreases and there are no errors before doing a full training run.

### Tips

- Set a random seed for reproducibility (`np.random.seed`, `torch.manual_seed`, `tf.random.set_seed`).
- If the model is a deep network, watch the loss for NaN or sudden jumps — usually a sign of a too-high learning rate or bad scaling.
- Train on the training set; monitor on the validation set; do not touch the test set.

For framework-specific patterns, see [13_ML_PIPELINE_GUIDE.md](13_ML_PIPELINE_GUIDE.md), [21_DEEP_LEARNING_KERAS_GUIDE.md](21_DEEP_LEARNING_KERAS_GUIDE.md), and [22_PYTORCH_GUIDE.md](22_PYTORCH_GUIDE.md). For debugging training failures, see [15_ML_DEBUGGING_GUIDE.md](15_ML_DEBUGGING_GUIDE.md).

---

## 8. Evaluate on Validation

Score the model on the **validation set** using the metric chosen in step 1. Compare against the baseline.

For more reliable estimates, run cross-validation on the combined train + validation portion (folds of that data, never crossing into the held-out test set).

### Tips

- Report performance with variance — a single number hides uncertainty.
- One run is not enough. Either train multiple seeds (deep learning) or do cross-validation (traditional ML).
- A model that beats the baseline by less than its own variance has not really beaten the baseline.

For metric definitions and cross-validation, see [14_MODEL_EVALUATION_GUIDE.md](14_MODEL_EVALUATION_GUIDE.md).

---

## 9. Tune the Model

Adjust hyperparameters to improve **validation** performance. Common starting points:

- Learning rate.
- Model size (number of layers / neurons, tree depth).
- Regularisation (dropout, L2 penalty).
- Number of training epochs / early stopping.

Use grid search or random search. Record every configuration tried and its validation score.

### Tips

- **Tune on validation only.** Never score on test during tuning.
- Change one thing at a time when you can — it makes the effect of each change interpretable.
- Stop tuning when improvements get small. Chasing the last 0.1% rarely changes the story.

For tuning patterns and search strategies, see [14_MODEL_EVALUATION_GUIDE.md](14_MODEL_EVALUATION_GUIDE.md).

---

## 10. Look at the Mistakes

Before declaring the model done, look at where it gets things wrong on the validation set:

- **Classification** — the confusion matrix, plus per-class precision and recall.
- **Regression** — the residual plot, plus the worst predictions.
- **Per-segment** — does the model perform much worse on a particular group?

Pull out specific wrong predictions and inspect them. You are looking for patterns.

### Tips

- This step almost always finds something — a data quality issue, a feature bug, or leakage hiding in plain sight.
- Do this before the final test run. If a problem is found, you can fix it without burning your one-shot test evaluation.
- Patterns in errors often become a paragraph in the report's discussion section.

For interpretation patterns, see [17_INTERPRETABILITY_GUIDE.md](17_INTERPRETABILITY_GUIDE.md). For debugging failure modes, see [15_ML_DEBUGGING_GUIDE.md](15_ML_DEBUGGING_GUIDE.md).

---

## 11. Final Test on Held-Out Test Set

Now, and only now, evaluate on the **test set**. Run the final tuned model once. Write the result down.

### Tips

- This is a **one-shot** evaluation. Do not go back and tune more after seeing the test number.
- If test performance is dramatically worse than validation, suspect overfitting to the validation set or a leak somewhere — do not just resplit and retry.
- Report the test number alongside the validation and baseline numbers in your report. The reader needs the comparison.

For the metric-reporting conventions used in the write-up, see [14_MODEL_EVALUATION_GUIDE.md](14_MODEL_EVALUATION_GUIDE.md).

---

## 12. Save and Document

Save everything needed to reproduce the result:

- Trained model file (e.g., `model.joblib`, `model.keras`, `model.pt`).
- The configuration that produced it — hyperparameters, seeds.
- A metrics file with the final scores.
- The exact requirements (`requirements.txt` or `environment.yml`).
- A README explaining how to run the code.

### Tips

- Write the README **as you work**, not at the end. Future-you will not remember the magic flag that made it run.
- Keep the model artefact, the config, and the metrics together in one folder.
- A future reader (lecturer, reviewer, employer) should be able to clone the repo, install the environment, and reproduce your numbers.

For repository layout and reproducibility practices, see [27_PROJECT_STRUCTURE_GUIDE.md](27_PROJECT_STRUCTURE_GUIDE.md).

---

## 13. Write the Report and Slides

Take the running notes, plots, and tables from previous steps and shape them into the final deliverables:

- **Report** — problem, related work, data, method, results, discussion, conclusion.
- **Slides** — a tighter version of the report, focused on what the audience should believe.

### Tips

- Keep a running notes file from step 1. By step 13 it should contain most of what the report needs.
- Reuse the plots you made during EDA and evaluation — they are already there.
- Lead with the headline number alongside the baseline. The reader needs context immediately.

For the standard report structure, see [33_REPORT_AND_PAPER_WRITING_GUIDE.md](33_REPORT_AND_PAPER_WRITING_GUIDE.md). For writing style, see [34_ACADEMIC_WRITING_STYLE_GUIDE.md](34_ACADEMIC_WRITING_STYLE_GUIDE.md). For slide design and delivery, see [35_PRESENTATION_GUIDE.md](35_PRESENTATION_GUIDE.md).

---

## 14. The Three-Set Workflow at a Glance

| Activity | Train | Validation | Test |
|----------|-------|------------|------|
| Fit preprocessing | Yes | No | No |
| Fit model | Yes | No | No |
| Monitor during training | (optional) | Yes | No |
| Compare candidate models | No | Yes | No |
| Tune hyperparameters | No | Yes | No |
| Final reported score | No | (intermediate) | Yes |

Any "Yes" in the **Test** column outside the final-score row indicates a leak.

---

## 15. Resources

- [Scikit-learn — Cross-validation and split strategies](https://scikit-learn.org/stable/modules/cross_validation.html) — official reference for `train_test_split`, K-fold, stratified, group-aware, and time-series splits.
- [Scikit-learn — Pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html) — the pattern that prevents most preprocessing leaks.
- [Scikit-learn — Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html) — official list of the same mistakes this guide warns against.

---

[← Previous: Research Proposal Writing](31_RESEARCH_PROPOSAL_GUIDE.md) | [Index](README.md) | [Next: Technical Report and Paper Writing →](33_REPORT_AND_PAPER_WRITING_GUIDE.md)
