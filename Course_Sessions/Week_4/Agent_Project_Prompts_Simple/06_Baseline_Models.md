# Step 06 — Baseline Models

**You'll finish with:** the "dumb" baseline number, 1–2 trained classical models, and the start of your model-comparison table.

Paste the Master Prompt first, then:

```text
STEP: BASELINE MODELS

Read PROJECT_STATE.md first. Today we find out what "good" means — before the fancy
models. Baselines are comparison points, not the main event.

1. The dumbest baseline first: what score does predicting the majority class (or the
   mean) get on the validation set, on our headline metric? Explain why every model
   we build must clearly beat this number to mean anything.

2. Train the 1-2 classical baselines from our plan (e.g. logistic/linear regression,
   decision tree, random forest, SVM) — using the SAME preprocessing and the SAME
   splits as everything else. Default or lightly-tuned settings only; we don't
   over-invest in baselines.

3. Validation set only. The test set stays locked until step 08 — remind me why.

4. Start the comparison table: results/model_comparison.csv — one row per model, our
   metrics on validation, training time. Add the dumb baseline and the classical
   models. Every later model joins this same table.

5. A quick look at the errors of the best baseline so far: confusion matrix
   (classification) or residual plot (regression), saved to results/figures/ with a
   caption. One short paragraph together: where does it fail, and what does that hint
   for the advanced models?

6. Update PROJECT_STATE.md and commit.

Then do the end-of-session routine.
```
