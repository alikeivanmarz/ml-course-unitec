# Step 08 — Evaluate and Analyse

**You'll finish with:** final test results for every model, the diagnostic figures, an error analysis, and an honest limitations list — everything the report's results section needs.

Paste the Master Prompt first, then:

```text
STEP: FINAL EVALUATION AND ERROR ANALYSIS

Read PROJECT_STATE.md first. Today we unlock the test set — once, and only once.

1. Confirm with me that all tuning and model selection is finished. Explain why the
   test set can be used exactly once, and why tuning after seeing test results would
   make them meaningless. Then evaluate EVERY model in the comparison table on the
   test set with the full set of suitable metrics (classification: accuracy,
   precision, recall, F1; regression: MAE, RMSE, R2; plus anything specific to my
   task).

2. Finish the comparison table: all models side by side on the same test metrics.
   Make a clean version for the report. Discuss with me: which model wins, by how
   much, and is that gap real or noise?

3. Diagnostics, saved to results/figures/ with captions: learning curves for the
   neural/iterative models, confusion matrices or residual plots for the leaders.

4. Error analysis on the best model, together: look at a sample of its worst
   mistakes. Are they concentrated somewhere (a class, a range, a group)? Which
   mistakes would matter most if this model were used for real? Keep written notes —
   they go into the report.

5. Trust check: rerun the pipeline with the project seed and confirm the headline
   numbers reproduce; rerun the leakage check from step 05; confirm every number in
   the table traces back to a saved results file.

6. Draft an honest limitations list with me: data limits, compute limits, scope
   limits. Record in PROJECT_STATE.md which model we recommend as the final answer
   and the evidence why. Commit.

Then do the end-of-session routine.
```
