# Step 07 — Advanced Models

**You'll finish with:** your main modern models trained, a documented tuning strategy, training curves, and an updated comparison table.

This is the heaviest step — **do one model per session** if training is slow. The prompt makes the agent save progress so you can stop and resume safely.

Paste the Master Prompt first, then:

```text
STEP: ADVANCED MODELS

Read PROJECT_STATE.md first — including which models (if any) are already done. Today
we build my main modern methods, one at a time. If we only finish one model this
session, that's fine: save everything, note exactly where we stopped in
PROJECT_STATE.md, and commit, so next session resumes cleanly.

For EACH main model, with me at every step:

1. Before any code: explain the method to me in plain language — how it learns, why
   it fits my task and data size, and what could go wrong (overfitting, compute,
   needing more data). If my compute makes full training unrealistic, offer cheaper
   options (transfer learning, pretrained models, a smaller version). I decide.

2. Tuning plan before training: how will we pick hyperparameters (manual / grid /
   random search), and over what ranges, given my compute budget? Options, your
   recommendation, my call. Log every setting we try and its validation score to
   results/experiments_log.csv — we'll need to describe this honestly in the report.

3. Train with the shared preprocessing, shared splits, and project seed. For neural
   models, save the training/validation curves to results/figures/ with captions.

4. Score on the VALIDATION set only and add the row to results/model_comparison.csv.
   The test set stays locked.

5. Sanity-check with me: does the score beat the baselines? If it looks too good,
   suspect leakage and check before celebrating. If it looks too bad, check the usual
   suspects (unscaled inputs, wrong loss, label mix-up) before tuning harder.

When all main models are done: show me the full table, give me your one-paragraph
reading of it, update PROJECT_STATE.md, commit.

Then do the end-of-session routine.
```
