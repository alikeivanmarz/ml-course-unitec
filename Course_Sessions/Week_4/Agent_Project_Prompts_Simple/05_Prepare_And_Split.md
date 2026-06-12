# Step 05 — Prepare and Split

**You'll finish with:** a justified train/validation/test split, a reusable preprocessing pipeline, and a check that proves there's no leakage.

Paste the Master Prompt first, then:

```text
STEP: SPLIT THE DATA AND BUILD THE PREPROCESSING

Read PROJECT_STATE.md first, especially the leakage risks from the last step.

1. SPLIT FIRST, before fitting anything. Show me the split options that fit my data
   (random / stratified / temporal / grouped) with simple pros and cons, recommend
   one, and let me choose. State the proportions and why. Handle every leakage risk
   we found (deduplicate first, group by ID, split by time — whatever applies). The
   split must come from the project seed and be saved, so it never changes.

2. Teach me the golden rule in a few sentences: preprocessing is FIT on training data
   only, then APPLIED to validation and test — and what goes wrong otherwise.

3. Build the preprocessing as reusable code in src/ (not buried in a notebook):
   missing values, encoding categories, scaling where needed, plus any feature
   engineering from our plan. For each choice: options, recommendation, my decision,
   noted in PROJECT_STATE.md. Every model in later steps uses this same pipeline.

4. Prove it's clean. Write and run a small check showing: no row is in two splits;
   the scaler/encoder statistics come from the training set only; the same seed gives
   the identical split twice. Show me the output.

5. Save processed data to data/processed/ (git-ignored), update PROJECT_STATE.md,
   commit.

Then do the end-of-session routine.
```
