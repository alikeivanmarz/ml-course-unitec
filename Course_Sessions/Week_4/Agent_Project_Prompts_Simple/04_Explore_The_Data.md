# Step 04 — Explore the Data

**You'll finish with:** the data loaded and checked, a data dictionary, an EDA notebook with report-quality saved figures, and a list of leakage risks.

Paste the Master Prompt first, then:

```text
STEP: LOAD AND EXPLORE THE DATA

Read PROJECT_STATE.md first. Today we understand the data — before any preprocessing
or modelling. Walk me through every finding; don't just dump output.

1. Load the dataset from the path/link I give you. Record the source link and licence
   in the README's data section so anyone could get the same data. If loading fails
   or anything looks wrong, stop and tell me — never work around it silently.

2. Basic checks, explained to me as we go: shape, column types, duplicates, impossible
   values, and how the target is distributed.

3. Data dictionary: every feature, its type (number / category / text / image / date),
   and its % missing. Save it in the notebook and to results/.

4. EDA notebook (notebooks/01_eda.ipynb) with markdown explanations:
   - target distribution and key feature distributions;
   - correlations (features with each other and with the target);
   - missing-data patterns;
   - class imbalance, if classification.
   Every figure: labelled, titled, SAVED to results/figures/, with a one-line caption
   added to results/figures/captions.md. These go straight into the report later, so
   make them good.

5. Leakage check (teach me what leakage is first, in two sentences): look for
   duplicate rows/IDs, features derived from the target or recorded after the
   outcome, and time ordering that would make a random split dishonest. Write what
   you find into PROJECT_STATE.md — we deal with it in the next step.

6. Tell me the 3 most important things this EDA says about how we should model the
   data. Update PROJECT_STATE.md and commit.

Then do the end-of-session routine.
```
