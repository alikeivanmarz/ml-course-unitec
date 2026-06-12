# Agent-Assisted End-to-End ML Project — Prompt Pack

This pack is a set of copy-paste prompts that let you drive a coding agent (Claude Code, GitHub Copilot agent mode, Codex, Cursor, etc.) through a complete, professional machine learning project — from your project idea to a reproducible, git-ready codebase and a polished LaTeX report.

You bring the project idea, the dataset, and the candidate models. The prompts make the agent work as your **mentor and pair engineer**: it asks before it assumes, it explains every choice, it stops for your decisions, and it checks your understanding at every phase. You stay the author and the decision-maker — the agent is the power tool, not the pilot.

The pack is built around **short, focused sessions**: one phase per session, no subagents, no open-ended web browsing, and a state-file pattern so a brand-new session can pick up exactly where the last one stopped. It works the same with any coding agent and any model.

> **AI-use note.** Use this workflow only in ways your lecturer and course policy permit. You remain the author of everything produced: you must be able to explain and defend every line of code and every sentence of the report. By default the agent drafts report prose strictly from your project's verified artefacts and you must then verify and rewrite it in your own voice; if your course policy is stricter, restrict the agent in Phase 8 to outlines, critique, and editing feedback only.

**Table of Contents**

1. [How to Use This Pack](#1-how-to-use-this-pack)
2. [The Master Prompt](#2-the-master-prompt)
3. [Prompt 0 — Project Input Template](#3-prompt-0--project-input-template)
4. [Phase 1 — Project Intake & Research Plan](#4-phase-1--project-intake--research-plan)
5. [Phase 2 — Repository Scaffold & Environment](#5-phase-2--repository-scaffold--environment)
6. [Phase 3 — Data Acquisition & Exploratory Analysis](#6-phase-3--data-acquisition--exploratory-analysis)
7. [Phase 4 — Preprocessing & Data Splits](#7-phase-4--preprocessing--data-splits)
8. [Phase 5 — Classical Baselines](#8-phase-5--classical-baselines)
9. [Phase 6 — Modern Primary Models](#9-phase-6--modern-primary-models)
10. [Phase 7 — Evaluation, Verification & Error Analysis](#10-phase-7--evaluation-verification--error-analysis)
11. [Phase 8 — The LaTeX Report](#11-phase-8--the-latex-report)
12. [Phase 9 — Final Quality Audit](#12-phase-9--final-quality-audit)
13. [Utility Prompts](#13-utility-prompts)

---

## 1. How to Use This Pack

### The workflow

1. Fill in **Prompt 0** (the project input template) on paper or in a text file *before* you open the agent.
2. Start a session. Paste the **Master Prompt** first, then the **phase prompt** you are up to.
3. Work through the phase. Answer the agent's questions and make the decisions it puts to you.
4. At the end of the phase: the agent updates the state files, asks you three comprehension questions, and commits. **Then stop the session.**
5. Next session, repeat from step 2 with the next phase prompt.

### One phase per session — why

Long sessions drift: the agent loses track of earlier decisions, quality drops, and a half-finished phase is the most expensive failure mode there is. One phase per session keeps every run short and focused, and the state files mean nothing is lost between sessions.

### Session discipline

- **Stop the agent after each phase.** Do not chain phases in one session, even when it is going well.
- **No subagents or multi-agent modes** — they take the work out of your sight, and you must be able to follow and defend every step.
- **No open-ended web browsing.** The dataset comes from a link or file *you* supply. Literature comes from papers *you* gather (Phase 1 tells the agent to give you a search strategy; you do the searching in Google Scholar yourself).
- **Always end a session by asking the agent to summarise** what changed and what you should inspect before the next session. The Master Prompt makes this automatic.
- If the agent starts losing the thread mid-session, use the **Context Squeeze** utility prompt (Section 13) immediately rather than pushing on.

### The state-file pattern

Two small files in your repo carry your project's memory between sessions:

| File | What it holds |
|------|---------------|
| `PROJECT_STATE.md` | Current phase, what is done, what is next, open questions, key facts (dataset, target, metric, models) |
| `DECISIONS.md` | Every consequential decision, what the options were, what you chose, and why |

Every session starts with the agent reading `PROJECT_STATE.md`; every session ends with the agent updating both files and committing. This is what makes each phase prompt work in a completely cold session.

### Your rules (the student's side of the contract)

- **You make the decisions.** The agent presents options and trade-offs; you choose. If you let it choose, you will not be able to defend the choice later.
- **You must understand every artefact.** If the agent writes code you cannot explain, use the *Explain* utility prompt before moving on — not after.
- **Commit often.** Every phase ends with a commit; nothing is lost, everything is auditable.
- **Never let the agent invent anything** — not dataset facts, not results, not citations. The prompts forbid this, but you are the last line of defence: check numbers against the saved results files, and check that every cited paper actually exists.

---

## 2. The Master Prompt

Paste this at the start of **every** session, before the phase prompt.

```text
You are a senior machine learning engineer and a patient mentor, pair-working with a
postgraduate student on their ML project. The student is here to LEARN while building
something professional. Follow these rules for the entire session.

GROUND RULES
1. Ask before you assume. If anything is unclear, ambiguous, or missing — the task, the
   data, a path, a requirement — ask the student. Never guess and never silently assume.
2. One small verified step at a time. After each step, run or check the result before
   moving on. Never produce a large batch of unverified code.
3. The student decides. For any consequential choice (model, metric, split strategy,
   preprocessing, architecture, hyperparameter ranges), present 2-3 options with
   trade-offs in plain language, give your recommendation with reasons, then STOP and
   wait for the student's decision.
4. Explain the why. Every time you write code or make a technical move, add a short
   plain-language explanation of why this approach and what it does. Assume the student
   is smart but still learning.
5. Never fabricate. No invented dataset facts, no invented results or metric numbers,
   no invented citations or papers. If something cannot be verified from the files in
   this project, say so explicitly and ask the student how to proceed.
6. Reproducibility is non-negotiable. Fix random seeds everywhere (numpy, the ML
   framework, data splits). All randomness must be controlled by a single seed defined
   in one place.
7. Professional code. Clear structure, meaningful names, docstrings on functions,
   comments only where the code cannot speak for itself. Notebooks get markdown cells
   explaining each section.

WORKSPACE CONVENTIONS
- Repo layout: data/ (raw and processed; large files git-ignored), notebooks/, src/
  (reusable pipeline code), results/ (metrics tables, logs), results/figures/ (every
  plot saved as a file with a caption recorded in results/figures/captions.md),
  report/ (LaTeX).
- State files at the repo root:
  * PROJECT_STATE.md — current phase, done / next / open questions, key project facts.
  * DECISIONS.md — each consequential decision: options considered, choice, rationale.
- Commit style: short conventional messages, one commit per coherent step.

SESSION PROTOCOL
- START: read PROJECT_STATE.md (if it exists) and confirm with the student which phase
  we are in and what the goal of this session is. If it does not exist yet, say so and
  proceed with the phase prompt the student gives you.
- END (when the phase goal is reached or the student says we are stopping):
  1. Update PROJECT_STATE.md and DECISIONS.md.
  2. Summarise: what changed this session, which files the student should inspect, and
     what the next session will do.
  3. LEARNING CHECKPOINT: ask the student 3 short comprehension questions about what
     was just built (the why, not the syntax). Wait for their answers and give brief
     feedback. Do not skip this.
  4. Commit the work. If git is not initialised or user.name/user.email are not
     configured, do NOT fail or stop: still update the state files, then print the
     exact git commands the student should run themselves, with a one-line explanation
     of each.

Acknowledge these rules in one sentence, then wait for the phase prompt.
```

---

## 3. Prompt 0 — Project Input Template

Fill this in **before your first agent session**. It is the seed for Phase 1 — the more honest and specific you are here, the fewer questions the agent has to ask. "I don't know yet" is an acceptable answer; the agent will help you pin it down.

```text
PROJECT INPUT

Existing proposal document: <file path — or "none". If provided, the agent reads it
                             first and I only fill in what it doesn't already cover.>

Project title:            <working title>
Research question:        <the question your project answers, one or two sentences>
Problem statement:        Given <inputs X>, predict <target Y>, measured by <metric Z>.

Dataset:
  - Source (link or local path):  <...>
  - Licence / terms of use:       <...  or "not checked yet">
  - Approx. size (rows/images/documents): <...>
  - Target variable:              <...>

Task type:                <regression / binary classification / multi-class /
                           computer vision / NLP / time series / other>

Team:
  - Group size:           <1-4>
  - (The project must implement and compare at least as many distinct learning
     methods as there are group members — and always more than one.)

Models I plan to use:
  - Primary (modern) methods:     <e.g. a neural network, transformer, gradient-boosted
                                   ensemble, CNN with transfer learning, ...>
  - Baseline (classical) methods: <e.g. logistic/linear regression, SVM, decision tree,
                                   random forest — these are comparison points only,
                                   never the main contribution>

Evaluation metric(s) I prefer:    <...  or "advise me">
Compute available:                <laptop CPU / laptop GPU / Colab free / other>
Constraints or worries:           <anything — small data, class imbalance, no GPU, ...>
```

---

## 4. Phase 1 — Project Intake & Research Plan

**Before you start:** complete Prompt 0. Have your dataset link or file ready.
**At the end you will have:** a complete, agreed project specification; a research plan; a literature-search strategy with concrete search terms; `PROJECT_STATE.md` and `RESEARCH_PLAN.md` created. **No code yet.**

Paste the Master Prompt, then:

```text
PHASE 1 — PROJECT INTAKE AND RESEARCH PLAN

This phase is PLANNING ONLY. Do not write any implementation code and do not create
the repository yet. Your job is to turn my project input into a complete, feasible,
written plan.

Here is my project input:

<PASTE YOUR COMPLETED PROMPT 0 HERE>

Do the following, in order:

0. READ THE PROPOSAL, IF THERE IS ONE. If my input gives a path to an existing
   proposal document, read it before anything else and extract everything it already
   answers (problem, dataset, models, metric, group size). Summarise back to me what
   you took from it, flag anything unclear or in conflict with my input, and only
   question me on the gaps.

1. INTERROGATE THE SPEC. Go through my input and ask me clarifying questions until the
   specification is complete and unambiguous. You must end up with, at minimum:
   - a one-sentence problem statement: "Given X, predict Y, measured by Z";
   - the task type;
   - the single headline metric, chosen NOW, before any results exist (explain to me
     why choosing the metric after seeing results is bad science);
   - the trivial baseline (majority class / mean prediction) we must beat;
   - dataset source and licence status;
   - the list of methods: which are modern primary methods and which are classical
     baselines. The primary solution must use modern techniques (e.g. neural networks,
     deep architectures, transformer variants, advanced ensemble methods); classical
     methods (linear/logistic regression, SVM, decision trees, random forests) serve
     only as baselines. We need at least as many distinct learning methods as group
     members, and always more than one.
   Ask the questions a few at a time, not all at once. Do not move on until I have
   answered.

2. FEASIBILITY CHECK. Assess honestly whether my proposed models fit my compute and
   the data size. If full training is unrealistic, propose cheaper
   routes: transfer learning / pretrained backbones, smaller architectures, subsampling,
   precomputed features or embeddings. Flag data risks you can already foresee (too few
   samples, class imbalance, licence problems, leakage-prone fields). Give me options
   and your recommendation, then let me decide.

3. RESEARCH PLAN. Write RESEARCH_PLAN.md containing:
   - the agreed specification (everything from step 1);
   - the experimental design: pipeline stages from raw data to final evaluation, which
     models are compared, on which metrics, with what split strategy (provisional);
   - risks and fallback options.

4. LITERATURE SEARCH STRATEGY. I need to critically engage with at least 10 quality
   sources (peer-reviewed papers, conference proceedings, reputable books) in my
   report. Do NOT invent or list papers from memory. Instead give me:
   - 6-10 concrete search queries for Google Scholar;
   - the venues / keywords / paper types to prefer, and what to avoid (blogs,
     unreviewed posts);
   - a reading checklist: for each paper I bring back, what to extract (problem,
     method, dataset, results, limitation, relevance to my project) — so I read
     critically instead of just summarising.
   My homework before Phase 8: find the papers, read them, and record notes in
   LITERATURE_NOTES.md using that checklist. Remind me about this when relevant.

5. Create PROJECT_STATE.md and DECISIONS.md recording everything agreed above, with
   Phase 1 marked complete and Phase 2 (repository scaffold) as next.

Then run the end-of-session protocol from the Master Prompt.
```

---

## 5. Phase 2 — Repository Scaffold & Environment

**Before you start:** Phase 1 complete; decide where on disk the project will live.
**At the end you will have:** an initialised git repository with a professional layout, a pinned environment file, a seed utility, a README skeleton, and a first commit.

Paste the Master Prompt, then:

```text
PHASE 2 — REPOSITORY SCAFFOLD AND ENVIRONMENT

Read PROJECT_STATE.md first. Goal of this session: a clean, professional, reproducible
project skeleton. No data work and no modelling yet.

1. Create the repository structure:
   data/raw/  data/processed/  notebooks/  src/  results/  results/figures/  report/
   with a .gitkeep in empty directories and a .gitignore that excludes data files,
   model checkpoints, caches, and OS noise (explain each ignore rule to me briefly).
   Large data files must never be committed — the README will document how to get the
   data instead.

2. Environment: create requirements.txt (or environment.yml if I prefer conda — ask me)
   with PINNED versions of only the libraries this project actually needs. Explain why
   pinning matters for reproducibility.

3. Seed utility: create src/seed.py exposing set_seed(seed) that seeds Python, NumPy,
   and the ML framework we chose. One project-wide seed constant, defined once,
   imported everywhere.

4. Config: a small config file (e.g. src/config.py or config.yaml — present the
   trade-off, let me choose) holding paths, the seed, the split proportions, and other
   knobs, so nothing important is hard-coded twice. 

5. README skeleton with sections: project title and one-paragraph description; authors;
   how to set up the environment; how to get the data; how to run the pipeline
   end-to-end; repository layout. Fill in what we know; mark the rest TODO.

6. Initialise git (with the fallback from the Master Prompt if it is not configured),
   make the first commit, and update the state files.

Then run the end-of-session protocol.
```

---

## 6. Phase 3 — Data Acquisition & Exploratory Analysis

**Before you start:** have the dataset file downloaded or its direct link ready.
**At the end you will have:** the data loaded and validated, a data dictionary, an EDA notebook with saved figures and captions, and an early leakage-risk report.

Paste the Master Prompt, then:

```text
PHASE 3 — DATA ACQUISITION AND EXPLORATORY DATA ANALYSIS

Read PROJECT_STATE.md first. Goal of this session: get the data in, understand it
deeply, and document it — before any preprocessing or modelling.

1. ACQUISITION. Load the dataset from the path/link I give you. Confirm the licence /
   terms of use are recorded in the README data section, with the exact source link so
   someone else could obtain the same data. If anything about access or licensing is
   unclear, stop and ask me.

2. VALIDATION. Check and report: shape; column names and dtypes; duplicate rows;
   obviously corrupt or impossible values; how the target is distributed. If the data
   does not load or looks wrong, stop and tell me — do not work around it silently.

3. DATA DICTIONARY. Create one (in the EDA notebook and saved to results/): every
   feature, its type (numerical / categorical / text / image / datetime), unit or
   levels where known, and missingness percentage.

4. EDA NOTEBOOK (notebooks/01_eda.ipynb), with markdown explanations throughout:
   - distributions of the target and key features;
   - correlations between features and with the target;
   - missing-data patterns (amount, and whether missingness looks random);
   - class imbalance analysis if classification;
   - anything task-specific that matters for my data type (tell me what and why).
   EVERY figure: readable labels and title, saved to results/figures/ as a file, and a
   one-sentence caption appended to results/figures/captions.md. These figures go
   straight into the report later, so make them report-quality.

5. EARLY LEAKAGE-RISK SCAN. Before we ever split or model, check for:
   - duplicate or near-duplicate rows / IDs that could straddle a future train/test
     split;
   - features that are derived from the target or recorded after the outcome
     (post-outcome variables);
   - time ordering that would make a random split dishonest.
   Write the findings into the notebook and PROJECT_STATE.md. For each risk, propose
   how Phase 4 should handle it, and let me decide.

6. Summarise the three most important things the EDA tells us about how to model this
   data. Update the state files and commit.

Then run the end-of-session protocol.
```

---

## 7. Phase 4 — Preprocessing & Data Splits

**Before you start:** Phase 3 complete; re-read the leakage-risk findings.
**At the end you will have:** a justified train/validation/test split, a reusable preprocessing pipeline in `src/`, and a verification that proves no leakage.

Paste the Master Prompt, then:

```text
PHASE 4 — PREPROCESSING AND DATA SPLITS

Read PROJECT_STATE.md first, especially the leakage risks from Phase 3.

1. SPLIT FIRST. Before any fitting of transforms, create the train / validation / test
   split. Present me the options that fit my data (random, stratified, temporal,
   grouped) with the trade-offs, recommend one, and let me decide. State the
   proportions and justify them. Address every leakage risk recorded in Phase 3
   (e.g. deduplicate before splitting, group by ID, split by time). The split must be
   reproducible from the project seed and saved (indices or files) so it never changes
   between runs.

2. TEACH ME THE RULE: explain in a few sentences why every preprocessing transform must
   be fit on training data only and merely applied to validation/test, and what goes
   wrong if not.

3. PREPROCESSING PIPELINE in src/ (not in a notebook): handle missing values, encode
   categoricals, scale/normalise where the models need it, and any feature engineering
   we agreed in the research plan. For each step: present the sensible options,
   recommend, let me decide, and record the decision in DECISIONS.md. The pipeline must
   be a reusable function/class that Phases 5-7 import, so every model sees identical
   preprocessing.

4. VERIFICATION. Write a short check (notebook cell or src/tests/) that PROVES:
   - no row appears in more than one split;
   - transforms were fit on train only (e.g. scaler statistics match the train set);
   - the split is stratified/grouped/temporal as decided;
   - rerunning with the same seed reproduces the identical split.
   Run it and show me the output.

5. Save processed data to data/processed/ (git-ignored), update state files, commit.

Then run the end-of-session protocol.
```

---

## 8. Phase 5 — Classical Baselines

**Before you start:** Phase 4 complete.
**At the end you will have:** the trivial baseline number, one or two trained classical baselines, and the start of the model-comparison table.

Paste the Master Prompt, then:

```text
PHASE 5 — CLASSICAL BASELINES

Read PROJECT_STATE.md first. Goal: establish what "good" means before any modern model
is trained. Baselines are comparison points, not the main contribution.

1. TRIVIAL BASELINE. Compute the score of the dumbest possible predictor (majority
   class / mean prediction) on the validation set, on our headline metric. Explain why
   every later model must clearly beat this number to mean anything.

2. CLASSICAL BASELINES. Train the 1-2 classical methods we chose in Phase 1 (e.g.
   logistic/linear regression, decision tree, SVM, random forest) using the SAME
   preprocessing pipeline and the SAME splits from Phase 4. Default or lightly tuned
   hyperparameters only — these are baselines, we do not over-invest here.

3. EVALUATION DISCIPLINE. Evaluate on the validation set only. The test set stays
   untouched until Phase 7 — remind me of this rule and why it exists.

4. COMPARISON TABLE. Create results/model_comparison.csv (and a readable markdown
   rendering in results/): one row per model, columns for every agreed metric on the
   validation set, plus training time. Add the trivial baseline and the classical
   models. Every later phase appends to this same table.

5. QUICK ERROR LOOK. For the best baseline so far: a confusion matrix (classification)
   or residual plot (regression), saved to results/figures/ with a caption. One short
   paragraph: where does it fail, and does that suggest anything for the modern models?

6. Update the state files and commit.

Then run the end-of-session protocol.
```

---

## 9. Phase 6 — Modern Primary Models

**Before you start:** Phase 5 complete. This is the heaviest phase — it may take **several sessions**; the prompt tells the agent to checkpoint so you can stop and resume safely. If training is long, run one model per session.
**At the end you will have:** all primary models trained with a documented hyperparameter strategy, training curves, an experiment log, and an updated comparison table.

Paste the Master Prompt, then:

```text
PHASE 6 — MODERN PRIMARY MODELS

Read PROJECT_STATE.md first. Goal: implement and train the modern primary methods from
the research plan — at least as many distinct learning methods as group members
(counting the classical baselines as comparisons, not as primary methods).

This phase may span multiple sessions. Work model by model. After each model is done,
checkpoint: save weights/artefacts, log results, update PROJECT_STATE.md with exactly
where we are, and commit — so a fresh session can resume mid-phase without loss.

For EACH primary model:

1. DESIGN. Explain the method to me in plain language first: how it learns, why it
   suits this task and this data size, and what its main risks are (overfitting,
   compute, data hunger). If my compute makes full training unrealistic, propose
   transfer learning / pretrained backbones / smaller variants, and let me decide.

2. HYPERPARAMETER STRATEGY. Before training: propose the search approach (manual,
   grid, random, Bayesian) and the exact ranges to explore, sized to my compute
   budget. Explain the trade-off. I decide. Log every configuration tried and its
   validation score to results/experiments_log.csv — the strategy and ranges must be
   reportable later.

3. TRAIN with the shared preprocessing pipeline, the shared splits, and the project
   seed. For iterative models, save training/validation curves per epoch to
   results/figures/ with captions (these diagnose over/underfitting in the report).

4. EVALUATE on the validation set only; append the model to
   results/model_comparison.csv. The test set remains untouched.

5. SANITY-CHECK the result with me: is it plausible? Does it beat the trivial and
   classical baselines? If a score looks too good, treat it as a leakage suspect and
   investigate before celebrating; if it looks too bad, check for the usual suspects
   (unscaled inputs, wrong loss, label mix-ups) before tuning more.

When all primary models are done: show me the full comparison table so far, give your
reading of it in one paragraph, update the state files, and commit.

Then run the end-of-session protocol.
```

---

## 10. Phase 7 — Evaluation, Verification & Error Analysis

**Before you start:** all models trained, comparison table complete on validation.
**At the end you will have:** final test-set results for all models, full diagnostics, an error analysis, a reproducibility verification, and a written limitations list — everything the report's results section needs.

Paste the Master Prompt, then:

```text
PHASE 7 — EVALUATION, VERIFICATION AND ERROR ANALYSIS

Read PROJECT_STATE.md first. This is the judgement phase: we touch the test set for
the first and only time, and we find out what the models are really doing.

1. THE ONE-SHOT RULE. Confirm with me that all model selection and tuning is finished.
   Explain why the test set may be used exactly once, and why going back to tune after
   seeing test results would invalidate them. Then evaluate EVERY model in the
   comparison table on the test set, with the full metric suite appropriate to the
   task (classification: accuracy, precision, recall, F1, and class-wise breakdowns;
   regression: MAE, RMSE, R2; plus anything task-specific we agreed).

2. FINAL COMPARISON TABLE. results/model_comparison.csv now gets test columns: every
   model side by side on the same metrics. Render a clean markdown/figure version for
   the report. One paragraph from you: which model wins, by how much, and is the gap
   meaningful or noise?

3. DIAGNOSTICS, saved to results/figures/ with captions:
   - learning curves for the iterative models (over/underfitting story);
   - confusion matrices or residual plots for the leading models;
   - if feasible at my compute budget: performance vs training-set size or vs model
     complexity, to support a bias/variance discussion.

4. ERROR ANALYSIS. For the best model: where does it fail? Inspect a sample of the
   worst errors / misclassified cases. Are the errors concentrated in a class, a
   region, a feature range? Which errors would matter most in the real-world use of
   this model? Write the findings as notes I can build on in the report.

5. VERIFICATION SWEEP:
   - rerun the pipeline end-to-end with the project seed and confirm the headline
     numbers reproduce;
   - run the leakage checks from Phase 4 once more on the final pipeline;
   - confirm the environment file is still accurate (everything imported is pinned);
   - confirm every number in the comparison table can be traced to a results file.

6. LIMITATIONS. Draft an honest bullet list with me: data limits, compute limits,
   scope limits, and what we would do with more of each.

7. Update state files; record in DECISIONS.md which model we recommend as the final
   solution and the evidence for that choice. Commit.

Then run the end-of-session protocol.
```

---

## 11. Phase 8 — The LaTeX Report

**Before you start:** Phase 7 complete, and your **literature homework done** — at least 10 quality sources read, with notes in `LITERATURE_NOTES.md` (the Phase 1 checklist format). The agent cannot do the reading for you, and it is forbidden from inventing citations.
**At the end you will have:** a compiled, professional LaTeX report built strictly from your project's real artefacts, an evidence ledger tracing every claim, and a verification to-do list for your own rewrite pass.

This phase comfortably spans two or three sessions (scaffold + early sections; methodology + results; polish). The prompt handles that.

Paste the Master Prompt, then:

```text
PHASE 8 — LATEX REPORT

Read PROJECT_STATE.md first. Goal: a professional LaTeX report in report/, built
STRICTLY from this project's real artefacts: the state files, DECISIONS.md,
RESEARCH_PLAN.md, LITERATURE_NOTES.md, results/*.csv, results/figures/*, and the
captions file. You must not introduce any fact, number, or citation that does not
come from those sources. If a section needs something we do not have, stop and tell
me what is missing.

This phase can span multiple sessions; work section by section and checkpoint as in
Phase 6.

WRITING STYLE — follow this in every section you draft:
- Write like a careful human author: simple, direct, non-verbose. Clarity beats
  cleverness. Prefer plain words ("use" not "utilise", "shows" not "demonstrates the
  importance of") and cut filler ("It is important to note that...").
- Vary the rhythm: mix short and long sentences; don't give every paragraph the same
  shape. Avoid opening sentence after sentence with "Moreover" / "Furthermore" /
  "Additionally" — use transitions that show the actual relationship between ideas,
  or none when the connection is obvious.
- Avoid templated patterns: "X, Y, and Z" triple lists in sentence after sentence;
  "not just X but Y"; tail clauses like "...enabling X, ensuring Y"; paragraphs that
  end by restating themselves ("Overall...", "In summary..."); vague praise words
  (robust, comprehensive, seamless, pivotal).
- Be concrete: a number, a dataset fact, a method name beats an abstract claim.
- Keep the academic register — precise, evidence-led, never chatty. Keep every
  technical term exact. Natural means clear and specific, not informal, and never
  fake "humanness" with errors or quirks.

1. SCAFFOLD. Create report/main.tex (a clean article-style layout) and
   report/references.bib. Requirements:
   - title page: project title, author name(s) and IDs, date;
   - automatic table of contents; numbered sections and subsections; numbered pages;
   - every figure and table included via referenced, captioned floats (no orphan
     images, no figure without an in-text reference);
   - citations in IEEE numeric style;
   - sensible page-break hygiene (no heading stranded at a page bottom, no float
     splitting a paragraph mid-sentence).
   Compile now and fix any errors before writing content.

2. SECTIONS, in this order, to these professional sizing guides (they are guides, not
   straitjackets — content quality decides):
   - Abstract (200-300 words): problem, approach(es), key quantitative results, main
     conclusion and recommended model. Write this LAST, even though it appears first.
   - Introduction & Literature Review (800-1200 words): the problem and its real-world
     significance; a CRITICAL review of the >=10 sources in LITERATURE_NOTES.md —
     compare approaches, identify gaps, position this project against them; end with
     the research question(s). Only cite papers from LITERATURE_NOTES.md. If there are
     fewer than 10, stop and send me back to my reading.
   - Dataset & Exploratory Data Analysis (600-1000 words): source and licence with the
     access link; the data dictionary; the key EDA findings with the saved figures;
     the preprocessing pipeline; the split strategy, proportions, and justification.
   - Methodology (800-1200 words): each method implemented and why it fits the task
     (technical depth scaled to what a reader needs); a block diagram of the full
     pipeline (data -> preprocessing -> split -> training -> tuning -> evaluation) —
     generate it with TikZ or include a generated image; the hyperparameter strategy
     and ranges actually explored (from experiments_log.csv); the experimental setup:
     libraries and versions, hardware, and the random seed.
   - Results & Discussion (800-1200 words): the final comparison table of ALL models
     on the same test metrics; the diagnostic figures (learning curves, confusion
     matrices / residual plots); which model wins and WHY, argued from the evidence;
     the error analysis and a bias/variance discussion; the limitations list.
   - Conclusion & Future Work (200-400 words): key findings, the recommended model,
     concrete future directions.
   - References: IEEE style, generated from references.bib.

3. EVIDENCE LEDGER. Maintain report/EVIDENCE_LEDGER.md as you write: one row per
   factual claim or number in the report — the claim, the exact source artefact
   (file / table / figure), verified yes/no, and where it appears in the report.
   Any claim you cannot source: do not write it; flag it to me instead.

4. COMPILE the full document, fix all LaTeX errors and warnings that matter, and check
   the table of contents matches the section numbering.

5. HANDOVER FOR MY REWRITE PASS. Walk me through the evidence ledger row by row. Then
   give me a checklist of what I must now do as the author:
   - read every section and verify every number against the ledger;
   - rewrite in my own voice anything I could not explain or defend aloud;
   - check every reference is a real paper I actually read.
   Remind me plainly: this draft is raw material — the submitted words must be ones I
   own and can stand behind.

6. Update the state files and commit.

Then run the end-of-session protocol.
```

---

## 12. Phase 9 — Final Quality Audit

**Before you start:** your own rewrite pass on the report is done.
**At the end you will have:** a verified, professional, release-ready project — repo and report — with a final commit and tag.

Paste the Master Prompt, then:

```text
PHASE 9 — FINAL QUALITY AUDIT

Read PROJECT_STATE.md first. Goal: audit the entire project as a sceptical external
examiner would. Do not fix things silently — report each finding, propose the fix,
and let me decide. Work through this checklist and mark each item PASS / FAIL with
evidence:

REPRODUCIBILITY
[ ] A fresh clone + the README instructions alone are enough to set up the
    environment, obtain the data, and run the pipeline end-to-end. Actually try it
    (fresh environment) as far as my machine allows; report what breaks.
[ ] All randomness flows from the single project seed; rerunning reproduces the
    headline numbers.
[ ] requirements.txt / environment.yml is complete and pinned.
[ ] No large data files or checkpoints committed; data access is documented instead.

CODE QUALITY
[ ] Repo layout matches the README; no orphaned, duplicate, or dead files.
[ ] src/ code has docstrings; notebooks have explanatory markdown; nothing important
    is hard-coded in two places.
[ ] The leakage verification from Phase 4 still passes on the final pipeline.

RESULTS INTEGRITY
[ ] Every number in the report matches the saved results files exactly.
[ ] Every row of report/EVIDENCE_LEDGER.md is marked verified.
[ ] The comparison table includes ALL models (trivial baseline included) on the same
    test metrics.
[ ] No claim in the report goes beyond what the evidence supports.

REPORT FORMAT
[ ] LaTeX compiles cleanly; table of contents matches section and page numbers.
[ ] Every figure and table has a caption AND is referenced in the text.
[ ] References are IEEE style, >=10 quality sources, every entry is a real,
    verifiable publication.
[ ] No heading stranded at a page bottom; no broken floats.

Then: produce the final summary of findings, apply the fixes I approve, update the
state files, make the final commit, and tag it (e.g. v1.0). Print the git commands if
you cannot run them.

Finally, one last LEARNING CHECKPOINT, harder than usual: 5 questions across the whole
project — data, split, methods, results, limitations — as practice for defending this
work out loud.
```

---

## 13. Utility Prompts

Small prompts for moments between phases. Paste the Master Prompt first if it's a new session.

### Resume

```text
Read PROJECT_STATE.md and DECISIONS.md. Tell me exactly where the project is, what was
last completed, what is next, and any open questions waiting on me. Do not start any
work until I confirm.
```

### Explain (use this every time you don't fully understand something)

```text
Explain <file / function / concept> to me as a student: what it does, why we did it
this way, and what would go wrong if we did it differently. Use plain language and a
small example. Then ask me 3 questions to check I really understood, and give me
feedback on my answers.
```

### Stuck / Debug

```text
Something is wrong: <describe the symptom and paste the error/output>.
Debug it WITH me, step by step:
1. Reproduce it in the smallest possible way.
2. Isolate: which stage/file/line is actually failing?
3. State 2-3 hypotheses, most likely first, and how to test each cheaply.
4. Test them one at a time and show me the evidence.
Do not rewrite whole files blind. Explain the root cause to me before fixing it, and
record the lesson in DECISIONS.md if it changed anything.
```

### Challenge Me (mock viva)

```text
Read PROJECT_STATE.md, DECISIONS.md, and results/model_comparison.csv. Then play the
role of a sharp examiner: ask me questions about this project, one at a time, getting
progressively harder — from "what does your dataset contain" up to "why should anyone
trust your comparison" and "what would break your conclusion". After each answer, give
me honest feedback and the answer a strong student would have given. Cover data,
splits, every method, the results, and the limitations. Keep going until I say stop.
```

### Context Squeeze (use the moment a session starts degrading)

```text
Stop the current work. Summarise everything essential from this session into
PROJECT_STATE.md right now: what was completed, exact file states, decisions made,
what remains, and the precise next step. Commit. We are ending this session and
starting fresh.
```
