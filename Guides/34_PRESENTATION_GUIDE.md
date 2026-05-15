# Presenting Technical ML Projects

A presentation about a machine learning project has a specific job: convey what was attempted, what was found, and what the audience should now believe or do — within a fixed time and to listeners with varied technical backgrounds. This guide covers structure, slide design for technical content, honest reporting of metrics, demo discipline, and Q&A handling.

**Table of Contents**

1. [Standard Talk Structure](#1-standard-talk-structure)
2. [Slide Design for Technical Content](#2-slide-design-for-technical-content)
3. [Communicating Metrics Honestly](#3-communicating-metrics-honestly)
4. [Live Demos](#4-live-demos)
5. [Handling Q&A](#5-handling-qa)
6. [Common Anti-Patterns](#6-common-anti-patterns)
7. [Resources](#7-resources)

---

## 1. Standard Talk Structure

A reliable structure for a 10–20 minute technical talk:

| Section | Duration (12 min) | Purpose |
|---------|-------------------|---------|
| Problem | 1.5 min | Why this work matters; who it serves |
| Data | 1.5 min | Source, size, key characteristics, preprocessing summary |
| Method | 2 min | Approach chosen and why; alternatives considered briefly |
| Results | 3 min | Headline metric, comparison to baseline, error analysis |
| Limitations | 1 min | Where the model fails; honest scope |
| Next steps | 1 min | What would be done with more time or data |
| Q&A | 2 min | Discussion |

The Problem section is the highest-leverage slide. An audience that does not understand why the work matters cannot evaluate the rest.

### 1.1 Time Budget

| Talk length | Slide count (approx.) |
|-------------|------------------------|
| 5 min | 5–7 |
| 10 min | 8–12 |
| 15 min | 12–18 |
| 30 min | 20–25 |

A common rule of thumb: one minute per content slide, plus title and Q&A. Dense slides take longer per slide than sparse ones; account for this in rehearsal.

### 1.2 The "What Should They Believe?" Test

Before any slide is built, the presenter should be able to complete this sentence:

> "After this talk, the audience should believe that __________ and should consider __________."

If the answer is unclear, the presentation lacks a thesis. Every slide should defend or build toward those conclusions; slides that do neither belong in an appendix.

---

## 2. Slide Design for Technical Content

### 2.1 Signal-to-Ink

| Practice | Reason |
|----------|--------|
| One idea per slide | The audience reads while listening; competing ideas split attention |
| Headline is a sentence, not a topic | "Random Forest beats Logistic Regression by 8% F1" beats "Results" |
| Remove decorative chrome | Borders, drop shadows, gradients distract from data |
| Chart axes labelled with units | Audience cannot ask "what's the y-axis?" mid-talk |
| Font size ≥ 24pt for body, ≥ 36pt for headlines | Legible from the back of a typical room |

### 2.2 Plot Legibility

Plots designed for a notebook are usually unreadable in a presentation. Adjustments:

- **Increase font sizes** before screenshotting; matplotlib's default is too small for projection.
- **Limit categories** shown — top 5 or 10, group the rest.
- **Annotate the point being made** directly on the plot (arrow, callout box) rather than in a caption.
- **Choose a colour palette that survives projection**. Avoid yellow on white; pale colours wash out.

```python
import matplotlib.pyplot as plt

# Presentation-ready defaults
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 150,
})
```

### 2.3 Code on Slides

Show code only when the code itself is the point — typically a critical algorithmic step or an unusual implementation detail. A few principles:

- Highlight the lines that matter; grey the rest.
- Strip imports and boilerplate.
- Keep to ≤ 10 lines visible at one time.

If the code is more than 10 lines, summarize its behaviour in prose and offer the full source in an appendix or repository link.

---

## 3. Communicating Metrics Honestly

### 3.1 Lead with the Right Metric

The headline metric should be the one that matters for the application — not the one with the highest number. For an imbalanced classification task, accuracy of a "predict majority" baseline is high and meaningless; F1 or ROC AUC carry the real signal.

| Task | Typical headline metric |
|------|--------------------------|
| Balanced classification | Accuracy, F1 |
| Imbalanced classification | F1 (macro or per-class), PR AUC, recall at fixed precision |
| Probabilistic classification | Log-loss, Brier score, calibration |
| Regression | RMSE, MAE; report MAPE only if zeros are absent |
| Ranking / retrieval | NDCG, MRR, recall@K |
| Generation | Task-specific (BLEU/ROUGE/CIDEr); add human evaluation |

### 3.2 Always Report a Baseline

A model's score is meaningless without a reference point. Report at least one of:

- **Trivial baseline**: predict majority class, predict the mean, random guess.
- **Simple baseline**: linear or logistic regression, nearest neighbour.
- **Prior-art baseline**: published results on the same dataset.

A 92% accuracy claim collapses when the trivial baseline is 91%.

### 3.3 Quantify Uncertainty

A single metric value invites overinterpretation. Report intervals:

- **Cross-validation standard deviation** for traditional ML.
- **Bootstrap confidence intervals** for held-out metrics.
- **Multiple training seeds** for deep learning, reporting mean ± std.

A difference of 0.5% F1 between two models is not meaningful if either has a 2% standard deviation across seeds.

### 3.4 Disaggregated Performance

Headline metrics hide subgroup failures. Report performance broken down by:

- Class (precision/recall per class)
- Demographic or relevant grouping if applicable
- Difficulty or input length (for generation tasks)
- Time period (for time-series)

A model that is 95% accurate overall but 60% accurate on a critical subgroup is a different model than the headline implies.

---

## 4. Live Demos

Live demos are high-risk and high-reward. They communicate capability viscerally but fail in ways that consume Q&A time and undermine the rest of the talk.

### 4.1 Failure Modes and Mitigations

| Failure mode | Mitigation |
|--------------|------------|
| Network outage | Self-contained local demo; no external API calls if possible |
| Cold-start latency | Warm the model and the UI before the talk begins |
| Unexpected input crashes the demo | Constrain inputs (form fields, dropdowns) rather than free text |
| Model produces an embarrassing output | Curated example inputs known to behave well; have a recorded fallback |
| Audience requests an input the demo cannot handle | Note the request; offer to follow up; do not improvise |

### 4.2 Demo Structure

A live demo should answer one question quickly: "does this work as claimed?" Three minutes is a long demo. Structure:

1. **Set the expectation** — what input goes in, what output comes out.
2. **Show one good case** — the model works as described.
3. **Show one revealing case** — an edge case, a failure mode, a strength.
4. **Stop and explain** — what the audience should take away.

A pre-recorded video with narration is often a better choice than live execution for high-stakes presentations.

---

## 5. Handling Q&A

### 5.1 Calibrated Answers

| Confidence | Phrasing |
|------------|----------|
| Confident | "Yes — the model achieves X under Y conditions." |
| Partially confident | "We saw evidence of X, but did not test Y exhaustively." |
| Unsure | "I don't know. My intuition is X, but I would want to check before answering." |
| Outside scope | "That wasn't tested in this work; it's a reasonable next step." |

"I don't know" is a complete answer. Fabricating an answer to seem authoritative damages credibility more than admitting a gap.

### 5.2 Difficult Questions

| Question type | Response pattern |
|---------------|------------------|
| Hostile or rhetorical | Address the literal question, ignore the framing |
| Multi-part | Answer the most important part; offer to take the rest offline |
| Beyond scope | Acknowledge briefly; redirect to what was tested |
| Suggesting a flaw | Engage seriously: confirm whether the flaw applies; thank the asker if it does |
| Asking for an opinion | Distinguish opinion from result: state the opinion, then note that it was not measured directly |

### 5.3 Buying Time

When a question requires thought, it is acceptable to pause. "Let me think about that for a moment" is more credible than a fast incorrect answer. Drinking water is a socially-accepted thinking-time signal.

---

## 6. Common Anti-Patterns

| Anti-pattern | Symptom | Fix |
|--------------|---------|-----|
| Results-only deck | No problem statement, no method | Allocate 30% of slides to context, 50% to results, 20% to method |
| Vanity metrics | "99% accuracy" without baseline or class balance | Always show baseline; always show class balance for classification |
| Wall of code | Slide unreadable from row two | Show only the critical lines; link the repo |
| Test-set leakage hidden in narrative | "We chose model X because it generalized best" — chosen on the test set | Distinguish validation tuning from final test evaluation |
| Live training on stage | Long pauses, unpredictable outcomes | Train ahead; show the training curve, not the training |
| Endless appendix slides shown sequentially | Time runs out before main results | Move to backup; use only if asked |
| Reading the slides aloud | Audience disengages | Slides as cues, not script; speak to the room |

---

## 7. Resources

- [Tufte, *The Visual Display of Quantitative Information*](https://www.edwardtufte.com/tufte/books_vdqi) — foundational treatment of chart design and information density.
- [Reynolds, *Presentation Zen*](https://www.presentationzen.com/) — slide design for clarity and impact.
- [Dahl et al., *A Practical Guide to Building Agents*](https://www.anthropic.com/research) — applied write-up patterns useful as a model for technical communication.
- [Wickham, *Tidy Data*](https://vita.had.co.nz/papers/tidy-data.pdf) — clean data presentation underlies clean charts.
- [Statwing — Visualization Selection Guide](https://www.data-to-viz.com/) — chart-type selector for data-to-visualization choices.

---

[← Previous: Academic Writing Style](33_ACADEMIC_WRITING_STYLE_GUIDE.md) | [Index](README.md)
