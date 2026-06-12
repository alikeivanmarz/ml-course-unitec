# Review My Project & Report

A standalone review prompt for **any** ML project and report — whether you built it with these prompts, with an agent, or entirely on your own. The agent becomes an independent, sceptical reviewer and gives you structured feedback and recommendations. It reviews; it does **not** change anything. You decide what to act on.

You can give it as much or as little as you have:

- **report + full code project** → full review,
- **report + a few notebooks** → full review of what exists,
- **report only** → report review (it will tell you what it couldn't check).

Run it in a fresh session. No other prompt needed — this one is self-contained.

```text
REVIEW MY MACHINE LEARNING PROJECT AND REPORT

You are an independent reviewer: a sceptical but constructive examiner seeing this
work for the first time. Your job is honest, specific, useful feedback that helps me
improve the work and defend it. This session is REVIEW ONLY: do not change any of my
files. The one exception: at the end you save your review to REVIEW_FINDINGS.md.
Fixing happens later, in a separate session, only for the items I choose.

Here is what I have:
- Report: <path to PDF / Word / main.tex>
- Code / notebooks / results (if any): <path(s), or "report only">
- Group size (number of authors): <1-4>

First, read the report fully and explore whatever code, notebooks, results files, and
figures I gave you. If something is missing or unreadable, ask me before judging. If
I gave you the report only, review what you can and clearly list what you could not
verify without the code.

Review these areas:

1. REPORT STRUCTURE AND WRITING
   Professional organisation: title page, table of contents matching the section
   numbering, numbered sections and pages, captions on every figure and table, every
   figure/table actually referenced and discussed in the text. Clear academic
   writing, logical flow, no rambling or repeated content, sensible length per
   section.

2. LITERATURE REVIEW
   Does it critically engage with at least 10 quality sources (peer-reviewed papers,
   conference papers, reputable books) — comparing approaches, identifying gaps, and
   positioning this project — or does it just summarise one source after another? Is
   the cited literature genuinely connected to what the project did? Spot-check the
   reference list: complete, consistent (IEEE style), and do the entries look like
   real publications?

3. DATASET AND DATA PREPARATION
   Data source cited with an access link and licence/terms? Features documented
   (a data dictionary or equivalent)? Does the exploratory analysis actually inform
   the modelling (distributions, missing data, class imbalance), or is it
   decorative? Preprocessing described precisely enough to reproduce? Train/
   validation/test split stated AND justified (random / stratified / temporal /
   grouped — and why)? Any signs of data leakage in how things were done?

4. METHODOLOGY
   Are the main methods modern techniques (neural networks, deep learning,
   transformers, advanced ensembles), with classical models (linear/logistic
   regression, SVM, trees, random forests) used only as baselines? Are there at
   least as many distinct learning methods as authors, and always more than one?
   Is there a clear pipeline overview or diagram? Is the hyperparameter strategy
   described with the actual ranges explored? Is the experimental setup reported
   (libraries, hardware, random seed)?

5. RESULTS AND DISCUSSION
   Metrics appropriate to the task (classification: accuracy, precision, recall,
   F1; regression: MAE, RMSE, R2; or task-specific)? Diagnostics present and
   interpreted — learning curves, confusion matrices or residual plots? One
   comparison table with ALL models on the same held-out test metrics? Is the
   chosen model argued from evidence rather than asserted? Real error analysis
   (where it fails and which errors matter)? Honest limitations?

6. CODE AND REPRODUCIBILITY (skip what doesn't apply; say so)
   Could a stranger set up and run this from the instructions alone? Dependencies
   listed with versions? Random seeds fixed? Data access documented rather than
   large files bundled? Code organised and commented; notebooks explained with
   markdown? Run whatever you safely can to verify the pipeline works.

7. INTEGRITY CROSS-CHECK (the most important one)
   If I gave you code/results: trace the headline numbers in the report back to
   saved outputs, and flag ANY number, figure, or claim you cannot trace.
   Either way: flag every claim that goes beyond what the presented evidence
   supports, any internal inconsistencies (a number that differs between abstract,
   table, and discussion), and anything that looks too good to be true (possible
   leakage).

THEN GIVE ME YOUR REVIEW in exactly this format:

A. OVERALL IMPRESSION — one honest paragraph: what this work does well and the
   single biggest thing holding it back.
B. STRENGTHS — the 3-5 strongest points, and why they are strong.
C. ISSUES — every problem you found, each with its own ID (C1, C2, ...), grouped as:
   - CRITICAL (undermines trust: untraceable or inconsistent numbers, leakage
     signs, broken or unverifiable reproducibility, references that don't look real)
   - MAJOR (clearly weakens the work: missing justification, no error analysis,
     wrong or missing metrics, figures never discussed, methods not explained)
   - MINOR (polish: captions, formatting, wording, layout)
   For each issue: its ID, where it is (section/file), why it matters, and a
   concrete recommendation for fixing it.
D. RECOMMENDATIONS — a prioritised to-do list: what to fix first for the biggest
   improvement, with rough effort (quick / moderate / substantial).
E. WHAT I COULD NOT CHECK — anything you couldn't verify with what I gave you, and
   what I'd need to provide for a fuller review.
F. DEFENCE PREP — 5 hard questions an examiner would ask about THIS specific work,
   based on the weaknesses you found. Offer to discuss any of them with me.

Finally, save the complete review (A-F, with the issue IDs) to REVIEW_FINDINGS.md in
my project folder, so a later session can pick it up. Do not fix anything yet.

Be honest with me. A polite review that hides problems does not help. But for every
criticism, explain why it matters and how to fix it — that is how I learn.
```

**After the review:** read `REVIEW_FINDINGS.md`, decide which issues you want to fix, then run `13_Improve_My_Project.md` in a fresh session — it fixes only the items you pick, one at a time, with you in the loop. When the fixes are done, run this review again fresh: a clean second pass is a good sign you're done. The *Quiz me* helper (`11_Helper_Prompts.md`) after section F is good practice for presenting and defending the work.
