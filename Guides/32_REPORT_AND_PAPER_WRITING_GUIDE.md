# Technical Report and Paper Writing

A technical report or research paper communicates work that has been performed and conclusions that follow from it. Its job is to convey, to a reader who was not present, what was done, what was found, and what to make of it. This guide covers the standard section structure used in machine learning and computer-science publications, the conventions for figures, tables, and citations, and the editorial decisions that distinguish clear papers from cluttered ones.

**Table of Contents**

1. [Standard Structure](#1-standard-structure)
2. [Abstract Anatomy](#2-abstract-anatomy)
3. [Introduction](#3-introduction)
4. [Related Work](#4-related-work)
5. [Method](#5-method)
6. [Experiments and Results](#6-experiments-and-results)
7. [Figures and Tables](#7-figures-and-tables)
8. [Discussion and Conclusion](#8-discussion-and-conclusion)
9. [Citation Mechanics](#9-citation-mechanics)
10. [Resources](#10-resources)

---

## 1. Standard Structure

Most empirical papers follow one of two structures.

### 1.1 IMRaD

IMRaD — Introduction, Methods, Results, and Discussion — is the standard scientific paper structure used across most disciplines.

| Section | Approximate length |
|---------|--------------------|
| Introduction | 10–15% |
| Methods | 20–30% |
| Results | 20–30% |
| Discussion | 15–25% |

### 1.2 ML Variant

Machine learning papers expand IMRaD into a six- or seven-section structure:

| Section | Purpose |
|---------|---------|
| Introduction | Frame the problem; list contributions |
| Related Work | Position against prior literature |
| Method / Approach | Describe the proposed technique |
| Experiments | Setup, baselines, datasets, metrics |
| Results | Headline outcomes, ablations |
| Discussion / Limitations | Interpretation; honest scope |
| Conclusion | Brief summary; future directions |

Related Work is its own section in ML papers because the field is dense and reviewers expect explicit positioning. Some venues separate Method from Implementation Details, or merge Experiments and Results.

### 1.3 Section Order in Practice

Although the conventional order places Related Work after Introduction, some authors place it before Conclusion to keep the early sections tightly focused on the contribution. Either is acceptable; consistency with venue conventions matters more than the choice itself.

---

## 2. Abstract Anatomy

The abstract is the most-read part of a paper. Most readers will read it; few will read the rest. It is also disproportionately weighted in reviewer first impressions.

### 2.1 The Four-Sentence Pattern

A reliable abstract structure:

| Sentence | Content |
|----------|---------|
| 1 — Context | What problem the paper addresses, in one sentence |
| 2 — Gap | What prior work has not addressed |
| 3 — Approach | What the paper does, in one sentence |
| 4 — Result | The headline finding, with a number when possible |

A fifth sentence on broader implication is sometimes added; longer abstracts (NeurIPS, ACL) can extend each element to two sentences without changing the structure.

### 2.2 What to Leave Out

- **Background**: the abstract is not the place to explain machine learning broadly.
- **Hedging**: avoid "we believe", "we hope"; the abstract reports what was done.
- **Forward references**: "as discussed in Section 3" is invalid in an abstract.
- **Vague claims**: replace "achieves competitive performance" with the actual number.

### 2.3 The One-Line Takeaway

Before writing the abstract, the author should write a single sentence: *"This paper shows that [X]."* Every sentence in the abstract should support that single sentence. If the takeaway cannot be stated in one sentence, the contribution is not yet clear enough to write the abstract.

---

## 3. Introduction

The introduction expands the abstract into 0.5–1.5 pages.

### 3.1 The Funnel Structure

Most introductions move from broad to specific in five paragraphs:

1. **Domain** — the broad area and why it matters.
2. **Problem** — the specific subproblem addressed.
3. **State of prior work** — what has been tried; what gap remains.
4. **This paper's approach** — what the work proposes.
5. **Contributions list** — bulleted, 3–5 items.

The funnel is reliable because readers from outside the subfield need the broader framing; readers inside it can skim quickly to the contributions.

### 3.2 The Contributions List

The contributions list at the end of the introduction is the most-cited part of the paper outside the abstract. Each contribution should be:

- **Concrete** — describes a specific output, not a goal.
- **Verifiable** — the reader can locate it in the paper.
- **Distinct** — not a paraphrase of another contribution.

Compare:

| Weak | Strong |
|------|--------|
| "We propose a novel method" | "We propose X, the first method to handle [setting] without [assumption]" |
| "We achieve good performance" | "We achieve 87.4% accuracy on [benchmark], a 3.2-point improvement over the prior state of the art" |
| "We demonstrate the effectiveness" | "We show, through ablations on three datasets, that component X accounts for 80% of the gain" |

Three to five contributions is the typical range; ten is too many to retain.

---

## 4. Related Work

A related work section positions the contribution against prior literature. It is structured argument, not summary.

### 4.1 Group by Theme, Not by Paper

Strong related-work sections organize by approach:

> "Two families of methods address [problem]. The first ... is exemplified by [A, B, C]. The second ... is represented by [D, E]. Both face the limitation that [gap], which the present work addresses by [contribution]."

Weak related-work sections walk through papers chronologically without grouping.

### 4.2 Comparison Targets

Cite generously, but note explicitly which prior works will appear in the experimental comparison and which will not. A reviewer who notices an unmentioned baseline being compared in the experiments section will assume something is being hidden.

### 4.3 Length

The expected length is 0.5–1.5 pages. A page-and-a-half related-work section is appropriate for a paper introducing a new method; a half-page section is appropriate for a paper extending a clearly-positioned prior approach. Excessively long related-work sections suggest a reviewer wrote them for the author.

---

## 5. Method

The method section describes the technical contribution in full enough detail that an informed reader could re-implement it.

### 5.1 Reproducibility-First Writing

A method section is reproducible if it specifies:

- **Inputs and outputs** of every component.
- **Architecture** — for models: layer types, dimensions, connections.
- **Initialisation** — for parameters: distribution, scale.
- **Training procedure** — loss, optimizer, learning rate, schedule.
- **Hyperparameters** — values and how they were chosen.
- **Preprocessing** — every transformation applied to the data.

If any of these are missing, the work cannot be reproduced from the paper alone — only from the paper plus the code, if released.

### 5.2 Notation Table

Papers using extensive notation should provide a notation table near the start of the method section. The table maps each symbol to its meaning and its dimension. Forward references to symbols defined later confuse readers; the table is a one-stop lookup.

### 5.3 Pseudocode vs Prose

| Form | Best for |
|------|----------|
| Prose | Conceptual descriptions; high-level intuition |
| Pseudocode | Algorithms with control flow (loops, conditionals, recursion) |
| Equations | Single-step transformations |
| Diagrams | Architectural overviews showing component composition |

Most method sections use a combination. Pseudocode replaces prose when prose would obscure the algorithmic structure; equations replace prose when symbols are clearer than English.

---

## 6. Experiments and Results

### 6.1 Experimental Setup

The setup section specifies, before any results are reported:

- **Datasets** — sources, sizes, splits, preprocessing applied.
- **Baselines** — what the new method is compared against, with citations.
- **Metrics** — what is measured; why this metric was chosen.
- **Evaluation protocol** — train/val/test splits, cross-validation, statistical tests.
- **Implementation details** — hardware, software versions, hyperparameter ranges searched.

A reader should be able to predict, from the setup section, what the results tables will look like.

### 6.2 Results Presentation

| Element | Best practice |
|---------|---------------|
| Headline result | First table; main metric highlighted |
| Comparison structure | One row per method, one column per dataset/condition |
| Best result | Bold |
| Second-best | Underlined |
| Variance | Standard deviation in parentheses, or `±` form |
| Statistical significance | Asterisks with footnote describing the test |

A results table that does not visually distinguish the best result requires the reader to do the comparison; this delays comprehension.

### 6.3 Ablation Studies

Ablation studies isolate the contribution of each design choice by removing or modifying components individually. A method paper without ablations is incomplete: the reader cannot tell which choices matter.

| Ablation pattern | Question answered |
|------------------|-------------------|
| Component removal | Does removing X harm performance? |
| Component substitution | Does replacing X with a simpler alternative harm performance? |
| Hyperparameter sensitivity | How much does performance vary with X across a sensible range? |
| Scale sweep | How does the contribution change with data or model size? |

Ablation tables follow the same conventions as main-result tables: bold the best, report variance, mark significance.

### 6.4 Reporting Variance

A single-number result hides uncertainty. Reports should include:

- Multiple training seeds (deep learning) — report mean ± standard deviation.
- Cross-validation folds (traditional ML) — report fold-wise scores or mean ± std.
- Bootstrap intervals on test-set metrics for finite-sample uncertainty.

A 0.5-point improvement is meaningless if the standard deviation across seeds is 1.0.

---

## 7. Figures and Tables

### 7.1 Caption Conventions

| Element | Caption position |
|---------|------------------|
| Table | Above the table |
| Figure | Below the figure |
| Algorithm (pseudocode) | Above the algorithm box |

Captions should be self-contained: a reader who looks only at the figure or table should understand what is shown without consulting the body text. The first sentence states what the artefact shows; subsequent sentences explain notation, axes, or grouping.

### 7.2 Figure Standards

| Property | Recommendation |
|----------|----------------|
| Format | Vector (PDF, SVG) for line plots and diagrams; PNG only for screenshots and rendered images |
| Font size | At least 8pt at final printed size |
| Colour palette | Colour-blind safe (ColorBrewer, viridis); test in greyscale |
| Axis labels | Variable name + units, on every axis |
| Legend | In-plot when space permits; avoid external legends if possible |
| Aspect ratio | Match the data; avoid forcing a square plot when one axis dominates |

Plots produced for a notebook are usually not publication-ready: fonts are too small, line widths too thin, and colour palettes default to non-colour-blind-safe choices.

### 7.3 Table Standards

| Property | Recommendation |
|----------|----------------|
| Vertical lines | Avoid; use horizontal rules only |
| Significant digits | Match measurement precision; do not over-report |
| Number alignment | Decimal-aligned within columns |
| Column headers | Brief; expand abbreviations in the caption |
| Method ordering | Logical (chronological, complexity, performance); not alphabetical |

The `booktabs` LaTeX package is the standard for high-quality tables; it enforces the no-vertical-lines convention and provides clean horizontal rules.

---

## 8. Discussion and Conclusion

### 8.1 Discussion

The discussion interprets results, relates them to prior work, and acknowledges limits. Common discussion subsections:

- **Interpretation** — what the results mean, beyond the numbers.
- **Comparison** — how the results stand against prior expectations or claims.
- **Limitations** — what the work does not establish; where it would not generalize.
- **Negative results** — findings that contradict initial expectations.

A discussion that hedges everything is unhelpful; a discussion that hedges nothing is suspicious. The right register acknowledges what the evidence supports and stops there.

### 8.2 Limitations

Many venues now require a dedicated limitations subsection. Strong limitations sections name specific scope boundaries:

- "The evaluation is restricted to English-language data; performance on other languages is unknown."
- "The largest model tested has 1B parameters; behaviour at larger scale is unverified."
- "Training was conducted on a single hardware configuration; results may not transfer to inference-constrained settings."

Hidden limitations are found by reviewers and reduce the paper's standing more than acknowledged limitations do.

### 8.3 Conclusion

The conclusion is brief — typically one paragraph for a conference paper, longer for a thesis. It contains:

1. A one- or two-sentence summary of the contribution.
2. The most important finding, restated.
3. A short, concrete future-work statement.

Avoid restating the paper at length; the abstract already does this. Avoid speculative future-work passages that read as a wish list.

---

## 9. Citation Mechanics

### 9.1 BibTeX Basics

A BibTeX entry has a type, a key, and fields:

```bibtex
@inproceedings{vaswani2017attention,
  title     = {Attention is All You Need},
  author    = {Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and others},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2017},
  volume    = {30},
}
```

Common entry types: `@article`, `@inproceedings`, `@book`, `@incollection`, `@techreport`, `@misc`, `@phdthesis`.

### 9.2 Citation Styles

| Style | In-text form | Common in |
|-------|--------------|-----------|
| Numeric | "[12]" | IEEE, ACM (some) |
| Author-year | "(Smith, 2020)" | APA, Harvard, ML conferences (NeurIPS, ICML) |
| Footnote | Superscript number | Humanities; some sciences |

ML venues vary: NeurIPS and ICML use author-year (\\citep, \\citet) by default; IEEE-affiliated venues use numeric. The venue's style file determines the format; the author chooses the cite-key strategy.

### 9.3 Cite-Key Conventions

A consistent cite-key convention prevents collisions and aids editing. Common patterns:

| Pattern | Example |
|---------|---------|
| `firstauthor + year + firstword` | `vaswani2017attention` |
| `firstauthor + year` | `vaswani2017` (collision-prone for prolific authors) |
| `lastnames + year` | `vaswani-shazeer-2017` |

The first form is widely adopted and minimizes collision risk.

### 9.4 Tools

| Tool | Role |
|------|------|
| Zotero, Mendeley, JabRef | Reference management; auto-import metadata; export BibTeX |
| Google Scholar "Cite" | Quick BibTeX export per paper (verify accuracy) |
| `biber` / `bibtex` | Bibliography processors invoked by LaTeX |
| `biblatex` package | Modern LaTeX bibliography handling; supports more styles than legacy `bibtex` |

Auto-imported BibTeX entries should be checked: title capitalisation, missing pages or volumes, incorrect venue names, and author lists truncated to "and others" are common errors that manifest in the rendered bibliography.

---

## 10. Resources

- [Mensh and Kording, *Ten Simple Rules for Structuring Papers* (2017)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005619) — concise treatment of paper structure across the sciences.
- [Strunk and White, *The Elements of Style*](https://www.gutenberg.org/ebooks/37134) — public-domain edition; foundational style reference.
- [Sword, *Stylish Academic Writing* (2012)](https://www.hup.harvard.edu/books/9780674064485) — empirical study of clear academic prose.
- [LaTeX `booktabs` documentation](https://ctan.org/pkg/booktabs) — table formatting standards.
- [The NeurIPS Paper Checklist](https://neurips.cc/public/guides/PaperChecklist) — reproducibility, ethics, and limitations expectations.
- [The ACM Author Guide](https://www.acm.org/publications/authors/information-for-authors) — formatting and citation conventions for ACM venues.
- [Wickham, *Tidy Data* (2014)](https://vita.had.co.nz/papers/tidy-data.pdf) — data presentation as a writing concern.
- [Tufte, *The Visual Display of Quantitative Information*](https://www.edwardtufte.com/tufte/books_vdqi) — figure design principles.
