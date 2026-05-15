# Research Proposal Writing

A research proposal articulates a planned investigation: what will be studied, why it matters, how it will be done, and how success will be evaluated. Its purpose is to obtain agreement — from a supervisor, funding body, ethics committee, or thesis panel — that the proposed work is worth attempting and is feasible to attempt. This guide covers the standard structure of a proposal, the discipline required to define a tractable research question, the patterns reviewers use to evaluate proposals, and the most common reasons proposals are rejected.

**Table of Contents**

1. [The Job of a Proposal](#1-the-job-of-a-proposal)
2. [Standard Structure](#2-standard-structure)
3. [Defining the Research Question](#3-defining-the-research-question)
4. [Scope Discipline](#4-scope-discipline)
5. [Method and Evaluation Plan](#5-method-and-evaluation-plan)
6. [Timeline and Risks](#6-timeline-and-risks)
7. [Ethics and Limitations](#7-ethics-and-limitations)
8. [What Reviewers Look For](#8-what-reviewers-look-for)
9. [Common Rejection Reasons](#9-common-rejection-reasons)
10. [Resources](#10-resources)

---

## 1. The Job of a Proposal

A proposal must persuade a reader of three things:

1. **The problem matters.** Solving it would change practice, theory, or both.
2. **The proposed work is feasible.** It can be completed in the available time, with the available resources, by the proposing author.
3. **The proposed approach is sound.** The method is appropriate to the question; the evaluation will produce interpretable results.

A proposal that establishes only one or two of these will be rejected or returned for revision. The reader is not the author's advocate — they are deliberately searching for reasons the proposed work will not succeed.

### 1.1 Three Reader Profiles

Most proposals are read by three kinds of reviewer:

| Reader | What they evaluate |
|--------|---------------------|
| Subject expert | Novelty, positioning against the literature, technical correctness |
| Methodology expert | Soundness of the proposed method and evaluation |
| Generalist | Clarity, motivation, broader significance |

A proposal that is technically rigorous but unreadable to the generalist will struggle in mixed panels. The introduction in particular must be accessible to all three audiences.

---

## 2. Standard Structure

The exact section names vary by venue and discipline, but most proposals contain the same elements.

| Section | Length (relative) | Purpose |
|---------|-------------------|---------|
| Title | One line | Specific and descriptive; not clever |
| Abstract / Summary | 150–300 words | The whole proposal in one paragraph |
| Background and Motivation | 10–15% | Why the problem matters |
| Literature Review | 15–25% | Where the work sits relative to prior art |
| Research Question(s) | 5% | What will be answered, stated explicitly |
| Proposed Method | 20–30% | How the question will be addressed |
| Evaluation Plan | 10–15% | How success will be measured |
| Timeline | 5–10% | When each piece will happen |
| Risks and Mitigations | 5–10% | What could go wrong; what the response will be |
| Ethics Statement | 5% (when applicable) | Compliance with ethical review requirements |
| References | As needed | Cited works |

A common error is over-allocating to background and under-allocating to method and evaluation. The reader is more interested in what will be done than in what is already known.

---

## 3. Defining the Research Question

The research question is the single most important sentence in the proposal. Every other section serves it.

### 3.1 Properties of a Strong Research Question

| Property | Test |
|----------|------|
| Specific | One reading is possible; alternative readings are obvious misreadings |
| Answerable | A defined method could produce evidence for or against |
| Bounded | Completable in the available time |
| Original | Not already answered conclusively |
| Relevant | The answer would change practice, theory, or both |

### 3.2 Question vs Hypothesis

A **research question** asks: *"How does X affect Y?"*
A **hypothesis** asserts: *"X increases Y by mechanism Z."*

Hypotheses are appropriate when prior evidence justifies a directional claim; questions are appropriate when the relationship is open. ML proposals are usually structured around questions, since the empirical territory is often unmapped.

### 3.3 Operationalisation

A vague question must be operationalised before it can be addressed. Compare:

| Vague | Operational |
|-------|-------------|
| "Does data augmentation help?" | "Does random horizontal flipping improve top-1 accuracy of ResNet-50 on CIFAR-10 by more than 1 percentage point?" |
| "Are transformers interpretable?" | "Do the attention maps of a fine-tuned BERT model on sentiment classification align with human-annotated salient tokens, measured by IoU?" |
| "Can ML predict customer churn?" | "Can a gradient-boosted classifier achieve AUC > 0.80 on a defined customer-churn dataset using only behavioural features available 30 days before churn?" |

The operational form specifies the measurement, the threshold, the model, the data, and the conditions. Without operationalisation, the proposal cannot define what success looks like.

---

## 4. Scope Discipline

The most common cause of failed research is over-scoped questions, not under-skilled researchers.

### 4.1 The "Single Sentence" Test

If the research question cannot be stated in a single sentence without conjunctions (no "and", no "as well as"), it is two questions. Either pick one or restructure them as a primary question with sub-questions clearly subordinated.

### 4.2 The "Minimum Viable Thesis"

Identify the smallest contribution that would still constitute completion. This is the floor; everything above it is stretch. Common patterns:

- **Reproduce + extend**: reproduce a published baseline, then test one variation. The reproduction alone is a defensible result if the extension fails.
- **Negative-result tolerance**: design the work so that a negative answer is publishable. "Method X does not work in setting Y" is valuable when prior work claimed it would.
- **Single-axis comparison**: vary one factor; hold all else constant. Multi-axis sweeps balloon in time and produce ambiguous attributions.

### 4.3 Cutting Non-Essentials

A proposal section is non-essential if removing it does not weaken the argument that the work is significant, feasible, and sound. Common cuts:

- Speculative future applications beyond the scope.
- Comprehensive history of the field when only the recent positioning matters.
- Tangential techniques that will not be used.
- Multiple alternative methods when only one will be implemented.

Reviewers reward focus. A 20-page proposal that says one thing well outperforms a 30-page proposal that says four things vaguely.

---

## 5. Method and Evaluation Plan

### 5.1 Method Must Be Operational

A proposal method section says what will be done, in what order, with what data, on what hardware, using what tools. Compare:

| Aspirational (weak) | Operational (strong) |
|---------------------|----------------------|
| "We will use deep learning" | "A ResNet-50 will be fine-tuned on the [dataset] training split, with cross-entropy loss, Adam optimizer, learning rate 1e-4, batch size 32, for 50 epochs" |
| "Models will be compared" | "Three models — A, B, C — will be compared on [metric] using 5-fold cross-validation; differences will be tested with a paired t-test at α = 0.05" |
| "Data will be cleaned" | "Records with missing target values will be dropped; numeric features will be standardized; categorical features with cardinality ≤ 10 will be one-hot encoded, others target-encoded" |

The reviewer should be able to predict, from the method section, what the experiments will look like.

### 5.2 Evaluation Defined Up Front

The evaluation plan must be defined before any experiment is run. It states:

- **Metric** — which quantitative measure will be used.
- **Baseline** — what alternative the new method must outperform.
- **Threshold** — what magnitude of improvement counts as a positive result.
- **Statistical treatment** — how variance and significance will be assessed.
- **Negative-result interpretation** — what is concluded if the threshold is not met.

A proposal that omits the negative-result plan invites the suspicion that the author will rationalize whatever happens as a success.

### 5.3 Pilot Study

For ambitious methods, a pilot study — a small-scale rehearsal — strengthens the proposal. Reporting that a simplified version of the method has already produced encouraging results de-risks the full proposal.

---

## 6. Timeline and Risks

### 6.1 Timeline

A timeline allocates calendar time to each stage of the work. Common formats:

- **Gantt chart** — visual; shows overlap and sequencing.
- **Phase table** — text; lists stages with start and end dates.
- **Milestone list** — checkpoint events with target dates.

Two patterns produce realistic timelines:

1. **Buffer**: allocate 20–30% of total time as unallocated slack. Unanticipated obstacles are the rule, not the exception.
2. **Critical path identification**: identify which sequential dependencies determine total duration. Optimization elsewhere does not shorten the project.

### 6.2 Risk Register

A risk register lists what could go wrong, how likely it is, how serious it would be, and what the response is.

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Dataset access denied | Medium | High | Identify two backup datasets in advance |
| Method fails to converge | Medium | Medium | Reduce model size; switch to validated baseline |
| Hardware insufficient | Low | High | Cloud credits or institutional cluster as fallback |
| Key collaborator unavailable | Low | Medium | Document dependencies; identify substitute |
| Scope creep | High | High | Reaffirm research question monthly; track time spent on out-of-scope work |

A proposal without a risk section reads as either naive or evasive. A proposal with a single line of risk acknowledgement is no better.

---

## 7. Ethics and Limitations

### 7.1 Ethics Statement

ML work touching on human data, sensitive attributes, or deployment contexts requires ethical review. The proposal should state:

- Whether the work involves human subjects or human-derived data.
- Whether ethics committee approval is required, sought, or obtained.
- How data will be handled (storage, retention, anonymisation).
- Known sources of bias in data sources.
- Foreseeable harms from the work and intended mitigations.

For projects that do not involve human data, a one-sentence statement of non-applicability is appropriate; the topic should not be skipped silently.

### 7.2 Acknowledged Limitations

Strong proposals state their limitations openly:

- Generalisation boundaries (the work covers domain X; not Y).
- Methodological choices not exhaustively justified (one model family chosen; others not explored).
- Resource constraints accepted (compute budget caps the model size).

Acknowledged limitations strengthen credibility. Hidden limitations weaken it: reviewers find them anyway.

---

## 8. What Reviewers Look For

A proposal is typically scored on a small number of dimensions, common across most rubrics:

| Dimension | Question reviewers ask |
|-----------|-------------------------|
| Significance | Does this matter? Would the result change anything? |
| Originality | Is this new, or restating prior work? |
| Soundness | Does the method actually address the question? |
| Feasibility | Can this be done in the time available, by this author? |
| Clarity | Is the proposal readable on a first pass? |
| Evaluation | Can the result be assessed objectively? |

A proposal scoring strongly on five and weakly on one is usually accepted with revision; scoring weakly on two or more is usually rejected.

### 8.1 The "So What?" Test

After reading the proposal, the reviewer should be able to answer: *"If this work succeeds, what will change?"*

If the answer is unclear, the significance argument is weak. The change might be modest — "this would be the first dataset for X" or "this would establish a baseline for Y" — but it must be articulable.

---

## 9. Common Rejection Reasons

| Reason | Typical phrasing in reviews |
|--------|------------------------------|
| Unclear research question | "It is not clear what the central question is" |
| Out-of-scope ambition | "The proposed work would require [much longer] to complete" |
| Method not specified | "The method section describes a goal, not a procedure" |
| No evaluation plan | "It is unclear how the success of the work would be assessed" |
| Missing related work | "The proposal does not engage with [prior work X]" |
| Weak motivation | "The significance of the work is not established" |
| Naive risk assessment | "The proposal does not anticipate likely difficulties" |
| Methodology inappropriate to question | "The proposed method cannot answer the stated question" |

Most rejected proposals fail for two or more of these reasons simultaneously. A revision addressing only one rarely succeeds; the rejection signals broader issues.

---

## 10. Resources

- [Locke, Spirduso, and Silverman, *Proposals That Work* (2014)](https://us.sagepub.com/en-us/nam/proposals-that-work/book242053) — comprehensive treatment of proposal structure across disciplines.
- [Kraicer, *The Art of Grantsmanship*](https://www.hfsp.org/sites/default/files/Sciences/forms/ArtofGrants.pdf) — applied to research grants but broadly applicable.
- [How to Write a Research Proposal — University of Birmingham](https://intranet.birmingham.ac.uk/as/libraryservices/library/skills/asc/documents/public/Short-Guide-Research-Proposals.pdf) — short, practical reference.
- [Booth, Colomb, Williams, *The Craft of Research* (2016)](https://press.uchicago.edu/ucp/books/book/chicago/C/bo24216433.html) — foundational treatment of research questions and argument structure.
- [Cohen, *Writing the Successful Thesis and Dissertation Proposal*](https://www.routledge.com/) — graduate-level proposal preparation.
- [The PhD Proposal: A Short Guide — Tara Brabazon (lecture series)](https://www.youtube.com/results?search_query=tara+brabazon+phd+proposal) — practical advice on common failure modes.

---

[← Previous: Literature Review](30_LITERATURE_REVIEW_GUIDE.md) | [Index](README.md) | [Next: End-to-End ML Project Workflow →](32_PROJECT_WORKFLOW_GUIDE.md)
