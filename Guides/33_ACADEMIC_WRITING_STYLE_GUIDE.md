# Academic Writing Style

Academic writing has its own conventions — for tense, voice, hedging, and word choice — that differ from journalistic, technical, or conversational registers. The conventions are not arbitrary: they reflect the discipline's preference for precision, traceable claims, and the separation of evidence from interpretation. This guide collects the conventions that recur in scientific and machine-learning writing, the patterns that produce clear prose, and the editorial pitfalls that obscure meaning.

**Table of Contents**

1. [Tense and Voice](#1-tense-and-voice)
2. [Hedging and Claiming](#2-hedging-and-claiming)
3. [Signposting and Topic Sentences](#3-signposting-and-topic-sentences)
4. [Sentence-Level Style](#4-sentence-level-style)
5. [Word Choice](#5-word-choice)
6. [Common Pitfalls for Non-Native Writers](#6-common-pitfalls-for-non-native-writers)
7. [Editing Tools, Used Responsibly](#7-editing-tools-used-responsibly)
8. [Resources](#8-resources)

---

## 1. Tense and Voice

### 1.1 Tense Conventions

| Tense | Use | Example |
|-------|-----|---------|
| Past | Completed work — what was done, what was found | "A ResNet-50 was trained on the dataset for 90 epochs." |
| Present | Established facts; descriptions of figures, tables, and the paper itself | "The dataset contains 50,000 examples." / "Figure 2 shows the loss curve." |
| Future | Genuinely future work — typically only in conclusions | "Future work will examine larger model sizes." |
| Present perfect | Prior work that informs the current study | "Several methods have been proposed for this task." |

Mixing tenses within a single paragraph is common and acceptable when each follows the convention above. Reviewers notice and object when methods or results are described in the present tense ("The model achieves X") instead of past tense ("The model achieved X").

### 1.2 Active vs Passive Voice

Modern style guides recommend active voice for clarity. Passive voice remains standard in some contexts — particularly the methods section, where the actor is unimportant.

| Section | Convention |
|---------|------------|
| Abstract | Active or passive; brevity wins |
| Introduction | Active voice preferred |
| Methods | Passive often natural ("The model was trained ...") |
| Results | Active voice for the model as agent ("The classifier achieved ...") |
| Discussion | Active voice; the author interprets |

### 1.3 First-Person Use

The strict prohibition on first-person ("we", "I") in academic writing has weakened in most modern venues. Many ML conferences now permit and even prefer "we" in introductions and discussions. Where first person is permitted, the convention is:

- **"We"** for plural authorship; acceptable even with one author by long convention.
- **"I"** is used in single-author humanities work; rare in ML.
- Avoid first person in methods sections, where the actor is the procedure, not the author.

---

## 2. Hedging and Claiming

The strength of a claim should match the strength of the evidence. Both over-claiming and over-hedging are weaknesses.

### 2.1 Strength Calibration

| Vocabulary | Strength | Use when |
|------------|----------|----------|
| "demonstrates", "establishes", "proves" | Strong | Direct evidence; mathematical proof |
| "shows", "indicates" | Strong | Clear empirical result |
| "suggests", "implies" | Moderate | Indirect evidence; consistent with multiple explanations |
| "may indicate", "is consistent with" | Weak | Suggestive but not conclusive |
| "appears to", "seems to" | Tentative | Initial observation; needs further investigation |

A common error is to claim "we prove" when the evidence is empirical. Proof has a specific meaning in mathematics; empirical evidence "shows" or "demonstrates" but does not "prove".

### 2.2 Quantified Claims

Where possible, replace qualitative claims with quantitative ones:

| Vague | Quantified |
|-------|------------|
| "Performance improves significantly" | "Accuracy improves by 3.2 points (p < 0.01)" |
| "The model is much faster" | "The model runs 4.7× faster on the same hardware" |
| "Many examples are misclassified" | "12% of examples are misclassified" |

Quantified claims are easier to verify and harder to misread.

### 2.3 Avoiding Over-Hedging

Hedging is a tool, not a default. Excessive hedging — "this might possibly suggest that perhaps the model could be slightly better" — undermines confidence in any claim. Reserve hedges for cases where the evidence genuinely permits multiple interpretations.

---

## 3. Signposting and Topic Sentences

### 3.1 Topic Sentences

Each paragraph opens with a sentence that states its main point. The remaining sentences support, qualify, or extend that point. A reader skimming only topic sentences should grasp the argument.

Compare:

> "Recent work has explored several directions. Smith et al. (2020) used method A. Jones et al. (2021) extended this with method B. ..."

Topic sentence: weak. The first sentence promises an organised treatment; the rest reads as a list.

> "Two families of methods have emerged for this problem: those based on assumption A and those based on assumption B. The first family, exemplified by Smith et al. (2020), ..."

Topic sentence: strong. The first sentence states the structure of the paragraph; the rest fills it in.

### 3.2 Signposts

Signposts orient the reader within the argument:

| Signpost | Function |
|----------|----------|
| "First", "second", "third" | Enumeration |
| "However", "in contrast" | Contrast |
| "In addition", "furthermore" | Continuation |
| "Therefore", "consequently" | Consequence |
| "For example", "specifically" | Illustration |
| "In summary" | Synthesis |

Used sparingly, signposts clarify; used densely, they clutter. One signpost per paragraph is typical; two or three across a dense paragraph is acceptable.

### 3.3 Section Roadmaps

Long sections benefit from a one-sentence roadmap at the start:

> "This section first describes the dataset (3.1), then the model architecture (3.2), and finally the training procedure (3.3)."

Roadmaps are particularly useful when section ordering is not obvious from the headings alone.

---

## 4. Sentence-Level Style

### 4.1 Subject Early, Verb Close

Readers track sentences by identifying the subject and verb. Sentences that delay the verb across long subordinate clauses become difficult to follow.

| Less clear | Clearer |
|------------|---------|
| "The model, which was trained on a dataset of 50,000 examples spanning three years of data and including both real and synthetic images, achieved 87% accuracy." | "The model achieved 87% accuracy. It was trained on 50,000 examples spanning three years, including both real and synthetic images." |

Long subordinate clauses are not wrong, but they multiply cognitive load. Two short sentences are usually clearer than one long one.

### 4.2 Avoid Nominalisation

Nominalisation converts verbs into nouns ("decide" → "decision", "analyse" → "analysis"). Excessive nominalisation produces dense, lifeless prose:

| Nominalised | Direct |
|-------------|--------|
| "The performance of the model showed an improvement of 3 points." | "The model improved by 3 points." |
| "The selection of features was made on the basis of correlation." | "Features were selected by correlation." |
| "The conclusion was reached that the method is effective." | "The method is effective." |

Verbs carry the action; nominalisation buries it.

### 4.3 Vary Sentence Length

A sequence of similar-length sentences becomes monotonous. Strong technical writing alternates short, punchy sentences with longer, qualified ones. The short sentences carry the headline points; the long ones provide the context.

### 4.4 Cut Redundant Qualifiers

Common redundancies in academic prose:

| Redundant | Direct |
|-----------|--------|
| "very important" | "important" |
| "completely eliminated" | "eliminated" |
| "absolutely necessary" | "necessary" |
| "a number of" | "several" or a number |
| "in order to" | "to" |
| "due to the fact that" | "because" |
| "at this point in time" | "now" |

Removing such qualifiers makes the prose tighter without losing meaning.

---

## 5. Word Choice

### 5.1 Precision

Choose precise terms over general ones:

| General | Precise |
|---------|---------|
| "approach" | "method", "algorithm", "framework", "technique" — choose the specific term |
| "data" | "dataset", "training set", "examples", "observations" |
| "result" | "accuracy", "F1 score", "loss", "output" |
| "improve" | "increase", "reduce", "stabilize" — depending on direction |

Precision is the difference between describing what was done and gesturing at it.

### 5.2 Define Jargon on First Use

Technical terms should be defined the first time they appear, with the abbreviation in parentheses:

> "Convolutional Neural Networks (CNNs) are a class of models that ..."

Subsequent uses can use the abbreviation alone. Abbreviations introduced and never reused should be removed; abbreviations used once or twice are usually clearer spelled out.

### 5.3 Latin Abbreviations

| Abbreviation | Meaning | Use |
|--------------|---------|-----|
| `e.g.` | "for example" | Followed by examples; comma after |
| `i.e.` | "that is" | Followed by clarification; comma after |
| `et al.` | "and others" | After first author in citations with > 2 authors |
| `cf.` | "compare" | Directs the reader to compare with a cited source |
| `etc.` | "and so on" | Avoid when the list is short or finite |

`e.g.` and `i.e.` are not interchangeable: "e.g." introduces examples; "i.e." restates. Mixing them is a common error.

### 5.4 Inclusive Language

Modern style guides recommend gender-neutral phrasing throughout:

| Avoid | Prefer |
|-------|--------|
| "he"/"she" as universal | "they" (singular), or rephrase to avoid pronouns |
| "manpower" | "workforce", "staffing" |
| "blacklist" / "whitelist" | "blocklist" / "allowlist" |
| "master" / "slave" (in technical contexts) | "primary" / "replica", or domain-appropriate alternatives |

Inclusive terminology has been adopted across most major venues and style guides.

---

## 6. Common Pitfalls for Non-Native Writers

### 6.1 Articles (a, an, the)

Article use is a common difficulty. Brief rules:

- **`a` / `an`**: introduces a noun for the first time, or refers to one of many.
- **`the`**: refers to a specific, identifiable noun; the reader should know which one.
- **No article**: plural or uncountable nouns when speaking generally ("models are trained ...").

Examples:

| Wrong | Correct |
|-------|---------|
| "We trained model on dataset." | "We trained a model on the dataset." |
| "The classification is task in which ..." | "Classification is a task in which ..." |
| "Model achieved best result." | "The model achieved the best result." |

### 6.2 Common Confusions

| Wrong | Right | Notes |
|-------|-------|-------|
| "less" with countable nouns | "fewer" | "fewer examples", not "less examples" |
| "data is" | Both acceptable | "data" is increasingly singular in technical writing; consistency matters |
| "comprise of" | "comprise" or "consist of" | "Comprise" is transitive: "the dataset comprises X" |
| "different than" | "different from" | "Than" is non-standard in formal writing |
| "affect" / "effect" | Verb / noun | "X affects Y"; "X has an effect on Y" |
| "its" / "it's" | Possessive / contraction | "the model and its parameters"; "it's" should be rare in formal writing |

### 6.3 Run-On Sentences

Long sentences that join multiple independent clauses with commas instead of full stops or semicolons are a common error:

| Run-on | Corrected |
|--------|-----------|
| "The model was trained, the loss converged, the accuracy was high." | "The model was trained. The loss converged, and the accuracy was high." |

A semicolon joins two related independent clauses; a comma alone does not.

---

## 7. Editing Tools, Used Responsibly

### 7.1 Surface-Level Tools

| Tool | What it catches |
|------|-----------------|
| Grammarly | Grammar, spelling, basic style |
| LanguageTool | Grammar, spelling; open-source alternative |
| Hemingway Editor | Sentence length, passive voice, readability score |
| Vale | Customisable style linter for technical writing |

These tools catch surface errors reliably. They do not catch structural problems — weak arguments, missing claims, poor organisation — which are the more important editorial concerns.

### 7.2 LLM-Assisted Editing

Large language models can be used to suggest rewrites, smooth awkward phrasing, and identify unclear passages. Used responsibly, they accelerate editing. Used carelessly, they introduce specific risks:

| Risk | Example |
|------|---------|
| Hallucinated facts | A rewrite asserts a number or citation not in the original |
| Smoothing of distinctive voice | The author's argument is paraphrased into a generic register |
| Substantive change disguised as stylistic | A hedge is removed or a qualifier added without notice |
| Citation generation | Fabricated references that look plausible |

A safe workflow: ask the model to suggest specific edits, accept or reject each individually, and verify any factual or citation changes against source material.

### 7.3 Human Review

No tool replaces a reading by a colleague. The errors that remain after automated checking — argument structure, unstated assumptions, missing context — are the errors that most hurt the work. A single careful read by a knowledgeable peer is more valuable than any number of tool passes.

---

## 8. Resources

- [Strunk and White, *The Elements of Style*](https://www.gutenberg.org/ebooks/37134) — foundational style reference; public-domain edition.
- [Williams, *Style: Lessons in Clarity and Grace* (2014)](https://www.pearson.com/store/p/style-lessons-in-clarity-and-grace/P100002566167) — modern treatment of clear academic prose.
- [Pinker, *The Sense of Style* (2014)](https://www.penguinrandomhouse.com/books/216960/the-sense-of-style-by-steven-pinker/) — cognitive grounding for clear writing.
- [Sword, *Stylish Academic Writing* (2012)](https://www.hup.harvard.edu/books/9780674064485) — empirical study of clarity in academic prose.
- [Mensh and Kording, *Ten Simple Rules for Structuring Papers* (2017)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1005619) — concise structural advice.
- [Purdue OWL — Academic Writing](https://owl.purdue.edu/owl/general_writing/academic_writing/index.html) — comprehensive online reference.
- [Hemingway Editor](https://hemingwayapp.com/) — readability and sentence-level analysis.
- [Vale](https://vale.sh/) — customisable, scriptable style linter.

---

[← Previous: Technical Report and Paper Writing](32_REPORT_AND_PAPER_WRITING_GUIDE.md) | [Index](README.md) | [Next: Presenting Technical ML Projects →](34_PRESENTATION_GUIDE.md)
