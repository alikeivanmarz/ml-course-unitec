# Literature Review

A literature review situates a piece of work within the existing field. It serves two distinct purposes: as a standalone deliverable that surveys what is known about a topic, and as a section within a research paper or proposal that establishes context and identifies gaps. The activities are similar in either case — locate the relevant work, read it efficiently, and synthesize it into a coherent narrative. This guide covers the search strategies, tools, and synthesis techniques used in each phase, along with the common failure modes that produce weak reviews.

**Table of Contents**

1. [Purpose and Scope](#1-purpose-and-scope)
2. [Search Strategies](#2-search-strategies)
3. [Tools and Databases](#3-tools-and-databases)
4. [Snowball Search](#4-snowball-search)
5. [Triage at Scale](#5-triage-at-scale)
6. [Synthesis Tables](#6-synthesis-tables)
7. [From Synthesis to Narrative](#7-from-synthesis-to-narrative)
8. [Common Pitfalls](#8-common-pitfalls)
9. [Resources](#9-resources)

---

## 1. Purpose and Scope

| Type | Goal | Length |
|------|------|--------|
| Standalone survey | Map the state of a field | 10,000+ words; book-length for major surveys |
| Systematic review | Answer a defined question via reproducible methodology | 5,000–15,000 words; PRISMA reporting |
| Background section in a paper | Establish context for a contribution | 500–1,500 words |
| Background section in a proposal | Justify the proposed work | 500–2,000 words |
| Personal reading map | Build expertise in a topic | Open-ended |

The scope of the review determines the search strategy, the synthesis depth, and the criteria for inclusion. A standalone survey aims for completeness; a paper background section aims for sufficiency to position the contribution.

### 1.1 Defining Boundaries

Before searching, define:

- **Topic boundary** — what is in scope, what is adjacent but excluded.
- **Time boundary** — typically the last 5–10 years for active fields, longer for foundational work.
- **Methodological boundary** — empirical, theoretical, or both; supervised or unsupervised; etc.
- **Quality boundary** — peer-reviewed only, or arXiv preprints included.

Without explicit boundaries, the review expands without limit. Reviewers and readers expect the boundaries to be stated explicitly.

---

## 2. Search Strategies

### 2.1 Keyword Construction

A keyword set has three layers:

| Layer | Example for "transformer interpretability" |
|-------|--------------------------------------------|
| Primary terms | "transformer interpretability", "attention interpretability" |
| Synonyms / alternates | "explaining transformers", "transformer explainability" |
| Adjacent terms | "attention probing", "circuit analysis", "mechanistic interpretability" |

Searches start with the primary terms and expand outward as related terminology emerges.

### 2.2 Search Operators

Most academic search engines support operators:

| Operator | Effect |
|----------|--------|
| `"exact phrase"` | Match the phrase verbatim |
| `term1 OR term2` | Match either term |
| `term1 -term2` | Include `term1`, exclude `term2` |
| `author:Smith` | Restrict by author |
| `year:2023..2025` | Restrict by year range |
| `venue:NeurIPS` | Restrict by venue (Semantic Scholar) |

Excessive operator use narrows recall too aggressively in a first pass. Begin broad, then narrow once the relevant subspace is clear.

### 2.3 Survey Papers as Entry Points

A recent survey paper on the topic is the highest-yield starting point. Surveys provide:

- Curated reference lists, often hundreds of papers.
- A taxonomy of approaches, useful for organizing reading.
- Identification of seminal papers and recent state-of-the-art.

Search "survey" or "review" with the topic keywords as a first query.

---

## 3. Tools and Databases

| Tool | Strength | Notes |
|------|----------|-------|
| [Google Scholar](https://scholar.google.com/) | Broadest coverage; fastest queries | Citation counts; "Cited by" link |
| [Semantic Scholar](https://www.semanticscholar.org/) | Influence metrics, citation context, AI-assisted summaries | Strong for CS and biomedicine |
| [Connected Papers](https://www.connectedpapers.com/) | Visual graph of related papers | Useful for finding adjacent work to a known paper |
| [arXiv](https://arxiv.org/) | Preprints; fastest access to new work | No peer review; quality filter required |
| [Papers With Code](https://paperswithcode.com/) | Papers paired with code and benchmark results | Strong for ML; benchmark leaderboards |
| [ACL Anthology](https://aclanthology.org/) | NLP venue archive | Comprehensive for NLP |
| [IEEE Xplore](https://ieeexplore.ieee.org/) | IEEE conferences and journals | Engineering and signal processing |
| [ACM Digital Library](https://dl.acm.org/) | ACM venues | Comprehensive for CS |
| [OpenReview](https://openreview.net/) | Reviews alongside papers (ICLR, NeurIPS) | Reviewer concerns visible publicly |
| [DBLP](https://dblp.org/) | Author-centric publication lists | Useful for tracking an author's full record |

Cross-database searching is necessary; no single source covers all relevant work. Google Scholar is the broadest first stop, but its results require quality filtering.

### 3.1 Reference Managers

A reference manager is essential beyond the first dozen papers.

| Tool | Strengths |
|------|-----------|
| [Zotero](https://www.zotero.org/) | Free, open-source, browser integration, BibTeX export |
| [Mendeley](https://www.mendeley.com/) | Free; PDF reading and annotation built in |
| [JabRef](https://www.jabref.org/) | BibTeX-native; integrates with LaTeX workflows |
| [Paperpile](https://paperpile.com/) | Web-based; tight Google Docs integration; paid |

Choose one early. Switching reference managers mid-project is costly because annotations and metadata do not always migrate cleanly.

---

## 4. Snowball Search

Snowball search uses citation chains rather than keyword search to discover relevant work.

### 4.1 Backward Snowballing

For a known relevant paper, examine its reference list. Papers it cites are candidates for inclusion. Backward snowballing is biased toward older work — it traces influences, not current activity.

### 4.2 Forward Snowballing

Find papers that cite the known paper. Tools:

- **Google Scholar** — "Cited by" link beneath each result.
- **Semantic Scholar** — explicit citation list with context.
- **Inspire-HEP** — for physics; not typical for ML.

Forward snowballing is biased toward newer work; useful for tracking how an idea has been extended.

### 4.3 Saturation

Search continues until new queries return mostly already-known papers — the saturation point. For a focused topic, saturation typically occurs after 30–80 papers; for a broad topic, 200+ may be required.

A useful heuristic: when the next page of search results contains zero or one new relevant papers, the search is approaching saturation.

---

## 5. Triage at Scale

Most papers found will not warrant deep reading. Triage assigns each paper a depth level cheaply.

### 5.1 Inclusion / Exclusion Criteria

For systematic reviews, criteria are explicit and documented.

| Criterion | Example |
|-----------|---------|
| Topic relevance | Must address [primary topic] |
| Time | Published 2018–present |
| Type | Empirical paper or system description |
| Venue quality | Peer-reviewed at recognized venue or arXiv with > N citations |
| Language | English |
| Availability | Full text accessible |

Each excluded paper is recorded with the criterion that excluded it. The PRISMA flow diagram visualises this process for systematic reviews.

### 5.2 Triage Workflow

| Step | Time per paper | Output |
|------|----------------|--------|
| Title screen | 10 seconds | Keep / discard |
| Abstract screen | 1 minute | Read / skip |
| First-pass read | 5–10 minutes | Synthesis-table row / archive |
| Full read | 60+ minutes | Detailed notes |

A 200-paper search list typically reduces to 50 abstracts read, 25 first-pass reads, and 5–10 full reads.

---

## 6. Synthesis Tables

A synthesis table is a row-per-paper, column-per-dimension matrix that supports comparison across the corpus. It is the primary tool for converting reading into a review.

### 6.1 Common Columns

| Column | Purpose |
|--------|---------|
| Citation key | Cross-reference to bibliography |
| Year | Identify temporal trends |
| Method category | Group by approach (e.g., "convolutional", "transformer") |
| Dataset(s) | Identify common benchmarks |
| Metric and result | Enable like-for-like comparison |
| Code released? | Reproducibility signal |
| Key contribution | One-line summary |
| Limitations noted | Honest scope; gaps to exploit |

### 6.2 Theme Tables

A second pass over the synthesis table groups rows by theme rather than chronology. Themes might include:

- Approaches to the same subproblem (e.g., "regularization techniques for X").
- Trade-offs along a single dimension (e.g., "accuracy vs interpretability").
- Successive refinements of a single method.

The theme table, not the per-paper table, is what feeds the narrative.

### 6.3 Tools

A spreadsheet (Google Sheets, Excel) is sufficient and portable. Specialized tools (Notion databases, Airtable) add filtering and views but introduce dependencies. The medium matters less than the discipline of completing every cell consistently.

---

## 7. From Synthesis to Narrative

A literature review section in a paper or proposal is not a list. It is a structured argument that organizes prior work in a way that motivates the new contribution.

### 7.1 Organize by Theme, Not by Paper

Weak literature reviews proceed paper by paper:

> "Smith et al. (2020) proposed X. Jones et al. (2021) extended this with Y. Brown et al. (2022) showed Z."

Stronger reviews organize by theme:

> "Two families of approaches have been proposed for [problem]. The first, exemplified by Smith et al. (2020) and extended by Jones et al. (2021), assumes [property A]. The second, originating with Brown et al. (2022), instead [property B]. Both face the limitation that [gap]."

The thematic structure positions the new contribution against a structured map rather than a chronological list.

### 7.2 Identify the Gap

Every literature review should make explicit what is missing from prior work and motivate the new contribution. The gap is typically one of:

| Gap type | Example phrasing |
|----------|------------------|
| Unaddressed setting | "No prior work addresses [setting] in the regime where [condition]" |
| Methodological limitation | "Existing methods rely on [assumption]; this work removes it by …" |
| Empirical limitation | "Prior evaluations are confined to [dataset / scale]; the present work extends to …" |
| Synthesis | "Two threads — A and B — have developed independently; this work unifies them" |

A literature review without a gap statement does not motivate the work that follows it.

### 7.3 Citation Density

Single citations are reserved for specific claims. Group citations are appropriate when listing examples of a category:

> "Several methods have been proposed (Smith et al., 2020; Jones et al., 2021; Brown et al., 2022)."

Excessive single citations fragment the narrative; excessive group citations hide differences between cited works. The right balance reflects the level of detail needed to support the argument.

---

## 8. Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Listing instead of synthesizing | Paragraph per paper | Reorganize by theme |
| Citing without reading | Citation context is wrong, or paper does not say what is claimed | Read every cited paper, at least to first pass |
| Missing seminal work | Older foundational paper omitted; reviewers will notice | Backward snowballing from any modern paper |
| Outdated review | No work after a certain year | Forward snowballing; targeted recent-year query |
| Confirmation bias | All cited work supports the new contribution | Deliberately search for contradicting work |
| Over-reliance on a single source | Most citations from one author or group | Diversify search across venues and authors |
| No gap statement | Review ends without motivating the new work | Add an explicit "what is missing" paragraph |
| Citing arXiv preprints uncritically | Cited work may have been retracted or substantially revised | Check for peer-reviewed version; cite the latest |

A review that reads as a string of summaries fails its purpose. A review that organizes the field, identifies tensions, and locates the new work within them is the standard a strong submission meets.

---

## 9. Resources

- [Webster and Watson, *Analyzing the Past to Prepare for the Future: Writing a Literature Review* (2002)](https://aisel.aisnet.org/misq/vol26/iss2/3/) — foundational treatment of structured reviews in the IS field.
- [Kitchenham, *Procedures for Performing Systematic Reviews* (2004)](https://www.inf.ufsc.br/~aldo.vw/kitchenham.pdf) — systematic review methodology, originally for software engineering.
- [PRISMA Statement](http://www.prisma-statement.org/) — reporting guidelines for systematic reviews; flow diagram template.
- [Wohlin, *Guidelines for snowballing in systematic literature studies* (2014)](https://dl.acm.org/doi/10.1145/2601248.2601268) — formalisation of snowballing as a search strategy.
- [Zotero documentation](https://www.zotero.org/support/) — reference management workflows.
- [The Litmaps blog — reading workflows](https://www.litmaps.com/learn) — applied search and tracking strategies.

---

[← Previous: Reading ML Research Papers](29_READING_ML_PAPERS_GUIDE.md) | [Index](README.md) | [Next: Research Proposal Writing →](31_RESEARCH_PROPOSAL_GUIDE.md)
