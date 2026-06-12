# Step 09 — Write the Report

**Before you start:** your literature reading must be done — at least 10 quality sources read, with your notes in `LITERATURE_NOTES.md`. The agent cannot read for you and is not allowed to invent citations.

**You'll finish with:** a compiled LaTeX report built only from your project's real files, and a checklist for your own rewrite pass.

This step usually takes **2–3 sessions** (setup + first sections; methodology + results; polish).

Paste the Master Prompt first, then:

```text
STEP: WRITE THE LATEX REPORT

Read PROJECT_STATE.md first. We write the report in report/ — section by section,
together, and STRICTLY from this project's real files: PROJECT_STATE.md,
RESEARCH_PLAN.md, LITERATURE_NOTES.md, results/*.csv, results/figures/* and the
captions file. No fact, number, or citation from anywhere else. If a section needs
something we don't have, stop and tell me what's missing.

WRITING STYLE — follow this in every section you draft:
- Write like a careful human author: simple, direct, non-verbose. Clarity beats
  cleverness. Prefer plain words ("use" not "utilise", "shows" not "demonstrates the
  importance of") and cut filler ("It is important to note that...", "In today's
  rapidly evolving world...").
- Vary the rhythm. Mix short and long sentences; don't make every paragraph the same
  shape and size. Avoid starting sentence after sentence with "Moreover",
  "Furthermore", "Additionally" — use transitions that show the actual relationship
  between ideas, or none when the connection is obvious.
- Avoid templated patterns: "X, Y, and Z" triple lists in sentence after sentence;
  "not just X but Y"; tail clauses like "...enabling X, ensuring Y, supporting Z";
  paragraph-ending sentences that just restate the paragraph ("Overall...",
  "In summary..."); vague praise words (robust, comprehensive, seamless, pivotal,
  cutting-edge).
- Be concrete. A number, a dataset fact, a method name beats an abstract claim every
  time.
- Keep the academic register: precise, evidence-led, never chatty. Keep every
  technical term exact. Never fake "humanness" with errors or quirks — natural means
  clear and specific, not informal.

1. Set up report/main.tex (clean article style) and report/references.bib:
   title page (title, author name(s) and IDs, date), automatic table of contents,
   numbered sections and pages, IEEE-style numeric citations. Compile NOW and fix any
   errors before writing content. Every figure/table goes in as a captioned, numbered
   float that is referenced in the text — no orphan images.

2. Write the sections in this order, showing me each one before moving on. The word
   ranges are professional guides, not straitjackets:
   - Introduction & Literature Review (800-1200 words): the problem and why it
     matters; a CRITICAL review of the 10+ sources in my LITERATURE_NOTES.md —
     compare them, find the gaps, position my project; end with my research question.
     Cite ONLY papers from my notes. Fewer than 10? Send me back to my reading.
   - Dataset & EDA (600-1000 words): source + licence + access link, the data
     dictionary, the key EDA findings with our saved figures, the preprocessing, the
     split strategy and why.
   - Methodology (800-1200 words): each method and why it fits; a block diagram of
     the pipeline (data -> preprocessing -> split -> training -> tuning ->
     evaluation); the tuning strategy and ranges we actually tried (from
     experiments_log.csv); the setup: libraries + versions, hardware, seed.
   - Results & Discussion (800-1200 words): the final comparison table of ALL models
     on the same test metrics; the learning curves and confusion matrices / residual
     plots; which model wins and WHY, argued from the evidence; the error analysis;
     the limitations.
   - Conclusion & Future Work (200-400 words): key findings, recommended model,
     concrete next steps.
   - Abstract (200-300 words) — written LAST: problem, approach, key numbers, final
     recommendation.
   - References: IEEE style from references.bib.

3. As you write, keep report/EVIDENCE_LEDGER.md: every number or factual claim in the
   report, the file/table/figure it came from, and where it appears. A claim you
   can't source doesn't get written — flag it to me instead.

4. Compile the full document, fix errors, check the table of contents matches the
   sections.

5. Hand it over for MY pass: walk me through the evidence ledger, then give me my
   author checklist — verify every number, rewrite anything I couldn't defend out
   loud in my own words, and confirm every reference is a real paper I actually read.
   Also point me to the sections where the wording still reads most generic or
   templated, so I rewrite those in my own voice first. This draft is raw material;
   the final words must be mine.

6. Update PROJECT_STATE.md and commit.

Then do the end-of-session routine.
```
