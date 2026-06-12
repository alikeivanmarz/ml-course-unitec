# Step 10 — Final Check

**Before you start:** your own rewrite pass on the report is done.
**You'll finish with:** a verified, professional project — repo and report — with a final commit and tag.

Paste the Master Prompt first, then:

```text
STEP: FINAL QUALITY CHECK

Read PROJECT_STATE.md first. Act as a sceptical external examiner. Go through this
checklist with me, marking each item PASS or FAIL with evidence. Don't fix anything
silently — show me each problem, propose the fix, I decide.

REPRODUCIBILITY
[ ] The README alone is enough to set up the environment, get the data, and run
    everything end-to-end. Try it as far as my machine allows; report what breaks.
[ ] One seed controls all randomness; rerunning reproduces the headline numbers.
[ ] requirements.txt / environment.yml is complete and pinned.
[ ] No data files or model checkpoints committed; data access documented instead.

CODE
[ ] Folder layout matches the README; no dead or duplicate files.
[ ] src/ has docstrings; notebooks have markdown explanations.
[ ] The leakage check from step 05 still passes.

RESULTS
[ ] Every number in the report matches the saved results files exactly.
[ ] Every row in report/EVIDENCE_LEDGER.md is verified.
[ ] The comparison table has ALL models, including the dumb baseline, on the same
    test metrics.
[ ] No claim in the report goes beyond the evidence.

REPORT
[ ] LaTeX compiles cleanly; table of contents matches sections and page numbers.
[ ] Every figure and table has a caption AND is mentioned in the text.
[ ] References: IEEE style, 10+ quality sources, all real and verifiable.

Then: summarise the findings, apply the fixes I approve, final commit, tag it v1.0
(print the git commands if you can't run them).

Last thing — a harder quiz: ask me 5 questions across the whole project (data, split,
methods, results, limitations) so I can practise defending this work out loud.
```
