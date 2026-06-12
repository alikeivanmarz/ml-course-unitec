# Step 02 — Plan the Project

**You'll finish with:** an agreed plan in `RESEARCH_PLAN.md`, search terms for your literature reading, and `PROJECT_STATE.md` started. **No code in this step.**

Paste the Master Prompt first, then:

```text
STEP: PLAN THE PROJECT (planning only — no code, no repo yet)

Here is my project input:

<PASTE YOUR FILLED-IN PROJECT INPUT FORM HERE>

Work through this with me:

0. If my input gives a path to a written proposal, READ IT FIRST and pull out
   everything it already answers (problem, dataset, models, metric, group size).
   Summarise back to me what you took from it, point out anything in it that seems
   unclear or contradicts my form answers, and then only ask me about the gaps.

1. Question me until the plan is solid. Go through my input and ask me about anything
   unclear — a few questions at a time. My answers are informal; it's your job to
   firm them up with me. We must end up agreeing on:
   - the one-sentence problem statement (given X, predict Y, measured by Z);
   - the task type (regression, classification, computer vision, NLP, ...);
   - the dataset source and its licence / terms of use;
   - ONE headline metric, chosen now, before any results exist (explain to me why
     choosing it later is bad science);
   - the simplest possible baseline (predicting the average / the majority class)
     that every model must beat;
   - which methods are my MAIN modern methods and which are classical baselines.
     The main solution must use modern techniques (neural networks, deep learning,
     transformers, advanced ensembles); classical models are baselines only. I need
     at least as many different methods as group members, and always more than one.

2. Reality check. Will my models actually train on my computer? If not, suggest
   cheaper routes (transfer learning, pretrained models, smaller versions, sampling
   the data). Flag any data risks you can already see. I decide what we do.

3. Write RESEARCH_PLAN.md: the agreed problem statement, metric, models, the pipeline
   from raw data to final evaluation, and the risks with fallbacks.

4. Literature homework for me. My report needs to engage critically with at least 10
   quality sources (peer-reviewed papers, conference papers, good books). Do NOT list
   papers from memory — instead give me 6-10 Google Scholar search queries for my
   topic, and a short note-taking template (problem / method / data / results /
   limitation / relevance to my project). I will find and read the papers myself and
   keep notes in LITERATURE_NOTES.md. Remind me about this in later steps.

5. Create PROJECT_STATE.md with everything we agreed, this step marked done, and
   "step 03: set up the repo" as next.

Then do the end-of-session routine from the Master Prompt.
```
