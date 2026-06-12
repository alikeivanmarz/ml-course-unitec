# Step 03 — Set Up the Repo

**You'll finish with:** a git repo with clean folders, a pinned environment file, a seed utility, a README skeleton, and a first commit.

Paste the Master Prompt first, then:

```text
STEP: SET UP THE REPOSITORY

Read PROJECT_STATE.md first. Today we build the project skeleton — no data work, no
models. Remember: one piece at a time, and explain each piece to me as we go.

1. Folders:
   data/raw  data/processed  notebooks  src  results  results/figures  report
   Plus a .gitignore that keeps data files, model checkpoints, and caches OUT of git
   (explain each ignore rule briefly). Big data files never get committed — the
   README will say how to get the data instead.

2. Environment: requirements.txt (ask me if I'd rather use conda's environment.yml),
   with pinned versions, only the libraries we actually need. Tell me why pinning
   matters.

3. One seed to rule everything: src/seed.py with set_seed(seed) covering Python,
   NumPy, and our ML framework. The seed value lives in ONE place.

4. A small config (src/config.py — or ask me if I prefer YAML) for paths, the seed,
   and split proportions, so nothing important is hard-coded twice.

5. README skeleton: title + one-paragraph description, authors, how to set up the
   environment, how to get the data, how to run everything, folder layout. Fill what
   we know, mark the rest TODO.

6. git init (use the git fallback from the Master Prompt if needed), first commit,
   update PROJECT_STATE.md.

Then do the end-of-session routine.
```
