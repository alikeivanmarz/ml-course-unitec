# Agent Project Prompts — Simple Pack

Copy-paste prompts for building a complete ML project **together with** any coding agent (Claude Code, Copilot agent mode, Codex, Cursor, ...): from your idea to a clean git repo and a LaTeX report.

**The most important idea:** you are present the whole time. The agent never runs ahead on its own — it tells you what it wants to do, waits for your OK, does one small step, then shows and explains the result. You make every decision. You learn by being in the loop, not by watching.

> **AI-use note:** use this workflow only in ways your lecturer and course policy permit. You are the author — you must be able to explain and defend every line of code and every sentence of the report.

## How to use

1. Fill in `01_Project_Input_Form.md` before your first session — or, if you already have a written proposal, just put its file path in question 0 and answer only what the proposal doesn't cover.
2. Each session: paste `00_Master_Prompt.md` first, then **one** step file. One step per session.
3. Work through it together. Answer the agent's questions. Make the choices.
4. At the end of the step, the agent saves progress to `PROJECT_STATE.md` and asks you 3 quick questions to check you understood. Then stop the session.
5. Next session: Master Prompt + the next step file.

## The files, in order

| File | What it does |
|------|--------------|
| `00_Master_Prompt.md` | Rules for the agent — paste at the start of **every** session |
| `01_Project_Input_Form.md` | You fill this in once, before starting |
| `02_Plan_The_Project.md` | Agree the plan — no code yet |
| `03_Setup_The_Repo.md` | Git repo, folders, environment, seeds |
| `04_Explore_The_Data.md` | Load data, explore it, save the figures |
| `05_Prepare_And_Split.md` | Train/validation/test split + preprocessing |
| `06_Baseline_Models.md` | Simple classical models first |
| `07_Advanced_Models.md` | Your main modern models (may take several sessions) |
| `08_Evaluate_And_Analyse.md` | Final test results, error analysis |
| `09_Write_The_Report.md` | The LaTeX report (may take 2–3 sessions) |
| `10_Final_Check.md` | Full quality check before you call it done |
| `11_Helper_Prompts.md` | Resume / explain / debug / quiz-me prompts for any time |
| `12_Review_My_Project.md` | Standalone: independent review of ANY finished project and report (yours or built with these prompts) — saves numbered findings to `REVIEW_FINDINGS.md` |
| `13_Improve_My_Project.md` | Fixes the review findings YOU choose, one at a time, with you in the loop |

## Working efficiently with the agent

These habits keep every session short, focused, and easy to follow — whatever agent or model you use:

- One step per session, always. Stop when the step is done, even when it's going well.
- No subagents, no web browsing — you supply the dataset; you find and read the papers yourself (the agent gives you search terms in step 02).
- If the agent starts getting confused mid-session, use the "Save and stop" helper in `11_Helper_Prompts.md` and start fresh.
