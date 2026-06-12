# Improve My Project

Use this **after** a review (`12_Review_My_Project.md`) has saved its findings to `REVIEW_FINDINGS.md`. The agent fixes only the issues **you** choose, one at a time, with you present for every change.

Run it in a fresh session. Self-contained — no other prompt needed.

```text
FIX SELECTED REVIEW FINDINGS

My project is at: <path>
My report is at: <path>

Read REVIEW_FINDINGS.md in my project folder (if it's missing, ask me to paste the
review instead). Then work WITH me, not for me:

1. Show me the open issues as a short list (ID, one line each) and ask me which ones
   to fix this session. I choose; you don't add extras.

2. Fix my chosen issues ONE AT A TIME. For each one:
   - tell me what you plan to change and why, and wait for my OK;
   - make the change;
   - show me what changed and explain it in a sentence or two;
   - if it touched code or results, rerun what's needed to confirm nothing broke and
     the report's numbers still match the saved outputs.

3. Mark each finished issue as FIXED in REVIEW_FINDINGS.md, with one line on what was
   done.

4. When my chosen items are done (or I say stop): summarise the session's changes,
   ask me 3 quick questions to check I understood the fixes, and commit the work
   (print the git commands if git isn't set up).

Never invent results, never "improve" numbers, and never touch issues I didn't pick.
```

**When all your chosen fixes are done:** run `12_Review_My_Project.md` again in a fresh session. A clean second pass means you're finished.
