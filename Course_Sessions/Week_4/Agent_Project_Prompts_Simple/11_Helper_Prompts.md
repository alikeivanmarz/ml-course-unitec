# Helper Prompts

Short prompts for any moment. In a new session, paste the Master Prompt first.

## Where are we? (start of any session)

```text
Read PROJECT_STATE.md and tell me exactly where the project is: what's done, what's
next, and anything waiting on me. Don't start any work until I say so.
```

## Explain it to me (use whenever you don't fully understand something)

```text
Explain <file / function / concept> to me like a mentor: what it does, why we did it
this way, and what would go wrong if we did it differently. Plain language, small
example. Then ask me 3 questions to check I really got it, and give me feedback on my
answers.
```

## Something's broken

```text
Something is wrong: <describe what happened and paste the error>.
Debug it WITH me, one step at a time:
1. Reproduce it in the smallest way possible.
2. Find exactly where it fails.
3. Give me 2-3 likely causes, most likely first, and a cheap test for each.
4. Test them one by one and show me the evidence.
No rewriting whole files blind. Explain the root cause to me before fixing it.
```

## Quiz me (great before presenting or defending the project)

```text
Read PROJECT_STATE.md and results/model_comparison.csv, then act as a sharp examiner.
Ask me questions about my project one at a time, getting progressively harder — from
"what's in your dataset" to "why should anyone trust your comparison". After each
answer, give me honest feedback and the answer a strong student would have given.
Keep going until I say stop.
```

## Save and stop (use the moment the agent seems confused or the session degrades)

```text
Stop the current work. Write everything essential into PROJECT_STATE.md right now:
what's done, exact state of the files, decisions made, and the precise next step.
Commit. We're ending this session and starting fresh.
```
