# Individual reflection — template (500–800 words per student)

Deliverable per briefing §14. **One reflection per student, in your own words.**

> ⚠️ **This is a template, not a submission.** The briefing (§18) requires each
> reflection to reflect the student's *own* understanding and contribution. Do
> **not** submit AI-generated reflections. Use the prompts below to write your
> own; delete this file (or the prompts) from your final hand-in. Each student
> should copy this into their own document and fill it in.

Recommended length: 500–800 words. Recommended structure: three parts —
**contribution**, **learning**, **challenges** — as the rubric names those.

## 1. Your contribution (≈40%)

- Which workstream/stage did you own? (See `PLAN.md` §7: Data engineering /
  Modelling / Monitoring / Platform & docs.)
- What specific files or components did you write or lead? Name them
  (e.g. `src/drift.py`, the trigger rule, the CI workflow, the report §8).
- What decisions did you make, and why? (e.g. "I chose PSI > 0.2 as the drift
  threshold because…")
- How did you collaborate — PR reviews, pairing, integration work?

## 2. What you learned (≈35%)

- One MLOps concept that was new to you and that you now understand (e.g. why
  PR-AUC over accuracy for imbalance; what a model registry is for; why time-
  based splitting matters; how a drift metric like PSI actually works).
- Something that changed how you think about ML "in production" versus in a
  notebook.
- A tool you can now use that you couldn't before (MLflow, Pandera, Evidently,
  DVC, GitHub Actions…), and what clicked.

## 3. Challenges and how you handled them (≈25%)

- A concrete problem you hit (a bug, a version conflict, a design disagreement,
  a stage that wouldn't reproduce) and how you resolved it.
- Something you'd do differently next time.
- An honest limitation of the project you understand (e.g. synthetic data vs real
  drift; local MLflow backend; label delay in production).

## Checklist before you submit

- [ ] 500–800 words, in your own voice.
- [ ] Names the specific stages/files you worked on.
- [ ] Mentions at least one concept you genuinely learned.
- [ ] Describes a real challenge and your response.
- [ ] Acknowledges any AI assistance you used and what you did to verify it
      (§18).

---

*Illustrative fragment (do not copy — write your own):* "I owned the monitoring
stage. I implemented `src/drift.py`, computing PSI and the KS test natively so
the trigger wouldn't depend on Evidently's API. The hardest part came when we ran
on the real data: the KS test flagged 100% of features as drifted on every batch.
Working out why — that statistical significance grows with sample size, so at
28k rows KS fires on trivial differences — taught me to base the drift flag on
PSI's magnitude instead, and to calibrate the threshold against the dataset's
~40% natural drift rather than a textbook default…"
