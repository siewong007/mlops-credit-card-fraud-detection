# 95+ delivery checklist

- Baseline design commit: `db3f518`
- Internal target: 97/100
- Delivery window: 21 days
- Feature freeze: end of Day 18
- Technical snapshot: `846ff54444a4d9123ce552a0b22a58ef546e4c5a`
  (recorded 2026-07-30 before the authoritative real-data run)
- Evidence freeze commit: open; requires the authoritative real-data run

## Gates

- [ ] G1 Core contracts pass with zero skipped tests — local technical proof
  passed; human team assignment and sign-off remain open.
- [ ] G2 Make, DVC, Docker, and CI reproduction pass — local Make, DVC, and
  Docker proof passed; hosted CI is not run and requires push approval.
- [ ] G3 Authoritative real-data evidence is frozen — open.
- [ ] G4 README, diagrams, model card, and report agree with evidence — open.
- [ ] G5 Deck, fallback demo, Q&A, and four reflections pass review — open.
- [ ] G6 Final bundle is visually inspected and submission-ready — open.

## Frozen filenames

- `train.csv`
- `model_valid.csv`
- `calibration.csv`
- `prod_1.csv`
- `prod_2.csv`
- `prod_3.csv`
- `validation_report.json`
- `experiment_comparison.json`
- `promotion_record.json`
- `operating_point.json`
- `metrics_default.json`
- `metrics_operating.json`
- `drift_summary.json`
- `trigger_decisions.json`
- `trigger_log.md`

## Prior core history

Recorded 2026-07-29: `pytest -q -ra` passed 40 tests with zero skips;
`python -m compileall -q src tests`, strict JSON serialization, and
`git diff --check` all exited successfully.

## G1 local technical proof

Recorded 2026-07-30 against technical snapshot
`846ff54444a4d9123ce552a0b22a58ef546e4c5a`:

- `SOURCE_COMMIT=846ff54444a4d9123ce552a0b22a58ef546e4c5a
  make verify` ran from a disposable archive of that exact commit. All pipeline
  stages, the manifest, and the dashboard completed.
- The generated-evidence full suite collected and completed 89 tests with zero
  skips. `tests/test_evidence_consistency.py` also passed separately (1 test).
- The authoritative evidence-independent suite passed 87 tests; the two
  integration-marked tests were deselected, not skipped.
- The disposable manifest reported source commit
  `846ff54444a4d9123ce552a0b22a58ef546e4c5a` and Python `3.13.9`.
- No synthetic reports, models, raw data, or dashboard were copied back to the
  authoritative worktree.

Automated independent technical reviews covering snapshot Tasks 4–6 (Codex
agents, not human/team-member sign-offs):

- Archimedes (`/root/repro_task_4_review`) — Task 4 and initial Task 5 review.
- Bernoulli (`/root/core_final_fixes`) — Task 5 fix re-review.
- `/root/repro_task_5` — Task 6 review; no display name was available.

Human cross-workstream reviewer assignments and human G1 sign-off have not
been supplied and remain open.

## G2 local reproducibility proof

Recorded 2026-07-30 against the same technical snapshot:

| Subproof | Result |
|---|---|
| Make | Exact-snapshot disposable `make verify` passed; 89 tests completed with zero skips and the consistency test passed. |
| DVC | `scripts/verify_dvc_clean_clone.sh` passed all stages and its consistency test; caller worktree raw checksum was `missing` before and after. |
| DVC remote | `dvc remote list` was empty before and after the clean-clone proof. |
| DVC lock | `dvc.lock` SHA-256: `ea23c87054be72e9e9f15435fb7cd944634887bea34a0901f2f8fee3b8a09968`. |
| Docker runtime | Image reported exactly `Python 3.13.9`; embedded `SOURCE_COMMIT` matched the technical snapshot. |
| Docker verification | Default image command (`make verify`) passed. |
| Docker image | `sha256:bf3401f884466573b2e5e22c7882ee58ebdc114737229228b5719ca51a59171a` (`linux/arm64`). |
| Workflow static proof | 16 focused reproducibility/trigger tests passed; both workflow YAML files parsed; stale workflow search returned no matches; both workflow Python versions are `3.13.9`. |
| Raw-data safety | Authoritative worktree raw checksum remained `missing`; main-checkout real raw SHA-256 remained `1700322b377ac8340ab6b75f22b974944fbcaf5b4f0daf3d5e8f620d96313026`. |
| Repository hygiene | `git diff --check` passed and the authoritative worktree was clean before this checklist-only update. |
| Hosted GitHub Actions | **Not run / requires push approval.** No CI URL or hosted result is claimed. |

Automated Codex reviews are identified under G1. They are technical review
evidence only and do not replace human assignment or sign-off.

## Open delivery work

### G3 — authoritative evidence

- [ ] Run the pipeline once against the preserved real CSV after technical
  freeze.
- [ ] Freeze and review the resulting real-data evidence commit.

### G4 — documentation

- [ ] Reconcile README, workflow diagrams, model card, and report against the
  frozen real-data evidence.
- [ ] Complete report word-count and visual-QA gates.

### G5 — presentation and reflections

- [ ] Complete and review the deck, fallback demo, Q&A, and rehearsals.
- [ ] Obtain four authentic 500–800-word team-member reflections; no reflection
  or personal contribution is attributed here.

### G6 — submission bundle

- [ ] Visually inspect every final artefact and exact upload file.
- [ ] Record the final commit, hosted CI URL, human assignments, human sign-off,
  and submission inspection only after those events occur.

## Human assignments and sign-off

| Workstream | Implementer | Cross-reviewer | Status |
|---|---|---|---|
| Data engineering | Not supplied | Not supplied | open |
| Modelling | Not supplied | Not supplied | open |
| Monitoring | Not supplied | Not supplied | open |
| Platform/documentation | Not supplied | Not supplied | open |

No student identity, human contribution, hosted CI run, or approval is inferred
from automated Codex work.
