# Presentation, Demo, and Reflections Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a ten-slide, evidence-led 8–10-minute presentation with reliable demo fallbacks, rehearsed Q&A, and four authentic 500–800-word individual reflections.

**Architecture:** The deck consumes the same frozen JSON and figures as the report, assigns each workstream owner only their genuine contribution, and uses pre-run evidence instead of live training. Reflections remain private and human-authored; automation is limited to factual, rubric, word-count, clarity, and privacy checks.

**Tech Stack:** PowerPoint, PDF, `presentations:Presentations`, `@oai/artifact-tool`, browser screenshots, Markdown, DOCX/PDF reflection files, Pandoc, Git evidence, and rehearsal logs.

## Global Constraints

- Begin slide production only after authoritative real-data evidence is frozen.
- Use exactly ten slides and target 9:15–9:40; the official acceptable duration is 8–10 minutes.
- One slide communicates one primary message.
- Visible metrics must come from frozen JSON, never memory or stale prose.
- Use operating-threshold evidence as primary and threshold 0.5 only as comparison.
- Each speaker presents work they genuinely contributed.
- Do not train live. Pre-run the pipeline and demonstrate fast evidence checks, static outputs, MLflow, and the dashboard.
- The deck must remain usable offline through local screenshots and PDF.
- Final deliverables are `submission/Fraud_MLOps_Presentation.pptx` and `.pdf`.
- Use the `presentations:Presentations` skill and `@oai/artifact-tool`; do not use `python-pptx`.
- With no user-supplied visual template, use the bundled Codex Grid layout route.
- Slides must use at least 50 pt deck title, 35 pt slide titles, 24 pt subheads/callouts, and 16 pt body text.
- Every externally sourced visual or non-trivial claim requires a `[Sources]` block in speaker notes.
- Render and inspect every slide individually; zero unintended overlap, overflow, clipping, or title wrapping is allowed.
- Each reflection must be 500–800 authentic words; target 550–700.
- The agent must not draft first-person reflection prose or invent contribution, learning, challenge, collaboration, emotion, or future goals.
- Private reflections and recordings stay under ignored `submission-private/` and never enter the public repository or release.
- Do not publish, tag, release, or submit without explicit user approval.

---

## File Structure

**Create publicly**

- `submission/Fraud_MLOps_Presentation.pptx`
- `submission/Fraud_MLOps_Presentation.pdf`
- `submission/demo-assets/01-mlflow-promotion.png`
- `submission/demo-assets/02-operating-point.png`
- `submission/demo-assets/03-monitoring-dashboard.png`
- `submission/demo-assets/04-trigger-decision.png`
- `submission/demo-assets/05-ci-verification.png`
- `docs/DEMO_FALLBACK.md`
- `docs/DEMO_QA.md`

**Modify publicly**

- `docs/DEMO_SCRIPT.md`
- `docs/REFLECTION_TEMPLATE.md`
- `submission/MANIFEST.md`
- `submission/SHA256SUMS.txt`

**Create privately; never commit**

- `submission-private/reflections/CONTRIBUTION_LEDGER.md`
- `submission-private/reflections/01_Data_Engineering_Reflection.docx`
- `submission-private/reflections/01_Data_Engineering_Reflection.pdf`
- `submission-private/reflections/02_Modelling_Reflection.docx`
- `submission-private/reflections/02_Modelling_Reflection.pdf`
- `submission-private/reflections/03_Monitoring_Reflection.docx`
- `submission-private/reflections/03_Monitoring_Reflection.pdf`
- `submission-private/reflections/04_Platform_Documentation_Reflection.docx`
- `submission-private/reflections/04_Platform_Documentation_Reflection.pdf`
- `submission-private/reflections/REVIEW_SIGNOFFS.md`
- `submission-private/rehearsals/REHEARSAL_LOG.md`
- `submission-private/demo/Fraud_MLOps_Demo_Fallback.mp4`
- `submission-private/Fraud_MLOps_University_Submission_v1.0.0.zip`

### Task 1: Freeze the Slide Evidence Map

**Files:**
- Create: `docs/delivery/presentation-evidence-map.md`
- Reference: frozen `reports/*.json`, figures, report, evidence matrix, and model card

**Interfaces:**
- Consumes: authoritative evidence commit
- Produces: exact slide → claim → source → owner → timing map

- [ ] **Step 1: Confirm the evidence package**

Run:

```bash
python -m pytest -q tests/test_evidence_consistency.py
python - <<'PY'
import json
from pathlib import Path

for name in (
    "experiment_comparison.json",
    "promotion_record.json",
    "operating_point.json",
    "metrics_default.json",
    "metrics_operating.json",
    "drift_summary.json",
    "trigger_decisions.json",
    "run_manifest.json",
):
    json.loads((Path("reports") / name).read_text())
    print(name)
PY
```

Expected: consistency passes and every source parses.

- [ ] **Step 2: Create the ten-slide source map**

Use this exact structure:

| Slide | Time | Primary message | Frozen sources | Genuine owner |
|---:|---:|---|---|---|
| 1 | 0:00–0:40 | Fraud detection must be operated, not merely trained once | briefing; `reports/run_manifest.json` | Data engineering |
| 2 | 0:40–1:20 | The original notebook lacked operational safeguards | baseline notebook; `notebooks/README.md` | Data engineering |
| 3 | 1:20–2:10 | Eight requirements become a validation-gated architecture | `docs/workflow_diagrams.md`; `docs/RUBRIC_EVIDENCE.md` | Data engineering |
| 4 | 2:10–3:00 | Six chronological slices prevent selection/calibration leakage | `reports/validation_report.json`; `dvc dag` | Data engineering |
| 5 | 3:00–4:00 | MLflow traces both candidates and promotes the best PR-AUC run | comparison and promotion JSON | Modelling |
| 6 | 4:00–5:10 | Cost calibration changes the operating decision | operating/default JSON; two confusion matrices; trade-off figure | Modelling |
| 7 | 5:10–6:30 | Native monitoring separates healthy and degraded batches and records label availability | drift JSON; PSI figures | Monitoring |
| 8 | 6:30–7:40 | A structured review decision closes the loop through human approval | trigger JSON/log; architecture | Monitoring |
| 9 | 7:40–8:40 | Make, DVC, Docker, tests, CI, and dashboard make claims reproducible | run manifest; DVC lock; CI; dashboard | Platform/documentation |
| 10 | 8:40–9:30 | Honest limitations demonstrate engineering judgment | model card; report limitations | Platform/documentation |

- [ ] **Step 3: Human-confirm speaker ownership**

Replace role labels with confirmed team member names only after each member
confirms the relevant contribution. If ownership differs from the table,
change the speaker assignment, not the contribution story.

- [ ] **Step 4: Add exact claim fields**

For slides 5–9, record the exact promoted model ID, run ID, registry version,
operating threshold, default/operating metrics, batch statuses, reason codes,
test count, and source commit beside their source path. Copy values using a
short JSON-reading command, not manual transcription.

- [ ] **Step 5: Commit the map**

```bash
git add docs/delivery/presentation-evidence-map.md
git commit -m "docs: map presentation claims to evidence"
```

### Task 2: Capture Offline Demo Assets

**Files:**
- Create: `submission/demo-assets/01-mlflow-promotion.png`
- Create: `submission/demo-assets/02-operating-point.png`
- Create: `submission/demo-assets/03-monitoring-dashboard.png`
- Create: `submission/demo-assets/04-trigger-decision.png`
- Create: `submission/demo-assets/05-ci-verification.png`

**Interfaces:**
- Consumes: local MLflow/database, report figures, dashboard, trigger JSON/log, successful CI
- Produces: five sharp, cropped, legible offline screenshots

- [ ] **Step 1: Pre-run local evidence**

Run:

```bash
SOURCE_COMMIT="$(git rev-parse HEAD)" make verify
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Keep MLflow running only for screenshot capture. Do not retrain during the
presentation.

- [ ] **Step 2: Capture MLflow promotion**

Use browser control to open the local MLflow experiment and registered
`fraud-detector` model. Capture the winning candidate run ID, comparison
metric, registered version, and `champion` alias in one readable screenshot.
Crop browser chrome that does not help the claim.

- [ ] **Step 3: Capture the operating point and decisions**

Create:

- `02-operating-point.png` from the final threshold trade-off and
  default/operating comparison;
- `03-monitoring-dashboard.png` from local `site/index.html`, including source,
  promoted model, operating threshold, and statuses;
- `04-trigger-decision.png` from the structured decision table with reason
  codes and human-approval note.

Use existing generated assets; do not redraw metrics manually.

- [ ] **Step 4: Capture successful CI**

After push approval and a successful final CI run, capture the job summary
showing fast tests, DVC reproduction, complete verification, evidence
consistency, Docker verification, commit SHA, and green result. If push approval
has not yet been granted, delay this screenshot rather than fabricating it.

- [ ] **Step 5: Inspect screenshot quality**

Open every image at its intended slide crop. Reject images with unreadable text,
browser notifications, personal account data, secrets, clipped values, stale
commit IDs, or synthetic results presented as authoritative.

- [ ] **Step 6: Commit safe public screenshots**

```bash
git add submission/demo-assets
git commit -m "docs: add offline demo evidence"
```

### Task 3: Build the Ten-Slide PowerPoint

**Files:**
- Create: `submission/Fraud_MLOps_Presentation.pptx`
- Create: `submission/Fraud_MLOps_Presentation.pdf`
- Temporary only: `tmp/presentation-build/build-deck.mjs`

**Interfaces:**
- Consumes: presentation evidence map, frozen JSON/figures, demo screenshots
- Produces: exactly ten slides with notes, timings, transitions, and source blocks

- [ ] **Step 1: Load and read the presentation runtime**

Call the workspace dependency loader. Read the complete
`presentations:Presentations` skill, `style_guidelines.md`,
`artifact_tool_docs/API_QUICK_START.md`, `artifact_tool_docs/api/API_DOCS.md`,
the Codex Grid `ARTIFACT.md`, `design_tokens.json`, template registry, preview
montage, and only the shortlisted layout modules.

Use:

```bash
SKILL_DIR="/Users/goaltosuceed/.codex/plugins/cache/openai-primary-runtime/presentations/26.727.11326/skills/presentations"
BUNDLED_NODE="/Users/goaltosuceed/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node"
BUNDLED_PYTHON="/Users/goaltosuceed/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
TMP_DIR="/Volumes/APPLE EXTERNAL SSD /Personal Projects/fraud-mlops/tmp/presentation-build"
FINAL_PPTX="/Volumes/APPLE EXTERNAL SSD /Personal Projects/fraud-mlops/submission/Fraud_MLOps_Presentation.pptx"
mkdir -p "$TMP_DIR"
"$BUNDLED_NODE" "$SKILL_DIR/container_tools/setup_artifact_tool_workspace.mjs" \
  --workspace "$TMP_DIR"
```

- [ ] **Step 2: Create the exact slide-data source**

In `build-deck.mjs`, define the content separately from layout code:

```javascript
const slideSpec = [
  { number: 1, title: "From fraud model to fraud operation", time: "0:00–0:40",
    message: "The goal is a repeatable decision system, not one notebook score.",
    sources: ["reports/run_manifest.json", "https://www.openml.org/d/1597",
      "MLOpsMaster_Final_Project_Briefing_Credit_Card_Fraud.docx"] },
  { number: 2, title: "The notebook stopped where operations begin", time: "0:40–1:20",
    message: "No validation, lineage, tracking, monitoring, or review gate.",
    sources: ["notebooks/README.md", "baseline notebook"] },
  { number: 3, title: "Eight requirements, one gated architecture", time: "1:20–2:10",
    message: "Each requirement maps to an executable stage and evidence contract.",
    sources: ["docs/workflow_diagrams.md", "docs/RUBRIC_EVIDENCE.md"] },
  { number: 4, title: "Chronology separates three different decisions", time: "2:10–3:00",
    message: "50% train; 10% select; 10% calibrate; three unseen 10% batches.",
    sources: ["reports/validation_report.json", "dvc.yaml"] },
  { number: 5, title: "Promotion is traceable to the winning run", time: "3:00–4:00",
    message: "MLflow compares candidates by model-validation PR-AUC and registers the winner.",
    sources: ["reports/experiment_comparison.json", "reports/promotion_record.json"] },
  { number: 6, title: "The operating threshold changes the business decision", time: "4:00–5:10",
    message: "A 100:1 missed-fraud cost makes threshold 0.5 a comparison, not the policy.",
    sources: ["reports/operating_point.json", "reports/metrics_default.json",
      "reports/metrics_operating.json"] },
  { number: 7, title: "Monitoring distinguishes drift from missing labels", time: "5:10–6:30",
    message: "Feature and prediction signals remain available while delayed performance is explicit.",
    sources: ["reports/drift_summary.json", "reports/figures/psi_prod_3.png"] },
  { number: 8, title: "Retraining is a structured recommendation", time: "6:30–7:40",
    message: "Stable reason codes lead to human review; no model is silently replaced.",
    sources: ["reports/trigger_decisions.json", "reports/trigger_log.md"] },
  { number: 9, title: "Four reproduction paths support the same claim", time: "7:40–8:40",
    message: "Make, DVC, Docker, and CI converge on tested evidence and a static dashboard.",
    sources: ["reports/run_manifest.json", "dvc.lock", ".github/workflows/ci.yml"] },
  { number: 10, title: "The limits define the next responsible step", time: "8:40–9:30",
    message: "Two-day anonymized data, injected drift, delayed labels, and local governance remain.",
    sources: ["docs/MODEL_CARD.md", "docs/TECHNICAL_REPORT.md"] },
];
```

Load every numeric value from JSON at build time and fail if any value does not
match `docs/delivery/presentation-evidence-map.md`.

- [ ] **Step 3: Compose the evidence-led visuals**

Use these compositions:

1. minimal title with one objective line and small real-data identity;
2. original linear flow plus five concise operational gaps;
3. final architecture visual plus eight requirement labels;
4. one horizontal 50/10/10/10/10/10 chronology with three decision labels;
5. two-candidate comparison chart plus promoted run/version callout;
6. default and operating confusion matrices side-by-side with cost/recall delta;
7. healthy/degraded monitoring comparison using real PSI/dashboard evidence;
8. structured status/reason-code table ending at a human approval gate;
9. CI screenshot plus compact Make/DVC/Docker/dashboard proof line;
10. four limitations and one evidence-backed conclusion.

Prefer existing figures and screenshots. Use native PowerPoint shapes only for
the simple chronology and approval flow, with connectors created before nodes.
Do not add decorative stock imagery that competes with evidence.

- [ ] **Step 4: Add speaker notes**

Before authoring notes, obtain the actual speaker name for every slide; stop
this step if the team has not confirmed them. Every slide's notes must contain
the actual owner, target time from the evidence map, concise talk track in that
speaker's style, one-sentence transition, and this source block:

```text
[Sources]
- exact repository path or authoritative external URL
[/Sources]
```

Talk-track phrasing requires human confirmation. Source blocks must include the
OpenML/ULB source on slide 1 and exact repository evidence on all quantitative
slides.

- [ ] **Step 5: Export the PPTX**

Implement the deck in the temporary ES module with `@oai/artifact-tool`, export
to `FINAL_PPTX`, and confirm the file opens. Do not keep the temporary builder
as a submission artifact.

- [ ] **Step 6: Run overflow and slide rendering checks**

```bash
SKILL_DIR="/Users/goaltosuceed/.codex/plugins/cache/openai-primary-runtime/presentations/26.727.11326/skills/presentations"
BUNDLED_PYTHON="/Users/goaltosuceed/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
"$BUNDLED_PYTHON" "$SKILL_DIR/container_tools/slides_test.py" \
  submission/Fraud_MLOps_Presentation.pptx
"$BUNDLED_PYTHON" "$SKILL_DIR/container_tools/render_slides.py" \
  submission/Fraud_MLOps_Presentation.pptx
"$BUNDLED_PYTHON" "$SKILL_DIR/container_tools/create_montage.py" \
  --input_dir submission/Fraud_MLOps_Presentation \
  --output_file tmp/presentation-build/montage.png
```

Expected: zero overflow and no unintended overlap warnings.

- [ ] **Step 7: Inspect every slide and iterate**

Inspect each `slide-*.png` individually at full size, then inspect the montage
for narrative rhythm. Correct title wrapping, small type, clipped screenshots,
unreadable chart labels, inconsistent footers, source/metric mismatches,
unresolved internal notes, and repetitive adjacent layouts. Re-render after
each correction.

- [ ] **Step 8: Export and inspect the PDF**

```bash
soffice --headless --convert-to pdf --outdir submission \
  submission/Fraud_MLOps_Presentation.pptx
pdfinfo submission/Fraud_MLOps_Presentation.pdf
rm -rf tmp/presentation-pdf
mkdir -p tmp/presentation-pdf
pdftoppm -png submission/Fraud_MLOps_Presentation.pdf \
  tmp/presentation-pdf/slide
```

Inspect all ten PDF pages. Confirm exactly ten slides and matching PPTX/PDF
content.

- [ ] **Step 9: Commit the verified deck**

```bash
git add submission/Fraud_MLOps_Presentation.pptx \
  submission/Fraud_MLOps_Presentation.pdf
git commit -m "docs: add visually verified presentation"
```

### Task 4: Finalize Demo Script, Fallback, and Q&A

**Files:**
- Modify: `docs/DEMO_SCRIPT.md`
- Create: `docs/DEMO_FALLBACK.md`
- Create: `docs/DEMO_QA.md`

**Interfaces:**
- Consumes: final slide numbers, speakers, timings, local assets, and evidence
- Produces: one-to-one presentation script, offline decision tree, and evidence-linked answers

- [ ] **Step 1: Align the demo script to the final deck**

Use exactly one row per slide with start/end time, confirmed speaker, message,
screen/asset, transition, and fallback. Replace every stale threshold, metric,
test count, URL, and old five-slice reference.

The live proof is:

```bash
python -m pytest -q tests/test_evidence_consistency.py
```

Then open pre-generated evidence. Do not run training live.

- [ ] **Step 2: Create the preflight and fallback decision tree**

`docs/DEMO_FALLBACK.md` must specify:

- 24 hours before: final clean verification, local copies, second machine, USB,
  recording, and link check;
- one hour before: open PPTX/PDF, dashboard, MLflow, terminal, screenshots, and
  Q&A;
- ten minutes before: airplane-mode test, audio/display check, notifications
  off, and first/last speaker ready;
- network failure: use local dashboard and screenshots;
- MLflow failure: use comparison/promotion JSON and screenshot;
- dashboard failure: use static HTML and screenshot;
- primary laptop failure: use the second machine/USB and backup recording;
- named person who decides to switch to fallback.

- [ ] **Step 3: Create evidence-linked Q&A**

Prepare concise answers for:

- why PR-AUC rather than accuracy;
- leakage and chronological splitting;
- why model validation and calibration are separate;
- why false negative cost is 100 and false positive cost is 1;
- DVC's actual scope and absent real-data remote;
- label delay and pending performance;
- PSI versus KS;
- natural versus injected drift;
- why scheduled verification may train ephemerally but cannot replace a
  production model;
- MLflow governance versus local runtime bundle;
- human approval before retraining;
- fairness limitations from absent subgroup attributes;
- what would change for production.

Every answer ends with an exact evidence path or slide number.

- [ ] **Step 4: Cross-review and commit**

Run:

```bash
rg -n "metrics_valid|valid\\.csv|threshold 0\\.98|13 tests|run make pipeline live|\
automatic retrain|DVC-tracked real data" \
  docs/DEMO_SCRIPT.md docs/DEMO_FALLBACK.md docs/DEMO_QA.md
git diff --check
```

Expected: no stale match. Then:

```bash
git add docs/DEMO_SCRIPT.md docs/DEMO_FALLBACK.md docs/DEMO_QA.md
git commit -m "docs: finalize demo and question handling"
```

### Task 5: Run the Four-Reflection Human Workflow

**Files:**
- Modify: `docs/REFLECTION_TEMPLATE.md`
- Create privately: reflection ledger, four DOCX/PDF pairs, and signoffs listed above

**Interfaces:**
- Consumes: verified personal contribution evidence and human-authored drafts
- Produces: four distinct, authentic, private 500–800-word reflections

- [ ] **Step 1: Make the public worksheet neutral**

Remove the illustrative first-person fragment from
`docs/REFLECTION_TEMPLATE.md`. Keep only neutral prompts for:

- specific files, commits, decisions, reviews, and pairing;
- one MLOps concept genuinely learned;
- one real challenge and response;
- collaboration and feedback;
- one limitation and future improvement;
- responsible-AI use and personal verification.

- [ ] **Step 2: Build the private contribution ledger**

Each author records their own verified:

- commits and pull requests;
- files/stages implemented;
- reviews and pairing sessions;
- decisions they can explain;
- challenge they personally encountered;
- evidence checked after AI/tool assistance.

Use repository history to verify claims:

```bash
git log --stat --oneline
git shortlog -sne HEAD
```

Names and events come from the team; the agent must not infer ownership from
role labels alone.

- [ ] **Step 3: Assign four distinct anchor experiences**

Before drafting, reviewers ensure the four authors do not center on the same
experience. Suggested distinct anchors are validation/DVC, MLflow/calibration,
monitoring/label delay, and CI/evidence/report integration, but use only anchors
that reflect actual work.

- [ ] **Step 4: Authors write first drafts**

Each human writes 550–700 words in their own voice. The agent may provide the
worksheet and later comments, but must not provide first-person sentences,
sample paragraphs, emotional language, or invented events to paraphrase.

- [ ] **Step 5: Run factual and rubric review**

For each draft:

1. a peer checks contribution claims against the ledger and Git;
2. another reviewer checks technical accuracy and rubric coverage;
3. the author resolves every comment personally;
4. the author adds an honest AI-use statement and how outputs were verified.

Allowed assistance: spelling, grammar, clarity comments, word count,
rubric-coverage flags, factual-consistency flags, and duplicate-phrase
detection. Assistance must not homogenize the four voices.

- [ ] **Step 6: Check word counts and duplicated phrasing**

Run:

```bash
for file in submission-private/reflections/*_Reflection.docx; do
  printf '%s: ' "$file"
  pandoc "$file" -t plain | wc -w
done
```

Expected: each count is 500–800. Extract each draft to text and compare
repeated eight-word sequences:

```bash
mkdir -p tmp/reflection-text
for file in submission-private/reflections/*_Reflection.docx; do
  base="$(basename "$file" .docx)"
  pandoc "$file" -t plain -o "tmp/reflection-text/$base.txt"
done
python - <<'PY'
from itertools import combinations
from pathlib import Path
import re

documents = {}
for path in sorted(Path("tmp/reflection-text").glob("*.txt")):
    words = re.findall(r"[a-z0-9']+", path.read_text().lower())
    documents[path.name] = {
        tuple(words[index:index + 8])
        for index in range(max(0, len(words) - 7))
    }
for (left_name, left), (right_name, right) in combinations(documents.items(), 2):
    shared = sorted(left & right)
    print(left_name, right_name, len(shared))
    for phrase in shared[:20]:
        print("  ", " ".join(phrase))
PY
```

Reviewers inspect every non-trivial overlap and authors revise only where
phrasing is not genuinely their own.

- [ ] **Step 7: Export and inspect private PDFs**

Use the `documents:documents` and `pdf:pdf` render workflows for each reflection.
Render exact files:

```bash
DOC_SKILL_DIR="/Users/goaltosuceed/.codex/plugins/cache/openai-primary-runtime/documents/26.727.11326/skills/documents"
BUNDLED_PYTHON="/Users/goaltosuceed/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
for file in submission-private/reflections/*_Reflection.docx; do
  base="$(basename "$file" .docx)"
  out="tmp/reflection-render/$base"
  mkdir -p "$out"
  "$BUNDLED_PYTHON" "$DOC_SKILL_DIR/render_docx.py" "$file" \
    --output_dir "$out" --emit_pdf
  cp "$out/$base.pdf" "submission-private/reflections/$base.pdf"
done
```

Inspect every DOCX render and PDF page for author identity, word-count
compliance, clipping, comments, tracked changes, private metadata, and
unintended prompts.

- [ ] **Step 8: Sign the review ledger**

`REVIEW_SIGNOFFS.md` records, for each role file:

- author confirmation of genuine experience and wording;
- factual reviewer;
- rubric/technical reviewer;
- final word count;
- AI-use disclosure present;
- privacy and visual QA complete.

Do not commit any private file.

- [ ] **Step 9: Commit only the neutral public worksheet**

```bash
git add docs/REFLECTION_TEMPLATE.md
git commit -m "docs: make reflection worksheet neutral"
```

### Task 6: Complete Three Rehearsals

**Files:**
- Create privately: `submission-private/rehearsals/REHEARSAL_LOG.md`
- Create privately: `submission-private/demo/Fraud_MLOps_Demo_Fallback.mp4`
- Modify publicly if needed: deck and demo documents

**Interfaces:**
- Consumes: final deck, script, fallback, Q&A, and confirmed speakers
- Produces: factual, timed, recorded presentation with reliable handoffs

- [ ] **Step 1: Rehearsal 1 — factual accuracy**

Run all ten slides without optimizing time. A reviewer checks every number,
model/run/version, threshold, status, reason code, command, and architecture
claim against evidence. Record corrections and owner.

- [ ] **Step 2: Rehearsal 2 — timing and transitions**

Time every slide and handoff. Target:

```text
Slide 1  0:40
Slide 2  0:40
Slide 3  0:50
Slide 4  0:50
Slide 5  1:00
Slide 6  1:10
Slide 7  1:20
Slide 8  1:10
Slide 9  1:00
Slide 10 0:50
Total    9:30
```

Shorten spoken wording before removing evidence or shrinking text.

- [ ] **Step 3: Rehearsal 3 — recorded dress rehearsal**

Record the full presentation plus two Q&A questions per speaker. Run once with
network disabled and exercise one fallback path. Save the private recording at
the specified MP4 path.

- [ ] **Step 4: Acceptance timing**

The final uninterrupted run must be between 9:15 and 9:40, with no speaker
reading code, no live training wait, no unsupported claim, and no transition
longer than five seconds.

- [ ] **Step 5: Fix and re-verify public artifacts**

If rehearsal corrections change the PPTX, PDF, screenshots, or docs, re-run
slide overflow/render/PDF inspection, evidence consistency, hashes, and
cross-review before committing.

### Task 7: Assemble and Inspect the Final University Bundle

**Files:**
- Create locally: `dist/Fraud_MLOps_Repository_v1.0.0.zip`
- Create privately: `submission-private/Fraud_MLOps_University_Submission_v1.0.0.zip`
- Modify: `submission/MANIFEST.md`
- Modify: `submission/SHA256SUMS.txt`
- Update: `docs/delivery/95-plus-checklist.md`

**Interfaces:**
- Consumes: final public commit and four signed-off private reflection PDFs
- Produces: tested public source archive and private university submission archive

- [ ] **Step 1: Run the complete technical/content gate**

```bash
SOURCE_COMMIT="$(git rev-parse HEAD)" make verify
scripts/verify_dvc_clean_clone.sh
pytest -q tests/test_evidence_consistency.py
git diff --check
```

Expected: all commands exit 0 and Pytest has zero skips.

- [ ] **Step 2: Recompute public artifact hashes**

```bash
cd submission
shasum -a 256 \
  Fraud_MLOps_Technical_Report.docx \
  Fraud_MLOps_Technical_Report.pdf \
  Fraud_MLOps_Presentation.pptx \
  Fraud_MLOps_Presentation.pdf \
  > SHA256SUMS.txt
shasum -a 256 -c SHA256SUMS.txt
cd ..
```

- [ ] **Step 3: Prove public privacy**

```bash
git ls-files | rg '(^submission-private/|Reflection\\.(docx|pdf)$|data\\.zip$|\
data/raw/creditcard\\.csv$|mlflow\\.db$)' && exit 1 || true
git status --short
```

Expected: no private/raw/database file is tracked.

- [ ] **Step 4: Create and test the repository archive**

```bash
mkdir -p dist
git archive --format=zip --prefix=fraud-mlops-v1.0.0/ \
  -o dist/Fraud_MLOps_Repository_v1.0.0.zip HEAD
unzip -t dist/Fraud_MLOps_Repository_v1.0.0.zip
unzip -l dist/Fraud_MLOps_Repository_v1.0.0.zip | \
  rg 'data\\.zip|creditcard\\.csv|submission-private|Reflection' && exit 1 || true
```

Expected: archive integrity passes and prohibited files are absent.

- [ ] **Step 5: Create the private university archive**

Create a clean temporary stage containing the report PDF, presentation
PPTX/PDF, repository source archive, and exactly four reflection PDFs:

```bash
repo_root="$(pwd)"
stage="$(mktemp -d)"
cp submission/Fraud_MLOps_Technical_Report.pdf "$stage/"
cp submission/Fraud_MLOps_Presentation.pptx "$stage/"
cp submission/Fraud_MLOps_Presentation.pdf "$stage/"
cp dist/Fraud_MLOps_Repository_v1.0.0.zip "$stage/"
cp submission-private/reflections/01_Data_Engineering_Reflection.pdf "$stage/"
cp submission-private/reflections/02_Modelling_Reflection.pdf "$stage/"
cp submission-private/reflections/03_Monitoring_Reflection.pdf "$stage/"
cp submission-private/reflections/04_Platform_Documentation_Reflection.pdf "$stage/"
(
  cd "$stage"
  zip -q "$repo_root/submission-private/Fraud_MLOps_University_Submission_v1.0.0.zip" ./*
)
rm -rf "$stage"
unzip -t submission-private/Fraud_MLOps_University_Submission_v1.0.0.zip
unzip -l submission-private/Fraud_MLOps_University_Submission_v1.0.0.zip
```

Expected: integrity passes and exactly four reflection PDFs are listed.

- [ ] **Step 6: Final human inspection**

Open every file directly from the private ZIP. Confirm identities, report word
count, deck slide count, four reflection word counts, repository link/archive,
hashes, and absence of draft/private cross-contamination.

- [ ] **Step 7: Close delivery gates and commit public records**

Record actual report word count, slide count, final rehearsal time, four private
word counts, final commit, verification results, and reviewers in the delivery
checklist and manifest.

```bash
git add submission/MANIFEST.md submission/SHA256SUMS.txt \
  docs/delivery/95-plus-checklist.md
git commit -m "docs: finalize submission verification record"
```

- [ ] **Step 8: Stop for explicit submission/publication approval**

Present the exact public and private file lists. Do not upload the university
ZIP, push, tag, create a release, or deploy anything until the user authorizes
the specific action.

## Presentation and Reflection Acceptance Gate

- Deck has exactly ten slides and runs 9:15–9:40.
- Every metric and model identifier traces to frozen evidence.
- Operating evidence is primary and default threshold is clearly comparison-only.
- Every speaker covers genuine work and confirms their notes.
- PPTX and PDF pass slide-by-slide visual inspection and overflow tests.
- Demo succeeds offline with screenshots, local files, and a private recording.
- Q&A covers imbalance, leakage, calibration, DVC scope, delayed labels, drift, approval, and limitations.
- Three rehearsals are logged, including a recorded dress rehearsal.
- Four private reflections are authentic, distinct, 500–800 words, and signed off.
- No reflection, private recording, raw data, database, or `data.zip` enters the public repository/archive.
- Final ZIPs pass integrity and open-file inspection.
- No external action occurs without explicit approval.
