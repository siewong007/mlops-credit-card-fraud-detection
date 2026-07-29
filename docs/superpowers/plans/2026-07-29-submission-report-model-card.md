# Submission Evidence, Report, and Model Card Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze one authoritative real-data run and turn it into a consistent rubric evidence matrix, model card, README, diagrams, and visually verified 4,000–5,000-word technical report.

**Architecture:** Quantitative claims flow from the frozen JSON evidence package into all human-facing documents. Documentation is reconciled by replacement rather than expansion, and the final DOCX/PDF is rendered page-by-page before acceptance.

**Tech Stack:** Markdown, generated JSON/PNG evidence, Pytest, Pandoc, the `documents:documents` skill, the `pdf:pdf` skill, DOCX, PDF, Poppler, and Git.

## Global Constraints

- Begin only after the core and reproducibility acceptance gates pass.
- Freeze all technical code, tests, parameters, orchestration, dependencies, Docker, and workflows before the authoritative run.
- Use the real ULB/OpenML dataset: 284,807 rows and 492 fraud cases.
- The run manifest source commit is the technical freeze commit, not the later evidence/document commit.
- Any post-freeze change to `src/`, `tests/`, `params.yaml`, `Makefile`, `dvc.yaml`, dependencies, Docker, or workflows invalidates the quantitative evidence and requires a new authoritative run.
- The operating-threshold view is primary; threshold 0.5 is comparison-only.
- Do not claim production validation, real-time serving, automatic retraining, a DVC data remote, fairness measurement without subgroup data, or a guaranteed 97/100 mark.
- The report must contain 4,000–5,000 rendered words; target 4,300–4,600 before final packaging.
- Remove draft notices, approval instructions, group-action prompts, stale counts, stale thresholds, and unsupported claims.
- Final report output is both DOCX and PDF.
- The document task must use the `documents:documents` skill and its render-to-PNG gate.
- The final PDF task must use the `pdf:pdf` skill and inspect every page.
- Team/member names, university metadata, contribution claims, and AI-use declarations must be supplied and confirmed by the humans; never invent them.
- Do not expose private reflections, raw data, `mlflow.db`, secrets, or the untracked `data.zip`.
- Do not tag, push, publish a release, or submit without explicit user approval.

---

## File Structure

**Create**

- `docs/RUBRIC_EVIDENCE.md`
- `docs/MODEL_CARD.md`
- `submission/Fraud_MLOps_Technical_Report.docx`
- `submission/Fraud_MLOps_Technical_Report.pdf`
- `submission/MANIFEST.md`
- `submission/RELEASE_NOTES.md`
- `submission/SHA256SUMS.txt`

**Modify**

- `README.md`
- `PLAN.md`
- `notebooks/README.md`
- `docs/workflow_diagrams.md`
- `docs/TECHNICAL_REPORT.md`
- `tests/test_evidence_consistency.py`

**Frozen generated evidence**

- `reports/validation_report.json`
- `reports/experiment_comparison.json`
- `reports/promotion_record.json`
- `reports/operating_point.json`
- `reports/metrics_default.json`
- `reports/metrics_operating.json`
- `reports/figures/confusion_matrix_default.png`
- `reports/figures/confusion_matrix_operating.png`
- `reports/drift_summary.json`
- `reports/trigger_decisions.json`
- `reports/trigger_log.md`
- `reports/run_manifest.json`

### Task 1: Freeze the Authoritative Real-Data Evidence

**Files:**
- Generate: the frozen evidence list above
- Update: `docs/delivery/95-plus-checklist.md`

**Interfaces:**
- Consumes: a clean technical freeze commit
- Produces: one internally consistent real-data evidence commit

- [ ] **Step 1: Prove the technical tree is ready to freeze**

Run:

```bash
make verify
scripts/verify_dvc_clean_clone.sh
source_commit="$(git rev-parse HEAD)"
docker build --pull --build-arg SOURCE_COMMIT="$source_commit" \
  --tag fraud-mlops:verify .
docker run --rm fraud-mlops:verify
git diff --check
git status --short
```

Expected: every verification exits 0; Pytest has zero skips; only deliberately
ignored files and the preserved pre-existing local files remain untracked.

- [ ] **Step 2: Commit any reviewed technical changes**

If Step 1 revealed a necessary technical fix, implement it through its owning
plan, repeat Step 1, and commit. Record:

```bash
git rev-parse HEAD
```

as the technical freeze commit in `docs/delivery/95-plus-checklist.md`.

- [ ] **Step 3: Fetch and verify the real source**

Run:

```bash
make fetch-data
python - <<'PY'
import json
import pandas as pd

df = pd.read_csv("data/raw/creditcard.csv")
provenance = json.load(open("data/raw/PROVENANCE.json"))
assert len(df) == 284_807
assert int(df["Class"].sum()) == 492
assert provenance["source"].startswith("REAL")
print(provenance)
PY
```

Expected: 284,807 rows, 492 fraud cases, and real-source provenance.

- [ ] **Step 4: Run the authoritative evidence path once**

Run:

```bash
source_commit="$(git rev-parse HEAD)"
SOURCE_COMMIT="$source_commit" make verify
pytest -q tests/test_evidence_consistency.py
```

Expected: the pipeline, dashboard, full zero-skip suite, and consistency test
all pass.

- [ ] **Step 5: Audit the frozen contracts**

Run:

```bash
python - <<'PY'
import json
from pathlib import Path

required = [
    "validation_report.json",
    "experiment_comparison.json",
    "promotion_record.json",
    "operating_point.json",
    "metrics_default.json",
    "metrics_operating.json",
    "drift_summary.json",
    "trigger_decisions.json",
    "run_manifest.json",
]
for name in required:
    payload = json.loads((Path("reports") / name).read_text())
    print(name, "OK", type(payload).__name__)
assert json.load(open("reports/run_manifest.json"))["data"]["source"].startswith("REAL")
PY
```

Expected: every JSON parses and the manifest identifies real data.

- [ ] **Step 6: Commit the authoritative evidence**

Stage only portable evidence; do not stage raw data, models, `mlflow.db`, or
optional generated HTML:

```bash
git add reports/*.json reports/drift/*.json reports/trigger_log.md reports/figures/*.png \
  docs/delivery/95-plus-checklist.md
git commit -m "evidence: freeze authoritative real-data run"
```

Record the new evidence commit separately from the technical freeze commit.

### Task 2: Build the Rubric Evidence Matrix

**Files:**
- Create: `docs/RUBRIC_EVIDENCE.md`

**Interfaces:**
- Consumes: frozen JSON, source locations, verification commands, public CI URL when available
- Produces: criterion → claim → implementation → evidence → verification → limitation mapping

- [ ] **Step 1: Create the metadata block**

Start with the title `# Rubric evidence matrix`, followed by a sentence stating
that 97/100 is an internal quality target and the assessor determines the final
mark. Add an `Evidence identity`/`Authoritative value` table containing the
actual technical freeze commit, evidence commit, dataset/source, dataset
fingerprint, parameter fingerprint, Python runtime, and successful final CI
run URL.

Generate the first six values instead of typing them:

```bash
python - <<'PY'
import json
import subprocess

manifest = json.load(open("reports/run_manifest.json"))
evidence_commit = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
rows = [
    ("Technical freeze commit", manifest["source_commit"]),
    ("Evidence commit", evidence_commit),
    ("Dataset", manifest["data"]["source"]),
    ("Dataset fingerprint", manifest["data"]["fingerprint_sha256"]),
    ("Parameter fingerprint", manifest["parameters"]["fingerprint_sha256"]),
    ("Python runtime", manifest["python_version"]),
]
for label, value in rows:
    print(f"| {label} | `{value}` |")
PY
```

The final CI URL requires explicit push approval and a successful run on the
evidence commit. If approval is not available, stop this task before calling
the matrix final; do not leave an empty cell or fabricate a URL.

- [ ] **Step 2: Add all six weighted criteria**

Use one section per rubric criterion:

```markdown
## 1. Workflow reorganisation and MLOps architecture — 20%
## 2. Appropriate use of MLOps tools — 20%
## 3. Systematic testing, monitoring, and retraining — 20%
## 4. Technical implementation quality — 15%
## 5. Reproducibility and documentation — 15%
## 6. Presentation/demo and individual reflection — 10%
```

Within each section, use this exact table:

```markdown
| Claim | Implementation | Generated evidence | Verification | Limitation |
|---|---|---|---|---|
```

- [ ] **Step 3: Map all eight operational requirements**

The matrix must explicitly cover:

1. regular transaction batches;
2. fraud-pattern change;
3. severe class imbalance;
4. asymmetric false-negative/false-positive costs;
5. reproducibility;
6. data-quality checks;
7. experiment tracking and promotion;
8. justified retraining-review decisions.

Every row must name exact paths and a runnable command. Examples of valid
evidence references are `src/validate.py`,
`reports/validation_report.json`, and `make verify`; “see repository” is not
sufficient.

- [ ] **Step 4: Record limitations honestly**

Include the two-day data window, anonymized PCA features, absent subgroup
attributes, injected drift in `prod_3`, delayed labels, local MLflow backend,
no DVC remote for the real CSV, batch-only inference, and human approval before
retraining.

- [ ] **Step 5: Run link and stale-name checks**

Run:

```bash
rg -n "metrics_valid|valid\\.csv|automatic retrain|production validated|DVC remote stores" \
  docs/RUBRIC_EVIDENCE.md
python - <<'PY'
from pathlib import Path
import re

text = Path("docs/RUBRIC_EVIDENCE.md").read_text()
paths = re.findall(r"`((?:src|tests|reports|docs)/[^`]+)`", text)
missing = [path for path in paths if not Path(path).exists()]
assert not missing, missing
print(f"checked {len(paths)} evidence links")
PY
```

Expected: no stale-claim match and no missing local path.

- [ ] **Step 6: Commit**

```bash
git add docs/RUBRIC_EVIDENCE.md
git commit -m "docs: add rubric evidence matrix"
```

### Task 3: Create the Model Card

**Files:**
- Create: `docs/MODEL_CARD.md`

**Interfaces:**
- Consumes: promotion, operating, default, monitoring, trigger, and manifest JSON
- Produces: an honest model-governance document for the promoted batch model

- [ ] **Step 1: Write the required section structure**

```markdown
# Model card — fraud detector

## Model and version summary
## Intended use
## Excluded uses
## Data and chronological slices
## Candidate comparison and promotion
## Threshold calibration and business costs
## Operating metrics
## Monitoring and label availability
## Retraining-review and human approval
## Ethical risks and customer impact
## Limitations
## Reproducibility
## Upgrade path
```

- [ ] **Step 2: Populate governance identity from JSON**

State the promoted model ID, candidate name, MLflow run ID, registry version,
selection metric, source fingerprint, parameter fingerprint, and calibration
fingerprint exactly as generated.

- [ ] **Step 3: Explain permitted and prohibited use**

Permitted: educational batch scoring and MLOps workflow demonstration on the
ULB schema. Prohibited: autonomous card blocking, real-time production
deployment, use on unrelated schemas, fairness claims, or silent model
replacement.

- [ ] **Step 4: State performance and decision policy**

Show operating metrics first and threshold 0.5 in a labeled comparison table.
Explain false-negative cost 100 versus false-positive cost 1, delayed-label
behavior, PSI/KS roles, stable reason codes, warning/retrain thresholds, and
human approval.

- [ ] **Step 5: Run prohibited-claim and evidence checks**

Run:

```bash
rg -n "production-ready|production validated|real-time|automatically retrains|fair across" \
  docs/MODEL_CARD.md
python -m pytest -q tests/test_evidence_consistency.py
```

Expected: the search has no unsupported claim and consistency passes.

- [ ] **Step 6: Commit**

```bash
git add docs/MODEL_CARD.md
git commit -m "docs: add promoted model card"
```

### Task 4: Reconcile README, Diagrams, and Project Records

**Files:**
- Modify: `README.md`
- Modify: `PLAN.md`
- Modify: `notebooks/README.md`
- Modify: `docs/workflow_diagrams.md`
- Modify: `tests/test_evidence_consistency.py`

**Interfaces:**
- Consumes: finalized command names, evidence paths, model/threshold metrics
- Produces: one consistent public project narrative with machine-checked evidence lines

- [ ] **Step 1: Give README three explicit execution paths**

The top-level usage section must include:

```bash
# Quick deterministic synthetic run in a clean clone
make verify

# Authoritative real-data run
make fetch-data
SOURCE_COMMIT="$(git rev-parse HEAD)" make verify

# Isolated DVC lineage verification
make verify-dvc

# Immutable Docker verification
docker build --build-arg SOURCE_COMMIT="$(git rev-parse HEAD)" \
  --tag fraud-mlops:verify .
docker run --rm fraud-mlops:verify
```

State Python 3.13.9 before the commands. Explain that the synthetic path must
start from a clone with no raw CSV. `make verify` preserves and uses an existing
real CSV; the DVC command always isolates its synthetic generator in a
disposable clone.

- [ ] **Step 2: Add an exact tool-role table**

Use:

| Tool | Actual role |
|---|---|
| Make | Authoritative dependency-ordered orchestration |
| Pandera | Executable labeled and inference data contracts |
| MLflow | Candidate tracking, model registry, and governance identity |
| Local joblib bundle | Batch runtime model/scaler/threshold contract |
| DVC | Disposable synthetic lineage, parameters, hashes, metrics, plots, and lock |
| Docker | Immutable Python 3.13.9 verification runtime |
| Pytest | Unit, integration, and evidence-consistency gates |
| GitHub Actions | Automated synthetic, DVC, Docker, and monitoring verification |
| Evidently | Optional human-readable drift layer; native PSI/KS JSON remains required |

- [ ] **Step 3: Replace the final workflow diagrams**

`docs/workflow_diagrams.md` must show:

- validation before ingestion;
- six chronological slices;
- model selection on `model_valid`;
- threshold selection/evaluation on `calibration`;
- three later production batches;
- `ok`, `warning`, and `retrain` structured statuses;
- explicit human review and approval before retraining;
- actual Make, DVC, Docker, MLflow, local bundle, and CI roles.

The `retrain` arrow must go to a human approval node, not directly to training.

- [ ] **Step 4: Reconcile project records**

Convert `PLAN.md` from future-tense claims to a completed project record with
actual evidence paths and limitations. Remove the unresolved “Group action”
instruction from `notebooks/README.md` and identify the baseline notebook as
historical reference only.

- [ ] **Step 5: Add machine-checked visible evidence lines**

Generate the exact “Authoritative evidence snapshot” rows:

```bash
python - <<'PY'
import json

promotion = json.load(open("reports/promotion_record.json"))
point = json.load(open("reports/operating_point.json"))
metrics = json.load(open("reports/metrics_operating.json"))
trigger = json.load(open("reports/trigger_decisions.json"))
rows = [
    ("Promoted model ID", promotion["promoted_model_id"]),
    ("MLflow run ID", promotion["mlflow_run_id"]),
    ("Registry version", promotion["registered_model_version"]),
    ("Operating threshold", f"{point['threshold']:.2f}"),
    ("Operating PR-AUC", f"{metrics['pr_auc']:.4f}"),
    ("Operating recall", f"{metrics['recall']:.4f}"),
    ("Overall monitoring status", trigger["overall_status"]),
]
for label, value in rows:
    print(f"| {label} | `{value}` |")
PY
```

Paste that unedited output under the heading “Authoritative evidence snapshot”
in both `README.md` and `docs/TECHNICAL_REPORT.md`. Then append this exact test:

```python
def test_public_documents_match_authoritative_evidence():
    promotion = _load("promotion_record.json")
    point = _load("operating_point.json")
    metrics = _load("metrics_operating.json")
    trigger = _load("trigger_decisions.json")
    expected = [
        f"| Promoted model ID | `{promotion['promoted_model_id']}` |",
        f"| MLflow run ID | `{promotion['mlflow_run_id']}` |",
        f"| Registry version | `{promotion['registered_model_version']}` |",
        f"| Operating threshold | `{point['threshold']:.2f}` |",
        f"| Operating PR-AUC | `{metrics['pr_auc']:.4f}` |",
        f"| Operating recall | `{metrics['recall']:.4f}` |",
        f"| Overall monitoring status | `{trigger['overall_status']}` |",
    ]
    for path in (ROOT / "README.md", ROOT / "docs" / "TECHNICAL_REPORT.md"):
        text = path.read_text()
        for line in expected:
            assert line in text, f"{path} missing evidence line: {line}"
```

- [ ] **Step 6: Scan every public document for contradictions**

Run:

```bash
rg -n "metrics_valid|confusion_matrix\\.png|dvc init|dvc add|python:3\\.13-slim|\
loop back to Train|validation fifth|13 tests|Draft status|Group action|automatic retrain|\
train_frac.*0\\.5.*valid_frac.*0\\.2" \
  README.md PLAN.md notebooks/README.md docs/*.md
```

Expected: no unresolved match. A historical comparison may retain a phrase
only when the surrounding sentence explicitly identifies it as the old
workflow.

- [ ] **Step 7: Run consistency and commit**

Run:

```bash
pytest -q tests/test_evidence_consistency.py
git diff --check
```

Then:

```bash
git add README.md PLAN.md notebooks/README.md docs/workflow_diagrams.md \
  docs/TECHNICAL_REPORT.md tests/test_evidence_consistency.py
git commit -m "docs: reconcile public evidence and architecture"
```

### Task 5: Rewrite the Technical Report by Replacement

**Files:**
- Modify: `docs/TECHNICAL_REPORT.md`

**Interfaces:**
- Consumes: frozen evidence, evidence matrix, model card, final diagrams
- Produces: a self-contained 4,300–4,600-word Markdown report source

- [ ] **Step 1: Remove authoring notices and count the clean source**

Delete the draft/approval notice and any instructions to group members. Run:

```bash
pandoc docs/TECHNICAL_REPORT.md -t plain | wc -w
```

Record the starting count; do not solve stale prose by appending.

- [ ] **Step 2: Rebalance the report to this body budget**

| Section | Target words |
|---|---:|
| Introduction | 250 |
| Data, constraints, and six chronological slices | 450 |
| Original workflow and eight requirements | 400 |
| Architecture, contracts, and tool choices | 600 |
| Experiments, promotion, and registry | 450 |
| Validation, testing, and reproducibility | 550 |
| Calibration and default/operating comparison | 500 |
| Inference, monitoring, and review decision | 650 |
| Limitations, responsible AI, and future work | 400 |
| Conclusion | 200 |

Use the budget to replace stale material. Keep the total rendered source within
4,300–4,600 before front matter.

- [ ] **Step 3: Apply mandatory factual corrections**

The report must:

- use the 50/10/10/10/10/10 chronology;
- show validation as the first required gate;
- distinguish model validation from threshold calibration;
- identify the winning candidate run and registry version;
- show operating metrics/matrix first and 0.5 only as comparison;
- state the actual final test count and zero skips;
- describe DVC without a real-data remote claim;
- distinguish automatic candidate promotion from approval-gated retraining;
- correct the old synthetic-threshold claim;
- show structured reason codes and label status;
- state real versus injected drift precisely;
- acknowledge the baseline notebook, sources, limitations, and responsible AI
  assistance in past tense.

- [ ] **Step 4: Use frozen figures and readable captions**

Include, at minimum:

- before/after architecture;
- experiment comparison;
- threshold trade-off;
- default confusion matrix;
- operating confusion matrix;
- calibration PR curve;
- healthy versus degraded drift evidence;
- structured decision summary.

Captions must name the slice and threshold. Do not use a code-like filename as
the visible caption.

- [ ] **Step 5: Enforce report/evidence consistency**

Run:

```bash
pytest -q tests/test_evidence_consistency.py
pandoc docs/TECHNICAL_REPORT.md -t plain | wc -w
rg -n "Draft status|approval|metrics_valid|valid\\.csv|13 tests|automatic retrain" \
  docs/TECHNICAL_REPORT.md
```

Expected: consistency passes, word count is 4,300–4,600, and no unresolved
match remains.

- [ ] **Step 6: Cross-review and commit**

One team member who did not lead report assembly must verify each evidence
snapshot line against JSON and sign the delivery checklist. Then:

```bash
git add docs/TECHNICAL_REPORT.md docs/delivery/95-plus-checklist.md
git commit -m "docs: finalize evidence-led technical report"
```

### Task 6: Produce and Visually Verify DOCX and PDF

**Files:**
- Create: `submission/Fraud_MLOps_Technical_Report.docx`
- Create: `submission/Fraud_MLOps_Technical_Report.pdf`

**Interfaces:**
- Consumes: final Markdown report, confirmed human metadata, final figures
- Produces: polished DOCX and PDF with matching content and 4,000–5,000 rendered words

- [ ] **Step 1: Load the document runtime and required skill instructions**

Call the workspace dependency loader. Read the complete
`documents:documents` skill, `references/design_presets.md`,
`references/header_templates.md`, `tasks/accessibility_a11y.md`,
`tasks/privacy_scrub_metadata.md`, `tasks/clean_tracked_changes.md`, and
`tasks/comments_manage.md`.

Use `standard_business_brief` as the resolved design preset, adapted for a
formal academic technical report. Obtain confirmed title-page names,
student IDs, module, instructor, institution, and submission date from the
team before generating the file.

- [ ] **Step 2: Build the DOCX with real document structure**

Create real Word heading styles, numbering, captions, cross-references, table
geometry, header/footer, and page numbers. Use a restrained formal palette,
US Letter portrait, 1-inch margins, readable body text, and sharp original PNG
figures. Do not fake headings, bullets, captions, or page numbers with plain
text.

- [ ] **Step 3: Scrub and structurally audit the DOCX**

Use the document-skill scripts:

```bash
DOC_SKILL_DIR="/Users/goaltosuceed/.codex/plugins/cache/openai-primary-runtime/documents/26.727.11326/skills/documents"
BUNDLED_PYTHON="/Users/goaltosuceed/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
mkdir -p tmp/report-audit
"$BUNDLED_PYTHON" "$DOC_SKILL_DIR/scripts/accept_tracked_changes.py" \
  submission/Fraud_MLOps_Technical_Report.docx \
  --mode accept \
  --out tmp/report-audit/Fraud_MLOps_Technical_Report.accepted.docx
"$BUNDLED_PYTHON" "$DOC_SKILL_DIR/scripts/comments_strip.py" \
  tmp/report-audit/Fraud_MLOps_Technical_Report.accepted.docx \
  --out tmp/report-audit/Fraud_MLOps_Technical_Report.clean.docx
"$BUNDLED_PYTHON" "$DOC_SKILL_DIR/scripts/privacy_scrub.py" \
  tmp/report-audit/Fraud_MLOps_Technical_Report.clean.docx \
  --out tmp/report-audit/Fraud_MLOps_Technical_Report.scrubbed.docx
mv tmp/report-audit/Fraud_MLOps_Technical_Report.scrubbed.docx \
  submission/Fraud_MLOps_Technical_Report.docx
"$BUNDLED_PYTHON" "$DOC_SKILL_DIR/scripts/accept_tracked_changes.py" \
  submission/Fraud_MLOps_Technical_Report.docx \
  --mode report
"$BUNDLED_PYTHON" "$DOC_SKILL_DIR/scripts/comments_extract.py" \
  submission/Fraud_MLOps_Technical_Report.docx \
  --out tmp/report-audit/comments.json
"$BUNDLED_PYTHON" "$DOC_SKILL_DIR/scripts/a11y_audit.py" \
  submission/Fraud_MLOps_Technical_Report.docx \
  --out_json tmp/report-audit/a11y.json
```

Expected: tracked-change report is zero, `comments.json` contains no comments,
and `a11y.json` contains no high-severity issue. Repair meaningful image alt
text, heading hierarchy, table headers, or hyperlink wording at the source,
then rerun the complete audit. Final output must contain no comments, tracked
changes, authoring prompts, or unintended personal metadata.

- [ ] **Step 4: Render the DOCX to page images and PDF**

Run:

```bash
DOC_SKILL_DIR="/Users/goaltosuceed/.codex/plugins/cache/openai-primary-runtime/documents/26.727.11326/skills/documents"
BUNDLED_PYTHON="/Users/goaltosuceed/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
rm -rf tmp/report-render
"$BUNDLED_PYTHON" "$DOC_SKILL_DIR/render_docx.py" \
  submission/Fraud_MLOps_Technical_Report.docx \
  --output_dir tmp/report-render \
  --emit_pdf
cp tmp/report-render/Fraud_MLOps_Technical_Report.pdf \
  submission/Fraud_MLOps_Technical_Report.pdf
```

- [ ] **Step 5: Inspect every DOCX render page**

Open every `tmp/report-render/page-*.png` at 100% zoom. Reject and iterate on:

- clipping, overlap, broken glyphs, or accidental blank space;
- split captions or orphaned headings;
- unreadable figures or labels;
- confusing default/operating threshold labels;
- broken table geometry;
- inconsistent headers, footers, or page numbers;
- non-working TOC, links, or cross-references.

Re-render after every correction until all pages pass.

- [ ] **Step 6: Independently inspect the PDF**

Use the `pdf:pdf` skill. Run:

```bash
pdfinfo submission/Fraud_MLOps_Technical_Report.pdf
rm -rf tmp/report-pdf
mkdir -p tmp/report-pdf
pdftoppm -png submission/Fraud_MLOps_Technical_Report.pdf \
  tmp/report-pdf/page
pandoc submission/Fraud_MLOps_Technical_Report.docx -t plain | wc -w
```

Inspect every PDF page. Expected: PDF and DOCX render the same content, all
pages are clean, and rendered word count is 4,000–5,000.

- [ ] **Step 7: Commit final report artifacts**

```bash
git add submission/Fraud_MLOps_Technical_Report.docx \
  submission/Fraud_MLOps_Technical_Report.pdf
git commit -m "docs: add visually verified technical report"
```

Do not commit `tmp/report-render` or `tmp/report-pdf`.

### Task 7: Build the Public Submission Manifest

Execute this task after Task 3 of the presentation/reflections plan has
produced the final PPTX and PDF.

**Files:**
- Create: `submission/MANIFEST.md`
- Create: `submission/RELEASE_NOTES.md`
- Create: `submission/SHA256SUMS.txt`

**Interfaces:**
- Consumes: public repository files and final report artifacts
- Produces: exact file inventory and integrity hashes; no private bundle

- [ ] **Step 1: Create the manifest**

`submission/MANIFEST.md` must identify:

- repository URL and final commit;
- technical freeze and evidence freeze commits;
- report DOCX/PDF;
- presentation PPTX/PDF added by the presentation plan;
- evidence matrix and model card;
- run-manifest data/parameter fingerprints;
- successful CI and Pages URLs after push approval;
- four private reflections as university-channel deliverables without listing
  their contents;
- excluded raw/private/local files.

- [ ] **Step 2: Write concise release notes**

Summarize the notebook-to-MLOps reorganization, six-slice chronology,
validation, MLflow promotion, operating threshold, monitoring, structured
human-review decision, Make/DVC/Docker/CI reproducibility, evidence package,
limitations, and final deliverables. Call 97/100 an internal target only.

- [ ] **Step 3: Generate public artifact hashes**

After the deck plan has produced its files, run:

```bash
cd submission
shasum -a 256 \
  Fraud_MLOps_Technical_Report.docx \
  Fraud_MLOps_Technical_Report.pdf \
  Fraud_MLOps_Presentation.pptx \
  Fraud_MLOps_Presentation.pdf \
  > SHA256SUMS.txt
cd ..
shasum -a 256 -c submission/SHA256SUMS.txt
```

Expected: all four artifacts verify.

- [ ] **Step 4: Prove private and large files are excluded**

Run:

```bash
git ls-files | rg '(^submission-private/|Reflection\\.(docx|pdf)$|data\\.zip$|\
data/raw/creditcard\\.csv$|mlflow\\.db$)' && exit 1 || true
find . -type f -size +100M -not -path './.git/*' -print
```

Expected: no tracked private/raw/database file. The size scan may show the
preserved ignored `data.zip` or real CSV locally; neither may enter a public
archive or release.

- [ ] **Step 5: Commit the public manifest after the deck exists**

```bash
git add submission/MANIFEST.md submission/RELEASE_NOTES.md \
  submission/SHA256SUMS.txt
git commit -m "docs: add submission manifest and hashes"
```

- [ ] **Step 6: Stop for explicit publication approval**

Do not create a tag, push, create a release, or submit files in this task.
Present the verified file list and request explicit user authorization for the
specific external action.

## Report Plan Acceptance Gate

- Frozen evidence comes from a verified real-data run and names its technical commit.
- `docs/RUBRIC_EVIDENCE.md` maps all six criteria and eight operational requirements.
- `docs/MODEL_CARD.md` states intended use, excluded use, risks, delayed labels, and approval policy.
- README exposes synthetic, real, DVC, Docker, and complete verification paths accurately.
- Diagrams show validation, six slices, separate model validation/calibration, and human approval.
- README and report evidence snapshot lines pass the consistency test.
- Report source contains 4,300–4,600 words; final rendered report contains 4,000–5,000.
- DOCX and PDF pass page-by-page visual inspection with no comments or tracked changes.
- Public manifest excludes raw data, private reflections, local database, secrets, and `data.zip`.
- No external publishing occurs without explicit approval.
