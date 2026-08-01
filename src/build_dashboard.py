"""Render pipeline evidence contracts as a self-contained monitoring dashboard."""
import json
import shutil
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from src.config import REPORTS_DIR, ROOT

SITE_DIR = ROOT / "site"

_BADGE = {
    "ok": ("ok", "badge-ok"),
    "warning": ("warning", "badge-warn"),
    "retrain": ("retrain", "badge-bad"),
}

_CSS = """
*{box-sizing:border-box}
:root{
  --bg:#ffffff; --fg:#1a1c1e; --muted:#5b6470; --card:#f6f8fa; --line:#d8dee4;
  --ok:#1a7f37; --ok-bg:#dafbe1; --warn:#9a6700; --warn-bg:#fff8c5;
  --bad:#cf222e; --bad-bg:#ffebe9; --accent:#0969da;
}
@media (prefers-color-scheme:dark){
  :root{--bg:#0d1117;--fg:#e6edf3;--muted:#9198a1;--card:#161b22;--line:#30363d;
        --ok:#3fb950;--ok-bg:#12261e;--warn:#d29922;--warn-bg:#2b2411;
        --bad:#f85149;--bad-bg:#2d1214;--accent:#4493f8}
}
html{color-scheme:light dark}
body{margin:0;padding:2rem 1.25rem 4rem;background:var(--bg);color:var(--fg);
  font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
.wrap{max-width:1080px;margin:0 auto}
h1{font-size:1.75rem;margin:0 0 .25rem}
h2{font-size:1.15rem;margin:2.5rem 0 .75rem;padding-bottom:.4rem;border-bottom:1px solid var(--line)}
.sub{color:var(--muted);margin:0 0 1.5rem}
.note{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--accent);
  border-radius:6px;padding:.75rem 1rem;margin:1rem 0;font-size:.9rem}
.grid{display:grid;gap:.75rem;grid-template-columns:repeat(auto-fit,minmax(160px,1fr))}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:.85rem 1rem}
.kpi .k{color:var(--muted);font-size:.75rem;text-transform:uppercase;letter-spacing:.04em}
.kpi .v{font-size:1.4rem;font-weight:600;margin-top:.15rem;font-variant-numeric:tabular-nums}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;min-width:920px;font-size:.9rem}
th,td{text-align:left;padding:.55rem .7rem;border-bottom:1px solid var(--line);
  font-variant-numeric:tabular-nums;white-space:nowrap}
th{color:var(--muted);font-weight:600;font-size:.78rem;text-transform:uppercase;letter-spacing:.04em}
.badge{display:inline-block;padding:.1rem .55rem;border-radius:999px;font-size:.78rem;font-weight:600}
.badge-ok{color:var(--ok);background:var(--ok-bg)}
.badge-warn{color:var(--warn);background:var(--warn-bg)}
.badge-bad{color:var(--bad);background:var(--bad-bg)}
.reason{color:var(--muted);font-size:.82rem;white-space:normal;min-width:260px}
figure{margin:0 0 1.25rem;background:var(--card);border:1px solid var(--line);
  border-radius:8px;padding:.85rem;overflow:hidden}
figure img{width:100%;height:auto;display:block;border-radius:4px;background:#fff}
figcaption{color:var(--muted);font-size:.82rem;margin-top:.5rem}
.figs{display:grid;gap:1rem;grid-template-columns:repeat(auto-fit,minmax(320px,1fr))}
ul.links{list-style:none;padding:0}
ul.links li{margin:.35rem 0}
a{color:var(--accent)}
code{background:var(--card);border:1px solid var(--line);border-radius:4px;padding:.1rem .35rem;font-size:.85em}
footer{margin-top:3rem;padding-top:1rem;border-top:1px solid var(--line);
  color:var(--muted);font-size:.82rem}
"""


def _kpi(label: str, value) -> str:
    return (
        f'<div class="kpi"><div class="k">{escape(label)}</div>'
        f'<div class="v">{escape("n/a" if value is None else str(value))}</div></div>'
    )


def _metric(value, digits=3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _count(value) -> str:
    return "n/a" if value is None else f"{value:,}"


def _badge(status: str) -> str:
    label, css = _BADGE[status]
    return f'<span class="badge {css}">{escape(label)}</span>'


def _copy_assets(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    if source.exists():
        shutil.copytree(source, destination)
    else:
        destination.mkdir()


def build(
    reports_dir: Path = REPORTS_DIR,
    site_dir: Path = SITE_DIR,
) -> dict:
    """Build a static page from the seven stable report JSON contracts."""
    promotion = json.loads((reports_dir / "promotion_record.json").read_text())
    operating_point = json.loads((reports_dir / "operating_point.json").read_text())
    operating = json.loads((reports_dir / "metrics_operating.json").read_text())
    default = json.loads((reports_dir / "metrics_default.json").read_text())
    summaries = json.loads((reports_dir / "drift_summary.json").read_text())
    trigger = json.loads((reports_dir / "trigger_decisions.json").read_text())
    manifest = json.loads((reports_dir / "run_manifest.json").read_text())

    decisions_by_batch = {decision["batch"]: decision for decision in trigger["decisions"]}
    summary_batches = {summary["batch"] for summary in summaries}
    if summary_batches != set(decisions_by_batch):
        raise ValueError("drift summaries and trigger decisions have mismatched batch sets")
    decisions = [decisions_by_batch[summary["batch"]] for summary in summaries]

    figures_dir = reports_dir / "figures"
    drift_dir = reports_dir / "drift"

    data = manifest["data"]
    for summary in summaries:
        native_report = drift_dir / f"{summary['batch']}.json"
        if not native_report.exists():
            raise FileNotFoundError(
                f"native per-batch drift evidence missing: {native_report}"
            )
        native = json.loads(native_report.read_text())
        if any(native.get(key) != value for key, value in summary.items()):
            raise ValueError(
                f"native per-batch drift evidence does not match current summary: "
                f"{summary['batch']}"
            )
        evidently_html = drift_dir / f"{summary['batch']}.html"
        if (
            summary["evidently"]["status"] == "generated"
            and not evidently_html.is_file()
        ):
            raise FileNotFoundError(
                f"generated Evidently HTML evidence missing: {evidently_html}"
            )

    site_dir.mkdir(parents=True, exist_ok=True)
    (site_dir / ".nojekyll").write_text("")
    fig_out = site_dir / "figures"
    drift_out = site_dir / "drift"
    _copy_assets(figures_dir, fig_out)
    _copy_assets(drift_dir, drift_out)

    def evidently_cell(summary):
        evidence = summary["evidently"]
        status = evidence["status"]
        error = (
            ""
            if status == "generated"
            else f"<br>{escape(str(evidence['error_type']))}: "
            f"{escape(str(evidence['message']))}"
        )
        return f"Evidently: {escape(status)}{error}"

    rows = "\n".join(
        f"<tr><td><code>{escape(summary['batch'])}</code></td>"
        f"<td>{_count(summary['n_rows'])}</td><td>{_metric(summary['pr_auc'])}</td>"
        f"<td>{_metric(summary['recall'])}</td><td>{_metric(summary['precision'])}</td>"
        f"<td>{_metric(summary['pct_drifted_features'], 1)}%</td>"
        f"<td>{_metric(summary['amount_psi'])}</td><td>{_metric(summary['prediction_psi'])}</td>"
        f"<td>{escape(decision['label_status'])}</td><td>{_badge(decision['status'])}</td>"
        f"<td class=\"reason\">{escape(', '.join(decision['reason_codes']))}<br>"
        f"{escape('; '.join(decision['reasons']))}</td>"
        f"<td class=\"reason\">{evidently_cell(summary)}</td></tr>"
        for summary, decision in zip(summaries, decisions)
    )

    figures = "\n".join(
        f'<figure><img src="figures/{escape(figure.name)}" alt="{escape(figure.stem)}">'
        f"<figcaption>{escape(figure.stem.replace('_', ' '))}</figcaption></figure>"
        for figure in sorted(figures_dir.glob("*.png"))
    )
    report_links = []
    for summary in summaries:
        batch = summary["batch"]
        report_links.append(
            f'<li><a href="drift/{escape(batch)}.json">'
            f"Native drift evidence — {escape(batch)}</a><br>"
            f'<span class="evidence-id">Native drift {escape(batch)} · model '
            f"{escape(str(summary['promoted_model_id']))} · operating threshold "
            f"{escape(_metric(summary['operating_threshold'], 2))} · batch fingerprint "
            f"{escape(summary['batch_data_fingerprint'])} · labels "
            f"{escape(summary['label_status'])} · Evidently "
            f"{escape(summary['evidently']['status'])}</span></li>"
        )
        html_report = drift_dir / f"{batch}.html"
        if summary["evidently"]["status"] == "generated":
            report_links.append(
                f'<li><a href="drift/{escape(batch)}.html">'
                f"Evidently HTML — {escape(batch)}</a></li>"
            )
    links = "\n".join(report_links)

    costs = operating_point["cost_assumptions"]
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    evidence_id = (
        f"Model {escape(promotion['promoted_model_id'])} · "
        f"MLflow run {escape(promotion['mlflow_run_id'])} · "
        f"registry version {escape(str(promotion['registered_model_version']))} · "
        f"operating threshold {escape(_metric(operating_point['threshold'], 2))} · "
        f"status {escape(trigger['overall_status'])}"
    )
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fraud detection — model monitoring dashboard</title>
<style>{_CSS}</style></head><body><div class="wrap">

<h1>Credit card fraud detection — monitoring dashboard</h1>
<p class="sub">Evidence-contract rendering · generated {generated}</p>
<p class="evidence-id">{evidence_id}</p>

<h2>Promotion and data evidence</h2>
<div class="grid">
  {_kpi("Promoted model", promotion["promoted_model_id"])}
  {_kpi("MLflow run", promotion["mlflow_run_id"])}
  {_kpi("Model version", f"Version {promotion['registered_model_version']}")}
  {_kpi("Data source", data["source"])}
  {_kpi("Data fingerprint", data["fingerprint_sha256"])}
  {_kpi("Overall status", trigger["overall_status"])}
</div>

<h2>Operating threshold evidence</h2>
<div class="grid">
  {_kpi("Operating threshold", _metric(operating_point["threshold"], 2))}
  {_kpi("PR-AUC", _metric(operating["pr_auc"]))}
  {_kpi("ROC-AUC", _metric(operating["roc_auc"]))}
  {_kpi("Recall", _metric(operating["recall"]))}
  {_kpi("Precision", _metric(operating["precision"]))}
</div>
<div class="note">Operating threshold {escape(_metric(operating_point["threshold"], 2))}
minimises the stated cost rationale: false negative {escape(str(costs["false_negative"]))}
and false positive {escape(str(costs["false_positive"]))}.</div>

<h2>Default threshold comparison</h2>
<div class="grid">
  {_kpi("Default threshold", _metric(default["threshold"], 2))}
  {_kpi("PR-AUC", _metric(default["pr_auc"]))}
  {_kpi("ROC-AUC", _metric(default["roc_auc"]))}
  {_kpi("Recall", _metric(default["recall"]))}
  {_kpi("Precision", _metric(default["precision"]))}
</div>

<h2>Per-batch monitoring evidence</h2>
<div class="scroll"><table>
<thead><tr><th>Batch</th><th>Rows</th><th>PR-AUC</th><th>Recall</th><th>Precision</th>
<th>Drifted</th><th>Amount PSI</th><th>Prediction PSI</th><th>Labels</th><th>Status</th>
<th>Decision evidence</th><th>Evidently</th></tr></thead>
<tbody>{rows}</tbody></table></div>

<h2>Local evaluation and drift figures</h2>
<div class="figs">{figures}</div>

<h2>Local drift evidence</h2>
<ul class="links">{links}</ul>

<footer>A <code>retrain</code> status is review-only and requires human approval;
it does not start training or replace the promoted model.</footer>

</div></body></html>
"""
    (site_dir / "index.html").write_text(page, encoding="utf-8")
    return {"status": trigger["overall_status"], "decisions": decisions, "site": site_dir}


def main() -> None:
    out = build()
    print(f"dashboard written to {out['site']}/index.html  (overall status: {out['status']})")


if __name__ == "__main__":
    main()
