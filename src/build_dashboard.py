"""Build the static monitoring dashboard published to GitHub Pages.

Turns the evidence produced by the pipeline (`reports/`, `models/`) into a
self-contained static site under `site/`:

    site/index.html      overview: dataset provenance, promoted model, operating
                         point, per-batch drift + trigger decisions, figures
    site/figures/*.png   evaluation and drift figures
    site/drift/*.html    the full Evidently reports (when present)

This is the "monitoring dashboard" face of the workflow: the same JSON contract
the retraining trigger consumes, rendered for humans. No external assets are
referenced, so the page renders offline and on Pages identically.
"""
import json
import shutil
from datetime import datetime, timezone

from src.config import (
    DRIFT_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    ROOT,
    load_params,
    read_provenance,
)
from src.retrain_trigger import decide

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
table{border-collapse:collapse;width:100%;min-width:640px;font-size:.9rem}
th,td{text-align:left;padding:.55rem .7rem;border-bottom:1px solid var(--line);
  font-variant-numeric:tabular-nums;white-space:nowrap}
th{color:var(--muted);font-weight:600;font-size:.78rem;text-transform:uppercase;letter-spacing:.04em}
.badge{display:inline-block;padding:.1rem .55rem;border-radius:999px;font-size:.78rem;font-weight:600}
.badge-ok{color:var(--ok);background:var(--ok-bg)}
.badge-warn{color:var(--warn);background:var(--warn-bg)}
.badge-bad{color:var(--bad);background:var(--bad-bg)}
.reason{color:var(--muted);font-size:.82rem;white-space:normal}
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


def _kpi(label: str, value: str) -> str:
    return f'<div class="kpi"><div class="k">{label}</div><div class="v">{value}</div></div>'


def _reasons(baseline_pr_auc: float, b: dict, params: dict) -> str:
    from src.retrain_trigger import _reasons as r

    return r(baseline_pr_auc, b, params)


def build() -> dict:
    params = load_params()
    baseline = json.loads((MODELS_DIR / "baseline.json").read_text())
    threshold = json.loads((MODELS_DIR / "threshold.json").read_text())
    summaries = json.loads((REPORTS_DIR / "drift_summary.json").read_text())
    valid = json.loads((REPORTS_DIR / "metrics_valid.json").read_text())
    prov = read_provenance(params)

    base_pr = baseline["valid_at_0.5"]["pr_auc"]
    t = threshold["threshold"]
    tm = threshold["valid_metrics_at_threshold"]

    decisions = [(b, decide(base_pr, b, params)) for b in summaries]
    worst = ("retrain" if any(d == "retrain" for _, d in decisions)
             else "warning" if any(d == "warning" for _, d in decisions) else "ok")

    # --- assets ------------------------------------------------------------
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    (SITE_DIR / ".nojekyll").write_text("")  # serve _-prefixed paths verbatim
    fig_out = SITE_DIR / "figures"
    fig_out.mkdir(exist_ok=True)
    for png in sorted(FIGURES_DIR.glob("*.png")):
        shutil.copy2(png, fig_out / png.name)
    drift_out = SITE_DIR / "drift"
    drift_out.mkdir(exist_ok=True)
    evidently = sorted(DRIFT_DIR.glob("*.html"))
    for html in evidently:
        shutil.copy2(html, drift_out / html.name)

    # --- html --------------------------------------------------------------
    is_real = prov.get("source", "").startswith("REAL")
    prov_note = (
        f'<div class="note"><strong>Data provenance:</strong> {prov.get("source", "unknown")}'
        + (f' — {prov["rows"]:,} rows, {prov["fraud"]} fraud '
           f'({prov["fraud_rate"]:.5f}).' if "rows" in prov else "")
        + ("" if is_real else
           " <strong>These figures come from synthetic data</strong> and demonstrate the "
           "workflow only — they are not submission-grade results.")
        + "</div>"
    )

    rows = "\n".join(
        f"<tr><td><code>{b['batch']}</code></td>"
        f"<td>{b['n_rows']:,}</td><td>{b['pr_auc']:.3f}</td><td>{b['recall']:.3f}</td>"
        f"<td>{b['precision']:.3f}</td><td>{b['pct_drifted_features']:.0f}%</td>"
        f"<td>{b['amount_psi']:.2f}</td><td>{b['prediction_psi']:.2f}</td>"
        f'<td><span class="badge {_BADGE[d][1]}">{_BADGE[d][0]}</span></td>'
        f'<td class="reason">{_reasons(base_pr, b, params)}</td></tr>'
        for b, d in decisions
    )

    figs = "\n".join(
        f'<figure><img src="figures/{name}" alt="{cap}"><figcaption>{cap}</figcaption></figure>'
        for name, cap in [
            ("pr_curve.png", "Precision–Recall curve (validation)"),
            ("confusion_matrix.png", "Confusion matrix (validation, threshold 0.5)"),
            ("threshold_tradeoff.png", "Cost-based threshold selection"),
            *[(f"psi_{b['batch']}.png", f"Feature PSI vs training reference — {b['batch']}")
              for b in summaries],
        ] if (fig_out / name).exists()
    )

    links = "\n".join(
        f'<li><a href="drift/{h.name}">Evidently drift report — {h.stem}</a></li>'
        for h in evidently
    ) or '<li class="reason">Evidently reports not generated in this run '\
         '(native PSI/KS drift analysis above is the primary evidence).</li>'

    trig = params["trigger"]
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fraud detection — model monitoring dashboard</title>
<style>{_CSS}</style></head><body><div class="wrap">

<h1>Credit card fraud detection — monitoring dashboard</h1>
<p class="sub">Automated output of the reorganised MLOps pipeline · generated {generated}</p>

{prov_note}

<h2>Promoted model &amp; operating point</h2>
<div class="grid">
  {_kpi("Promoted model", baseline["model"].replace("_", " "))}
  {_kpi("Baseline PR-AUC", f"{base_pr:.3f}")}
  {_kpi("ROC-AUC", f"{valid['roc_auc']:.3f}")}
  {_kpi("Threshold", f"{t:.2f}")}
  {_kpi("Recall @ threshold", f"{tm['recall']:.3f}")}
  {_kpi("Precision @ threshold", f"{tm['precision']:.3f}")}
  {_kpi("Overall status", _BADGE[worst][0])}
</div>
<div class="note">The operating threshold is chosen by minimising expected business
cost with a false negative weighted {params['threshold']['cost_false_negative']}&times; a
false positive — not left at the default 0.5.</div>

<h2>Per-batch monitoring &amp; retraining trigger</h2>
<div class="scroll"><table>
<thead><tr><th>Batch</th><th>Rows</th><th>PR-AUC</th><th>Recall</th><th>Precision</th>
<th>Drifted</th><th>Amount PSI</th><th>Pred PSI</th><th>Decision</th><th>Reason</th></tr></thead>
<tbody>
{rows}
</tbody></table></div>
<div class="note"><strong>Trigger rule:</strong> retrain if PR-AUC drops more than
{trig['pr_auc_drop_pct']}% versus the validation baseline, <em>or</em> recall at the
operating threshold falls below {trig['recall_floor']}, <em>or</em> more than
{trig['drifted_features_pct']}% of monitored features drift (PSI &gt;
{params['drift']['psi_threshold']}). Drift is flagged on PSI magnitude rather than a KS
p-value, which is over-sensitive at these batch sizes.</div>

<h2>Evaluation &amp; drift figures</h2>
<div class="figs">
{figs}
</div>

<h2>Full drift reports</h2>
<ul class="links">
{links}
</ul>

<footer>
Generated by <code>python -m src.build_dashboard</code> from the pipeline's own evidence
(<code>reports/</code>, <code>models/</code>). Academic project — the dataset covers two
days of transactions, so this demonstrates operational monitoring rather than long-term
production drift.
</footer>

</div></body></html>
"""
    (SITE_DIR / "index.html").write_text(html)
    return {"status": worst, "decisions": decisions, "site": SITE_DIR}


def main() -> None:
    if not (MODELS_DIR / "baseline.json").exists():
        raise SystemExit("models/baseline.json missing — run `make pipeline` first")
    out = build()
    print(f"dashboard written to {out['site']}/index.html  (overall status: {out['status']})")


if __name__ == "__main__":
    main()
