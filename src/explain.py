"""SHAP explainability for the promoted model (Owner: Chun Yun).

Requirement 4 (asymmetric FN/FP costs need explainable flags) and briefing §8
(SHAP for model interpretation and audit support). Produces global feature
attribution over the calibration batch plus a local explanation for the
highest-scoring flagged transaction, so an analyst can answer "why was this
transaction queued for review?".
"""
import numpy as np
import pandas as pd
import shap


def _build_explainer(model, background: pd.DataFrame):
    """Pick the explainer that matches the promoted model family."""
    name = type(model).__name__
    if name in {"XGBClassifier", "LGBMClassifier", "RandomForestClassifier",
                "GradientBoostingClassifier", "DecisionTreeClassifier"}:
        return shap.TreeExplainer(model), "TreeExplainer"
    if name == "LogisticRegression":
        masker = shap.maskers.Independent(background, max_samples=len(background))
        return shap.LinearExplainer(model, masker), "LinearExplainer"
        #return shap.LinearExplainer(model, background), "LinearExplainer"
    # Fallback: model-agnostic, slower but always correct.
    return shap.Explainer(model.predict_proba, background), "Explainer"


def _positive_class_values(raw) -> np.ndarray:
    """Normalise SHAP output shape to (n_samples, n_features) for class 1."""
    if isinstance(raw, list):            # [class0, class1]
        return np.asarray(raw[1])
    arr = np.asarray(raw)
    if arr.ndim == 3:                    # (n, features, classes)
        return arr[:, :, 1]
    return arr


#def explain_probabilities(model, X: pd.DataFrame, cfg: dict) -> tuple[dict, np.ndarray, pd.DataFrame]:
    #sample = X.sample(n=min(cfg["sample_size"], len(X)), random_state=cfg["seed"])
def explain_probabilities(model, X: pd.DataFrame, y: pd.Series, cfg: dict) -> tuple[dict, np.ndarray, pd.DataFrame]: #Fixed
    sample = X.sample(n=min(cfg["sample_size"], len(X)), random_state=cfg["seed"]) #Fixed
    explainer, kind = _build_explainer(model, sample)
    values = _positive_class_values(explainer.shap_values(sample)
                                    if hasattr(explainer, "shap_values")
                                    else explainer(sample).values)

    ranking = (pd.Series(np.abs(values).mean(axis=0), index=sample.columns)
               .sort_values(ascending=False))
    proba = model.predict_proba(sample)[:, 1]
    flagged = int(np.argmax(proba))

    report = {
        "explainer": kind,
        "sample_size": int(len(sample)),
        "seed": int(cfg["seed"]),
        "top_features": [{"feature": f, "mean_abs_shap": float(v)}
                         for f, v in ranking.head(cfg["top_k"]).items()],
        "explained_transaction": {
            "sample_position": flagged,
            "source_row_index": int(sample.index[flagged]), # Fixed #Added
            "actual_class": int(y.loc[sample.index[flagged]]), # Added
            "predicted_probability": float(proba[flagged]),
            "top_contributions": [
                {"feature": sample.columns[i], "shap_value": float(values[flagged, i])}
                for i in np.argsort(np.abs(values[flagged]))[::-1][:cfg["top_k"]]
            ],
        },
    }
    return report, values, sample


def _plot_summary(values, sample, path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shap.summary_plot(values, sample, show=False, max_display=15, plot_size=(6, 5))
    plt.title("Global SHAP attribution — calibration batch")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


#def _plot_local(values, sample, row: int, path) -> None:\ # Fixed # Version 1
#def _plot_local(values, sample, row: int, path, top_k: int = 15) -> None: # Fixed # Version 2
def _plot_local(values, sample, row: int, path, top_k: int = 15, actual: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    #order = np.argsort(np.abs(values[row]))[::-1][:12][::-1]
    order = np.argsort(np.abs(values[row]))[::-1][:top_k][::-1]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.barh([sample.columns[i] for i in order], [values[row, i] for i in order],
            color=["#c0392b" if values[row, i] > 0 else "#2980b9" for i in order])
    ax.axvline(0, color="black", linewidth=0.8)
    #ax.set(xlabel="SHAP value (→ pushes toward fraud)",
           #title="Local explanation — highest-scoring flagged transaction")
    #ax.set(xlabel="SHAP value (→ pushes toward fraud)", # Fixed # Version 1
           #title=f"Local explanation — highest-scoring transaction{actual}") # Fixed # Version 1
    ax.set(xlabel="SHAP value (→ pushes toward fraud)")
    ax.set_title(f"Local explanation — highest-scoring transaction\n{actual.strip(' ()')}",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    from src import features
    from src.artifacts import load_model, load_threshold
    from src.config import FIGURES_DIR, REPORTS_DIR, batch_dir, ensure_dirs, load_params
    from src.evidence import write_json
    from src.validate import read_stable_csv

    ensure_dirs()
    params = load_params()
    cfg = params["explain"]
    operating_point = load_threshold()
    calibration, _ = read_stable_csv(
        batch_dir(params) / "calibration.csv",
        expected_sha256=operating_point["calibration_data_fingerprint"],
    )
    model, scaler = load_model(), features.load_scaler()
    #X, _ = features.xy(features.transform(calibration, scaler))

    #report, values, sample = explain_probabilities(model, X, cfg)
    X, y = features.xy(features.transform(calibration, scaler)) # Fixed

    report, values, sample = explain_probabilities(model, X, y, cfg) # Fixed
    report.update({
        "evidence_role": "model_explainability",
        "promoted_model_id": operating_point["promoted_model_id"],
        "calibration_data_fingerprint": operating_point["calibration_data_fingerprint"],
    })

    print(f"SHAP ({report['explainer']}) on {report['sample_size']} calibration rows")
    for entry in report["top_features"][:10]:
        print(f"  {entry['feature']:>8}: {entry['mean_abs_shap']:.5f}")
    et = report["explained_transaction"] #Added
    print(f"  explained row {et['source_row_index']}: p={et['predicted_probability']:.6f}, " # Added
          f"actual={'fraud' if et['actual_class'] else 'legitimate'}") # Added

    write_json(REPORTS_DIR / "explainability.json", report)
    _plot_summary(values, sample, FIGURES_DIR / "shap_summary.png")
    #_plot_local(values, sample, report["explained_transaction"]["sample_position"],
                #FIGURES_DIR / "shap_local_flagged.png")
    #_plot_local(values, sample, report["explained_transaction"]["sample_position"], # Fixed # Version 1
               #FIGURES_DIR / "shap_local_flagged.png", cfg["top_k"]) # Fixed # Version 1
    actual = " (actual: fraud)" if report["explained_transaction"]["actual_class"] else " (actual: legitimate)" # Fixed # Version 2
    _plot_local(values, sample, report["explained_transaction"]["sample_position"], # Fixed # Version 2
                FIGURES_DIR / "shap_local_flagged.png", cfg["top_k"], actual) # Fixed # Version 2
    print(f"saved explainability evidence + figures to {REPORTS_DIR}")


if __name__ == "__main__":
    main()