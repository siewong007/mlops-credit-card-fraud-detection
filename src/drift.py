"""Drift monitoring with Evidently (Owner: C). Requirement 2, SS12.

Compare each production batch against the training reference:
feature drift (KS/PSI), prediction drift, target/fraud-rate drift,
performance drift (recompute metrics where labels available).
"""


def main() -> None:
    # TODO(C): for each prod batch build an Evidently Report
    # (DataDriftPreset + custom metrics), save HTML to reports/,
    # and emit a summary JSON: {batch, pct_drifted_features, pr_auc, recall}
    # used by src.retrain_trigger.
    raise SystemExit("TODO: implement drift analysis")


if __name__ == "__main__":
    main()
