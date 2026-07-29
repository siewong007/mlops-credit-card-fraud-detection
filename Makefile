.PHONY: fetch-data data validate ingest train threshold evaluate inference drift \
	trigger manifest pipeline dashboard test-fast test verify verify-dvc clean

# Download the REAL ULB dataset from OpenML (no Kaggle account needed).
# Run this once locally to work with real data.
fetch-data:
	python -m src.fetch_data

# Ensure *some* dataset exists: prefer the real CSV if already present,
# otherwise generate a synthetic one so the pipeline runs out-of-the-box in
# dev/CI. Use `make fetch-data` for the real dataset.
data:
	@test -f data/raw/creditcard.csv || python -m src.simulate_data

validate: data
	python -m src.validate

ingest: validate
	python -m src.ingest

train: ingest
	python -m src.train

threshold: train
	python -m src.threshold

evaluate: threshold
	python -m src.evaluate

inference: evaluate
	python -m src.batch_inference

drift: inference
	python -m src.drift

trigger: drift
	python -m src.retrain_trigger

manifest: trigger
	python -m src.evidence

pipeline: manifest

dashboard: manifest
	python -m src.build_dashboard

test-fast:
	python -m pytest -q -m "not integration"

test:
	python -m pytest -q -ra

verify: dashboard
	python -m pytest -q -ra

verify-dvc:
	scripts/verify_dvc_clean_clone.sh

clean:
	rm -rf models site reports/drift reports/*.json reports/trigger_log.md \
	       reports/figures/*.png data/batches/*.csv data/batches/preds_*.csv mlruns
