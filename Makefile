.PHONY: fetch-data data ingest validate train threshold evaluate inference drift trigger dashboard test pipeline clean

# Download the REAL ULB dataset from OpenML (no Kaggle account needed).
# Run this once locally to work with real data.
fetch-data:
	python -m src.fetch_data

# Ensure *some* dataset exists: prefer the real CSV if already present,
# otherwise generate a synthetic one so the pipeline runs out-of-the-box in
# dev/CI. Use `make fetch-data` for the real dataset.
data:
	@test -f data/raw/creditcard.csv || python -m src.simulate_data

ingest: data
	python -m src.ingest
validate:
	python -m src.validate
train:
	python -m src.train
threshold:
	python -m src.threshold
evaluate:
	python -m src.evaluate
inference:
	python -m src.batch_inference
drift:
	python -m src.drift
trigger:
	python -m src.retrain_trigger
dashboard:
	python -m src.build_dashboard
test:
	python -m pytest

# Full reorganised workflow, in dependency order.
pipeline: ingest validate train threshold evaluate inference drift trigger

clean:
	rm -rf models site reports/drift reports/*.json reports/trigger_log.md \
	       reports/figures/*.png data/batches/*.csv data/batches/preds_*.csv mlruns
