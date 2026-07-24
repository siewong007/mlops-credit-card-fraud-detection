.PHONY: data ingest validate train threshold evaluate inference drift trigger test pipeline clean

# Generate a synthetic dataset only if the real Kaggle CSV is absent, so the
# pipeline runs out-of-the-box in dev/CI. Drop the real creditcard.csv into
# data/raw/ and this step is skipped.
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
test:
	python -m pytest

# Full reorganised workflow, in dependency order.
pipeline: ingest validate train threshold evaluate inference drift trigger

clean:
	rm -rf models reports/drift reports/*.json reports/trigger_log.md \
	       reports/figures/*.png data/batches/*.csv data/batches/preds_*.csv mlruns
