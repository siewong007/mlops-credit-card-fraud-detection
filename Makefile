.PHONY: ingest validate train evaluate inference drift trigger test pipeline

ingest:
	python -m src.ingest
validate:
	python -m src.validate
train:
	python -m src.train
evaluate:
	python -m src.evaluate
inference:
	python -m src.batch_inference
drift:
	python -m src.drift
trigger:
	python -m src.retrain_trigger
test:
	pytest -q
pipeline: ingest validate train evaluate inference drift trigger
