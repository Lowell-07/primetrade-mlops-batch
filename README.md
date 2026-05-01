# Production MLOps Batch Pipeline

Production-grade financial batch pipeline with:

- staged execution: `ingest -> validate -> transform -> features -> signals -> metrics -> output`
- data contracts for OHLCV schema safety
- deterministic metric computation
- lineage via `run_id`, `dataset_hash`, `config_hash`
- structured JSON logs
- CI gates for linting, typing, unit tests, regression validation, and pipeline execution

## Local Run

```bash
pip install -r requirements.txt
python run.py --input data.csv --config config.yaml --output metrics.json --log-file run.log
```
