# MLOps Batch Pipeline

Minimal, deterministic batch pipeline for generating trading signals from OHLCV data.

## What It Does

1. Loads a CSV with OHLCV data
2. Validates the `close` column exists
3. Computes rolling mean on `close` with configurable window
4. Generates binary signal: `1` if close &gt; rolling mean, else `0`
5. Outputs structured metrics JSON and execution logs

## CLI

```bash
python run.py \
  --input data.csv \
  --config config.yaml \
  --output metrics.json \
  --log-file run.log