# MLOps Batch Pipeline

I built this for a technical assessment — a minimal, working batch pipeline that takes OHLCV data, computes a rolling mean on the close price, and spits out a binary trading signal. Nothing fancy, but it handles edge cases and won't crash on bad input.

## What It Actually Does

1. **Loads config** from YAML — checks that `seed` (int), `window` (int &gt; 0), and `version` (string) exist and are valid types
2. **Loads CSV** — validates the file exists, isn't empty, parses as CSV, and has a `close` column
3. **Validates data quality** — checks `close` is numeric, has no Inf values, and that the dataset isn't smaller than the rolling window
4. **Computes rolling mean** — first `window-1` rows get NaN (expected behavior, no cheating)
5. **Generates signal** — `1` if `close &gt; rolling_mean`, else `0`; NaN where rolling mean is NaN
6. **Outputs metrics JSON** — exact schema required by the task
7. **Logs everything** — plain text logs to file, plus JSON printed to stdout for Docker visibility

## The CLI

```bash
python run.py \
  --input data.csv \
  --config config.yaml \
  --output metrics.json \
  --log-file run.log