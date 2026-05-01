# PrimeTrade MLOps Batch Pipeline

A production-style MLOps batch pipeline for deterministic financial signal generation on OHLCV data.

This project demonstrates how a small ML system can be engineered with production discipline — focusing on reproducibility, observability, validation, and deployment readiness.

---

## 🧠 What this system does

The pipeline processes market data and generates trading signals using a simple but deterministic rule:

> `signal = 1 if close > rolling_mean(window), else 0`

It is designed to behave exactly the same across runs given the same input and configuration.

---

## ⚙️ Pipeline Architecture

The system is built as a staged batch pipeline:

```
ingest → validate → transform → features → signals → metrics → output
```

Each stage is isolated, testable, and observable.

### Stage responsibilities:

- **Ingest** → Load and validate CSV input
- **Validate** → Enforce OHLCV schema + data integrity rules
- **Transform** → Clean and normalize data
- **Features** → Compute rolling mean
- **Signals** → Generate binary trading signal
- **Metrics** → Compute signal rate + quality metrics
- **Output** → Persist JSON + logs

---

## 📊 Observability & Lineage

Every run is fully traceable using:

- `run_id` → unique execution identifier  
- `dataset_hash` → input data fingerprint  
- `config_hash` → configuration fingerprint  

### Metrics tracked:

- rows_input
- rows_valid
- rows_dropped
- rows_used
- signal_rate
- data_quality_score
- coverage
- latency_ms

Logs are emitted in structured JSON format for machine readability.

---

## 🔁 Reproducibility

The pipeline is fully deterministic:

- Config-driven execution (YAML)
- No hardcoded parameters
- Same input + config → same output
- Regression validation enforced in CI

---

## 🚀 How to run locally

```bash
python run.py \
  --input data.csv \
  --config config.yaml \
  --output metrics.json \
  --log-file run.log
```

---

## 🐳 Docker Execution

Build and run the pipeline in a container:

```bash
docker build -t mlops-task .
docker run --rm mlops-task
```

This executes the full pipeline and prints final metrics.

---

## 🧪 Testing & CI

The project includes a full CI pipeline:

- linting (ruff)
- type checking (mypy)
- unit tests (pytest)
- regression validation (deterministic recomputation)

GitHub Actions ensures every commit is validated before merge.

---

## 🧰 Tech Stack

- Python 3.9
- Pandas
- PyYAML
- Pytest
- Ruff
- MyPy
- Docker

---

## 📦 Key Design Principles

- **Deterministic execution**
- **Strict data contracts (OHLCV validation)**
- **Explicit failure handling**
- **Structured observability**
- **CI-gated correctness**
- **Production-style pipeline separation**

---

## 📈 Example Output

```json
{
  "status": "success",
  "version": "v2",
  "signal_rate": 0.5625,
  "rows_input": 20,
  "rows_valid": 20,
  "rows_used": 16,
  "coverage": 0.8,
  "data_quality_score": 1.0,
  "latency_ms": 32
}
```

---

## 🎯 What this project demonstrates

This is not a model-heavy ML system — it is an engineering-focused MLOps pipeline.

It demonstrates:

- how to design reproducible batch ML systems
- how to enforce data correctness via contracts
- how to structure observable pipelines
- how to productionize even simple ML logic

---

## ⚠️ Notes

- All outputs are deterministic
- All failures are explicit and traceable
- All runs are reproducible via config + dataset hash
