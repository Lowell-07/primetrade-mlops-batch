from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from pipeline_app.hashing import sha256_file, sha256_mapping

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")
NUMERIC_COLUMNS = REQUIRED_COLUMNS


class PipelineError(Exception):
    retryable = False


class RetryablePipelineError(PipelineError):
    retryable = True


class PipelineExecutionError(Exception):
    def __init__(self, error_payload: dict[str, Any]) -> None:
        message = error_payload.get("error_message", "Pipeline execution failed")
        super().__init__(message)
        self.error_payload = error_payload


@dataclass(frozen=True)
class PipelineConfig:
    version: str
    window: int
    max_invalid_close_ratio: float
    min_rows_required: int
    timestamp_column: str


@dataclass(frozen=True)
class RunContext:
    run_id: str
    dataset_hash: str
    config_hash: str


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
        }
        for field in ("event", "run_id", "stage", "details"):
            if hasattr(record, field):
                payload[field] = getattr(record, field)
        return json.dumps(payload, sort_keys=True)


def run_pipeline(
    *,
    input_path: Path,
    config_path: Path,
    output_path: Path,
    log_path: Path,
) -> dict[str, Any]:
    run_id = str(uuid.uuid4())
    dataset_hash = sha256_file(input_path) if input_path.exists() else "unavailable"

    try:
        config, config_hash = load_config(config_path)
    except PipelineError as exc:
        payload = failure_payload(
            version="unknown",
            run_id=run_id,
            dataset_hash=dataset_hash,
            config_hash="unavailable",
            error_type=type(exc).__name__,
            error_message=str(exc),
            retryable=exc.retryable,
        )
        write_json(output_path, payload)
        raise PipelineExecutionError(payload) from exc

    context = RunContext(
        run_id=run_id,
        dataset_hash=dataset_hash,
        config_hash=config_hash,
    )
    logger = setup_logger(log_path)
    start = time.perf_counter()

    try:
        raw_df = read_csv(input_path)
        log_event(
            logger,
            "ingest_completed",
            context,
            "ingest",
            {"rows_input": int(len(raw_df)), "columns": list(raw_df.columns)},
        )

        df, quality = validate_data(raw_df, config)
        log_event(logger, "validation_completed", context, "validate", quality)

        feature_start = time.perf_counter()
        df["rolling_mean"] = (
            df["close"].rolling(window=config.window, min_periods=config.window).mean()
        )
        valid_mask = df["rolling_mean"].notna()
        df.loc[:, "signal"] = pd.Series(pd.NA, index=df.index, dtype="object")
        df.loc[valid_mask, "signal"] = (
            df.loc[valid_mask, "close"] > df.loc[valid_mask, "rolling_mean"]
        ).astype(int)
        feature_ms = elapsed_ms(feature_start)
        rows_used = int(valid_mask.sum())
        log_event(
            logger,
            "signal_generation_completed",
            context,
            "features",
            {
                "rows_with_feature": rows_used,
                "rows_without_feature": int(len(df)) - rows_used,
                "feature_ms": feature_ms,
            },
        )

        metrics = build_metrics(df, config, context, quality, elapsed_ms(start))
        write_json(output_path, metrics)
        log_event(
            logger,
            "job_completed",
            context,
            "output",
            {"output_path": str(output_path), "latency_ms": metrics["latency_ms"]},
        )
        return metrics
    except PipelineError as exc:
        payload = failure_payload(
            version=config.version,
            run_id=context.run_id,
            dataset_hash=context.dataset_hash,
            config_hash=context.config_hash,
            error_type=type(exc).__name__,
            error_message=str(exc),
            retryable=exc.retryable,
        )
        write_json(output_path, payload)
        log_event(logger, "job_failed", context, "failure", payload, logging.ERROR)
        raise PipelineExecutionError(payload) from exc
    except OSError as exc:
        payload = failure_payload(
            version=config.version,
            run_id=context.run_id,
            dataset_hash=context.dataset_hash,
            config_hash=context.config_hash,
            error_type="RetryablePipelineError",
            error_message=str(exc),
            retryable=True,
        )
        write_json(output_path, payload)
        log_event(logger, "job_failed", context, "failure", payload, logging.ERROR)
        raise PipelineExecutionError(payload) from exc


def load_config(config_path: Path) -> tuple[PipelineConfig, str]:
    if not config_path.exists():
        raise RetryablePipelineError(f"Config file not found: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except OSError as exc:
        raise RetryablePipelineError(f"Failed to read config: {exc}") from exc
    except yaml.YAMLError as exc:
        raise PipelineError(f"Invalid YAML config: {exc}") from exc

    if not isinstance(raw, dict):
        raise PipelineError("Config must be a YAML mapping")

    version = raw.get("version")
    window = raw.get("window")
    max_invalid_close_ratio = raw.get("max_invalid_close_ratio", 0.0)
    min_rows_required = raw.get("min_rows_required", 1)
    timestamp_column = raw.get("timestamp_column", "date")

    if not isinstance(version, str) or not version.strip():
        raise PipelineError("Config 'version' must be a non-empty string")
    if not isinstance(window, int) or window <= 0:
        raise PipelineError("Config 'window' must be an integer > 0")
    if not isinstance(max_invalid_close_ratio, (int, float)):
        raise PipelineError("Config 'max_invalid_close_ratio' must be numeric")
    if not 0.0 <= float(max_invalid_close_ratio) <= 1.0:
        raise PipelineError("Config 'max_invalid_close_ratio' must be between 0 and 1")
    if not isinstance(min_rows_required, int) or min_rows_required <= 0:
        raise PipelineError("Config 'min_rows_required' must be an integer > 0")
    if not isinstance(timestamp_column, str) or not timestamp_column.strip():
        raise PipelineError("Config 'timestamp_column' must be a non-empty string")

    return (
        PipelineConfig(
            version=version.strip(),
            window=window,
            max_invalid_close_ratio=float(max_invalid_close_ratio),
            min_rows_required=min_rows_required,
            timestamp_column=timestamp_column.strip(),
        ),
        sha256_mapping(raw),
    )


def read_csv(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise RetryablePipelineError(f"Input file not found: {input_path}")
    if input_path.stat().st_size == 0:
        raise PipelineError(f"Input file is empty: {input_path}")
    try:
        return pd.read_csv(input_path)
    except pd.errors.EmptyDataError as exc:
        raise PipelineError(f"Input file has no data: {input_path}") from exc
    except pd.errors.ParserError as exc:
        raise PipelineError(f"Invalid CSV format: {exc}") from exc
    except OSError as exc:
        raise RetryablePipelineError(f"Failed to read input file: {exc}") from exc


def validate_data(
    dataframe: pd.DataFrame, config: PipelineConfig
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows_input = int(len(dataframe))
    if rows_input == 0:
        raise PipelineError("Input CSV has no rows")

    timestamp_column = config.timestamp_column
    required_columns = [timestamp_column, *REQUIRED_COLUMNS]
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise PipelineError(
            "Missing required columns: "
            f"{missing}. Available columns: {list(dataframe.columns)}"
        )

    df = dataframe.loc[:, required_columns].copy()
    df[timestamp_column] = pd.to_datetime(df[timestamp_column], errors="coerce")
    invalid_timestamp_rows = int(df[timestamp_column].isna().sum())
    if invalid_timestamp_rows:
        raise PipelineError(
            f"Found {invalid_timestamp_rows} rows with invalid timestamps"
        )
    if df[timestamp_column].duplicated().any():
        duplicates = int(df[timestamp_column].duplicated().sum())
        raise PipelineError(f"Found {duplicates} duplicate timestamps")
    if not df[timestamp_column].is_monotonic_increasing:
        raise PipelineError("Timestamps must be strictly increasing")

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    invalid_close_rows = int(df["close"].isna().sum())
    if invalid_close_rows / rows_input > config.max_invalid_close_ratio:
        raise PipelineError(
            "Invalid close ratio exceeds threshold: "
            f"{invalid_close_rows}/{rows_input} > {config.max_invalid_close_ratio:.4f}"
        )
    if invalid_close_rows:
        df = df.loc[df["close"].notna()]
    if df.empty:
        raise PipelineError("All rows became invalid after close-value coercion")

    other_numeric_nulls = {
        column: int(df[column].isna().sum())
        for column in ("open", "high", "low", "volume")
        if int(df[column].isna().sum()) > 0
    }
    if other_numeric_nulls:
        raise PipelineError(
            f"Found invalid numeric values outside 'close': {other_numeric_nulls}"
        )
    if int((df["volume"] < 0).sum()):
        raise PipelineError("Found rows with negative volume")
    if int((df[list(REQUIRED_COLUMNS[:-1])] <= 0).any(axis=1).sum()):
        raise PipelineError("Found rows with non-positive OHLC prices")

    invalid_ohlc = (
        (df["high"] < df[["open", "close", "low"]].max(axis=1))
        | (df["low"] > df[["open", "close", "high"]].min(axis=1))
        | (df["low"] > df["high"])
    )
    if int(invalid_ohlc.sum()):
        raise PipelineError("Found rows with inconsistent OHLC relationships")

    rows_valid = int(len(df))
    rows_dropped = rows_input - rows_valid
    min_rows = max(config.min_rows_required, config.window)
    if rows_valid < min_rows:
        raise PipelineError(
            "Not enough valid rows for processing: "
            f"rows_valid={rows_valid}, min_rows={min_rows}"
        )

    quality = {
        "rows_input": rows_input,
        "rows_valid": rows_valid,
        "rows_dropped": rows_dropped,
        "rows_used": max(rows_valid - config.window + 1, 0),
        "invalid_close_rows": invalid_close_rows,
        "warnings_count": int(invalid_close_rows > 0),
    }
    return df.reset_index(drop=True), quality


def build_metrics(
    dataframe: pd.DataFrame,
    config: PipelineConfig,
    context: RunContext,
    quality: dict[str, Any],
    latency_ms: int,
) -> dict[str, Any]:
    signal = dataframe["signal"].dropna()
    if signal.empty:
        raise PipelineError("No valid signals generated after feature computation")

    signal_rate = float(f"{signal.astype(float).mean():.4f}")
    coverage = float(f"{len(signal) / quality['rows_valid']:.4f}")
    data_quality_score = float(f"{quality['rows_valid'] / quality['rows_input']:.4f}")
    return {
        "status": "success",
        "version": config.version,
        "run_id": context.run_id,
        "dataset_hash": context.dataset_hash,
        "config_hash": context.config_hash,
        "metric": "signal_rate",
        "value": signal_rate,
        "signal_rate": signal_rate,
        "data_quality_score": data_quality_score,
        "coverage": coverage,
        "rows_input": quality["rows_input"],
        "rows_valid": quality["rows_valid"],
        "rows_dropped": quality["rows_dropped"],
        "rows_used": int(len(signal)),
        "warnings_count": quality["warnings_count"],
        "latency_ms": latency_ms,
    }


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("pipeline_app")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = JsonFormatter()
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def log_event(
    logger: logging.Logger,
    event: str,
    context: RunContext,
    stage: str,
    details: dict[str, Any],
    level: int = logging.INFO,
) -> None:
    logger.log(
        level,
        event,
        extra={
            "event": event,
            "run_id": context.run_id,
            "stage": stage,
            "details": details,
        },
    )


def write_json(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def failure_payload(
    *,
    version: str,
    run_id: str,
    dataset_hash: str,
    config_hash: str,
    error_type: str,
    error_message: str,
    retryable: bool,
) -> dict[str, Any]:
    return asdict(
        FailureRecord(
            status="error",
            version=version,
            run_id=run_id,
            dataset_hash=dataset_hash,
            config_hash=config_hash,
            error_type=error_type,
            error_message=error_message,
            retryable=retryable,
        )
    )


@dataclass(frozen=True)
class FailureRecord:
    status: str
    version: str
    run_id: str
    dataset_hash: str
    config_hash: str
    error_type: str
    error_message: str
    retryable: bool


def elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)
