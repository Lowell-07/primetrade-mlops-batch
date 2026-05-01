#!/usr/bin/env python3
"""
MLOps Batch Pipeline: Rolling Mean Signal Generator.

Deterministic, observable, Docker-ready batch job for OHLCV data.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="MLOps Batch Pipeline: Rolling Mean Signal Generator"
    )
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--output", required=True, help="Path to output metrics JSON")
    parser.add_argument("--log-file", required=True, help="Path to log file")
    return parser.parse_args()


def setup_logger(log_path: Path) -> logging.Logger:
    """Configure structured logging to file."""
    logger = logging.getLogger("mlops_pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(handler)
    return logger


def load_config(config_path: Path, logger: logging.Logger) -> dict[str, Any]:
    """Load and validate YAML configuration."""
    logger.info("Loading config from %s", config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ValueError("Config must be a YAML mapping")

    expected_keys = {"seed", "window", "version"}
    actual_keys = set(config.keys())
    if actual_keys != expected_keys:
        raise ValueError(
            f"Config must contain exactly: {sorted(expected_keys)}. "
            f"Found: {sorted(actual_keys)}"
        )

    if not isinstance(config["seed"], int):
        raise TypeError("Config 'seed' must be an integer")
    if not isinstance(config["window"], int) or config["window"] <= 0:
        raise ValueError("Config 'window' must be an integer > 0")
    if not isinstance(config["version"], str) or not config["version"]:
        raise TypeError("Config 'version' must be a non-empty string")

    logger.info(
        "Config loaded: version=%s, seed=%s, window=%s",
        config["version"],
        config["seed"],
        config["window"],
    )
    return config


def load_data(input_path: Path, window: int, logger: logging.Logger) -> pd.DataFrame:
    """Load and validate input CSV with comprehensive edge case handling."""
    logger.info("Loading data from %s", input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.stat().st_size == 0:
        raise ValueError(f"Input file is empty: {input_path}")

    try:
        df = pd.read_csv(input_path)
    except pd.errors.EmptyDataError:
        raise ValueError(f"Input file has no data: {input_path}")
    except pd.errors.ParserError as exc:
        raise ValueError(f"Invalid CSV format: {exc}")
    except Exception as exc:
        raise ValueError(f"Failed to read CSV: {exc}")

    if "close" not in df.columns:
        raise ValueError(f"Missing required column 'close'. Found: {list(df.columns)}")

    if df.empty:
        raise ValueError("Input CSV has no rows")

    # Validate close is numeric
    if not pd.api.types.is_numeric_dtype(df["close"]):
        raise TypeError(f"'close' must be numeric, got {df['close'].dtype}")

    # Check for Inf values
    if np.isinf(df["close"]).any():
        raise ValueError("'close' contains Inf values")

    # Check window vs dataset size
    if len(df) < window:
        raise ValueError(f"Dataset has {len(df)} rows but window requires {window}")

    logger.info("Dataset loaded: rows=%s, columns=%s", len(df), list(df.columns))
    return df


def compute_pipeline(
    df: pd.DataFrame, window: int, logger: logging.Logger
) -> pd.DataFrame:
    """Compute rolling mean and binary signal."""
    result = df.copy()

    result["rolling_mean"] = (
        result["close"].rolling(window=window, min_periods=window).mean()
    )
    logger.info("Rolling mean computed (window=%s)", window)

    # Signal: 1 if close > rolling_mean, else 0
    # Explicitly NaN where rolling_mean is NaN
    result["signal"] = np.nan
    valid_mask = result["rolling_mean"].notna()
    result.loc[valid_mask, "signal"] = (
        result.loc[valid_mask, "close"] > result.loc[valid_mask, "rolling_mean"]
    ).astype(int)

    valid_count = result["signal"].notna().sum()
    logger.info("Signal computed: %s valid signals", valid_count)
    return result


def build_metrics(
    df: pd.DataFrame,
    version: str,
    seed: int,
    latency_ms: int,
    logger: logging.Logger,
) -> dict[str, Any]:
    """Compute final metrics with deterministic output."""
    signal_rate = df["signal"].mean()
    if pd.isna(signal_rate):
        raise ValueError("No valid signals generated")

    metrics = {
        "version": version,
        "rows_processed": int(len(df)),
        "metric": "signal_rate",
        "value": round(float(signal_rate), 4),
        "latency_ms": latency_ms,
        "seed": seed,
        "status": "success",
    }
    logger.info(
        "Metrics: rows=%s, signal_rate=%.4f, latency=%sms",
        metrics["rows_processed"],
        metrics["value"],
        latency_ms,
    )
    return metrics


def write_json(output_path: Path, payload: dict[str, Any]) -> None:
    """Write payload to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    """Main entry point."""
    args = parse_args()
    start = time.perf_counter()
    logger = setup_logger(Path(args.log_file))

    logger.info("=" * 60)
    logger.info("Job started")

    error_payload = {
        "version": "v1",
        "status": "error",
        "error_message": "Unknown error",
    }

    try:
        config = load_config(Path(args.config), logger)
        error_payload["version"] = config["version"]

        # Seed reserved for future stochastic operations
        np.random.seed(config["seed"])

        df = load_data(Path(args.input), config["window"], logger)
        df = compute_pipeline(df, config["window"], logger)

        latency_ms = int((time.perf_counter() - start) * 1000)
        metrics = build_metrics(
            df,
            config["version"],
            config["seed"],
            latency_ms,
            logger,
        )
        write_json(Path(args.output), metrics)
        logger.info("Metrics written to %s", args.output)
        logger.info("Job completed successfully")
        logger.info("=" * 60)

        # Print to stdout for Docker/container visibility
        print(json.dumps(metrics, indent=2))
        sys.exit(0)

    except Exception as exc:
        logger.exception("Job failed: %s", exc)
        error_payload["error_message"] = str(exc)
        write_json(Path(args.output), error_payload)
        logger.info("Error metrics written to %s", args.output)
        logger.info("Job ended with error")
        logger.info("=" * 60)

        print(json.dumps(error_payload, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
