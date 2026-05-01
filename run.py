#!/usr/bin/env python3
"""
Minimal MLOps batch pipeline for trading signal generation.

Computes rolling mean and binary signal from OHLCV data.
Designed for deterministic, observable, Dockerized execution.
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yaml


def setup_logging(log_file: Optional[str] = None) -> logging.Logger:
    """
    Configure structured logging to console and optional file.
    
    Args:
        log_file: Path to log file. If None, logs only to console.
        
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("mlops_pipeline")
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate handlers if called multiple times
    if logger.handlers:
        logger.handlers.clear()
    
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: str, logger: logging.Logger) -> Dict[str, Any]:
    """
    Load and validate YAML configuration.
    
    Required fields:
        - seed (int): Random seed for determinism.
        - window (int > 0): Rolling window size.
        - version (str): Pipeline version identifier.
        
    Args:
        config_path: Path to YAML config file.
        logger: Logger instance.
        
    Returns:
        Validated config dictionary.
        
    Raises:
        FileNotFoundError: If config file does not exist.
        ValueError: If required fields are missing or invalid.
        TypeError: If field types are incorrect.
    """
    logger.info(f"Loading config from: {config_path}")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    if config is None:
        raise ValueError("Config file is empty or invalid YAML")
    
    required_fields = ["seed", "window", "version"]
    missing = [f for f in required_fields if f not in config]
    if missing:
        raise ValueError(f"Missing required config fields: {missing}")
    
    # Type and value validation
    if not isinstance(config["seed"], int):
        raise TypeError(f"Config 'seed' must be int, got {type(config['seed']).__name__}")
    
    if not isinstance(config["window"], int) or config["window"] <= 0:
        raise ValueError(f"Config 'window' must be int > 0, got {config['window']}")
    
    if not isinstance(config["version"], str):
        raise TypeError(f"Config 'version' must be str, got {type(config['version']).__name__}")
    
    logger.info(
        f"Config validated: seed={config['seed']}, window={config['window']}, "
        f"version={config['version']}"
    )
    return config


def validate_data(input_path: str, logger: logging.Logger) -> pd.DataFrame:
    """
    Load and validate input CSV data.
    
    Validates:
        - File exists
        - File is non-empty
        - Valid CSV format
        - Contains required 'close' column
        - Contains at least one data row
        
    Args:
        input_path: Path to input CSV file.
        logger: Logger instance.
        
    Returns:
        Validated DataFrame.
        
    Raises:
        FileNotFoundError: If input file does not exist.
        ValueError: If file is empty, invalid CSV, or missing 'close' column.
    """
    logger.info(f"Loading data from: {input_path}")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if os.path.getsize(input_path) == 0:
        raise ValueError(f"Input file is empty: {input_path}")
    
    try:
        df = pd.read_csv(input_path)
    except pd.errors.EmptyDataError:
        raise ValueError(f"Input file has no data: {input_path}")
    except pd.errors.ParserError as e:
        raise ValueError(f"Invalid CSV format: {e}")
    except Exception as e:
        raise ValueError(f"Failed to read CSV: {e}")
    
    if df.empty:
        raise ValueError("Input CSV has no rows")
    
    if "close" not in df.columns:
        raise ValueError(
            f"Missing required column 'close'. Available columns: {list(df.columns)}"
        )
    
    logger.info(f"Data loaded: {len(df)} rows, columns={list(df.columns)}")
    return df


def compute_features(df: pd.DataFrame, window: int, logger: logging.Logger) -> pd.DataFrame:
    """
    Compute rolling mean on the 'close' column.
    
    The first (window - 1) rows will have NaN values for rolling_mean,
    which is the expected and correct behavior.
    
    Args:
        df: Input DataFrame with 'close' column.
        window: Rolling window size.
        logger: Logger instance.
        
    Returns:
        DataFrame with added 'rolling_mean' column.
    """
    logger.info(f"Computing rolling mean with window={window}")
    
    df = df.copy()
    df["rolling_mean"] = df["close"].rolling(window=window, min_periods=window).mean()
    
    nan_count = df["rolling_mean"].isna().sum()
    logger.info(f"Rolling mean computed. NaN count: {nan_count}")
    
    return df


def generate_signals(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Generate binary signal: 1 if close > rolling_mean, else 0.
    
    Signal is explicitly set to NaN where rolling_mean is NaN
    to ensure correct metric computation.
    
    Args:
        df: DataFrame with 'close' and 'rolling_mean' columns.
        logger: Logger instance.
        
    Returns:
        DataFrame with added 'signal' column.
    """
    logger.info("Generating signals")
    
    df = df.copy()
    # Initialize with NaN to ensure rows with invalid rolling_mean are excluded
    df["signal"] = np.nan
    
    valid_mask = df["rolling_mean"].notna()
    df.loc[valid_mask, "signal"] = (
        df.loc[valid_mask, "close"] > df.loc[valid_mask, "rolling_mean"]
    ).astype(int)
    
    valid_signals = df["signal"].notna().sum()
    logger.info(f"Signals generated: {valid_signals} valid signals")
    
    return df


def compute_metrics(
    df: pd.DataFrame,
    config: Dict[str, Any],
    latency_ms: int,
    logger: logging.Logger
) -> Dict[str, Any]:
    """
    Compute final pipeline metrics.
    
    Args:
        df: DataFrame with 'signal' column.
        config: Validated config dictionary.
        latency_ms: Total execution time in milliseconds.
        logger: Logger instance.
        
    Returns:
        Metrics dictionary for JSON output.
    """
    logger.info("Computing metrics")
    
    # mean() automatically excludes NaN values
    signal_rate = float(df["signal"].mean())
    rows_processed = int(len(df))
    
    metrics = {
        "version": config["version"],
        "rows_processed": rows_processed,
        "metric": "signal_rate",
        "value": round(signal_rate, 4),
        "latency_ms": latency_ms,
        "seed": config["seed"],
        "status": "success"
    }
    
    logger.info(
        f"Metrics: rows={rows_processed}, signal_rate={signal_rate:.4f}, "
        f"latency={latency_ms}ms"
    )
    return metrics


def write_metrics(metrics: Dict[str, Any], output_path: str, logger: logging.Logger) -> None:
    """
    Write metrics dictionary to JSON file.
    
    Args:
        metrics: Metrics dictionary.
        output_path: Path to output JSON file.
        logger: Logger instance.
    """
    logger.info(f"Writing metrics to: {output_path}")
    
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    
    logger.info("Metrics written successfully")


def main() -> None:
    """Main entry point for the batch pipeline."""
    parser = argparse.ArgumentParser(
        description="MLOps Batch Pipeline: Rolling Mean Signal Generator"
    )
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--output", required=True, help="Path to output metrics JSON")
    parser.add_argument("--log-file", required=True, help="Path to log file")
    args = parser.parse_args()
    
    start_time = time.perf_counter()
    logger = setup_logging(args.log_file)
    
    logger.info("=" * 60)
    logger.info("Job started")
    logger.info(
        f"CLI args: input={args.input}, config={args.config}, "
        f"output={args.output}, log={args.log_file}"
    )
    
    # Default error metrics (version fallback if config fails to load)
    config: Optional[Dict[str, Any]] = None
    error_metrics: Dict[str, Any] = {
        "version": "v1",
        "status": "error",
        "error_message": "Unknown error"
    }
    
    try:
        # Step 1: Load and validate config
        config = load_config(args.config, logger)
        error_metrics["version"] = config["version"]
        
        # Step 2: Set seed for determinism
        np.random.seed(config["seed"])
        logger.info(f"Random seed set to {config['seed']}")
        
        # Step 3: Load and validate data
        df = validate_data(args.input, logger)
        
        # Step 4: Compute features
        df = compute_features(df, config["window"], logger)
        
        # Step 5: Generate signals
        df = generate_signals(df, logger)
        
        # Step 6: Compute latency and metrics
        latency_ms = int((time.perf_counter() - start_time) * 1000)
        metrics = compute_metrics(df, config, latency_ms, logger)
        
        # Step 7: Write output
        write_metrics(metrics, args.output, logger)
        
        # Print to stdout for Docker/container observability
        print(json.dumps(metrics, indent=2))
        
        logger.info("Job completed successfully")
        logger.info("=" * 60)
        sys.exit(0)
        
    except Exception as e:
        logger.error(f"Job failed: {e}", exc_info=True)
        
        error_metrics["error_message"] = str(e)
        
        # Attempt to write error metrics even on failure
        try:
            write_metrics(error_metrics, args.output, logger)
            print(json.dumps(error_metrics, indent=2))
        except Exception as write_err:
            logger.critical(f"Failed to write error metrics: {write_err}")
            # Last resort: print to stdout so something is visible
            print(json.dumps(error_metrics, indent=2))
            sys.exit(1)
        
        logger.info("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()