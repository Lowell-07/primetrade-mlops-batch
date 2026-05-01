#!/usr/bin/env python3
"""CLI entrypoint for the production batch pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pipeline_app.core import PipelineExecutionError, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Production MLOps Batch Pipeline: Rolling Mean Signal Generator"
    )
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--output", required=True, help="Path to output metrics JSON")
    parser.add_argument("--log-file", required=True, help="Path to log file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        metrics = run_pipeline(
            input_path=Path(args.input),
            config_path=Path(args.config),
            output_path=Path(args.output),
            log_path=Path(args.log_file),
        )
        print(json.dumps(metrics, indent=2))
        sys.exit(0)
    except PipelineExecutionError as exc:
        print(json.dumps(exc.error_payload, indent=2))
        sys.exit(1)


if __name__ == "__main__":
    main()
