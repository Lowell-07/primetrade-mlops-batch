from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pipeline_app.core import (
    PipelineExecutionError,
    load_config,
    read_csv,
    run_pipeline,
    validate_data,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data.csv"
CONFIG_PATH = ROOT / "config.yaml"


def test_validate_market_data_rejects_invalid_ohlc() -> None:
    config, _ = load_config(CONFIG_PATH)
    frame = pd.DataFrame(
        {
            "date": ["2023-01-01", "2023-01-02"],
            "open": [100.0, 101.0],
            "high": [99.0, 102.0],
            "low": [98.0, 100.0],
            "close": [100.5, 101.5],
            "volume": [1000, 1100],
        }
    )

    with pytest.raises(Exception, match="inconsistent OHLC relationships"):
        validate_data(frame, config)


def test_validate_market_data_tracks_dropped_close_rows() -> None:
    config, _ = load_config(CONFIG_PATH)
    frame = pd.DataFrame(
        {
            "date": [
                "2023-01-01",
                "2023-01-02",
                "2023-01-03",
                "2023-01-04",
                "2023-01-05",
            ],
            "open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.5, "bad", 102.5, 103.5, 104.5],
            "volume": [1000, 1100, 1200, 1300, 1400],
        }
    )
    permissive_config = config.__class__(
        version=config.version,
        window=4,
        min_rows_required=4,
        max_invalid_close_ratio=0.25,
        timestamp_column=config.timestamp_column,
    )

    result_df, result = validate_data(frame, permissive_config)

    assert result["rows_input"] == 5
    assert len(result_df) == 4
    assert result["rows_valid"] == 4
    assert result["rows_dropped"] == 1
    assert result["invalid_close_rows"] == 1


def test_pipeline_regression_matches_expected_metric(tmp_path: Path) -> None:
    output_path = tmp_path / "metrics.json"
    log_path = tmp_path / "run.log"

    metrics = run_pipeline(
        input_path=DATA_PATH,
        config_path=CONFIG_PATH,
        output_path=output_path,
        log_path=log_path,
    )

    assert metrics["status"] == "success"
    assert metrics["signal_rate"] == 0.5625
    assert metrics["rows_input"] == 20
    assert metrics["rows_valid"] == 20
    assert metrics["rows_dropped"] == 0
    assert metrics["rows_used"] == 16
    assert metrics["coverage"] == 0.8
    assert metrics["data_quality_score"] == 1.0
    assert output_path.exists()
    assert log_path.exists()


def test_pipeline_execution_writes_failure_payload(tmp_path: Path) -> None:
    bad_data = tmp_path / "bad.csv"
    bad_data.write_text(
        "\n".join(
            [
                "date,open,high,low,close,volume",
                "2023-01-01,100,99,98,100,1000",
                "2023-01-02,101,100,99,101,1000",
            ]
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "metrics.json"
    log_path = tmp_path / "run.log"

    with pytest.raises(PipelineExecutionError):
        run_pipeline(
            input_path=bad_data,
            config_path=CONFIG_PATH,
            output_path=output_path,
            log_path=log_path,
        )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["status"] == "error"
    assert payload["retryable"] is False


def test_ingest_csv_reads_fixture() -> None:
    frame = read_csv(DATA_PATH)
    assert list(frame.columns) == ["date", "open", "high", "low", "close", "volume"]
    assert len(frame) == 20
