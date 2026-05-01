from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def run_pipeline(
    tmp_path: Path, data_path: Path | None = None
) -> subprocess.CompletedProcess[str]:
    output_path = tmp_path / "metrics.json"
    log_path = tmp_path / "run.log"
    input_path = data_path or ROOT / "data.csv"
    return subprocess.run(
        [
            sys.executable,
            "run.py",
            "--input",
            str(input_path),
            "--config",
            str(ROOT / "config.yaml"),
            "--output",
            str(output_path),
            "--log-file",
            str(log_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_config_contains_only_required_fields() -> None:
    config = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    assert set(config.keys()) == {"seed", "window", "version"}


def test_success_output_matches_exact_schema(tmp_path: Path) -> None:
    result = run_pipeline(tmp_path)
    assert result.returncode == 0

    payload = json.loads(result.stdout)
    assert set(payload.keys()) == {
        "version",
        "rows_processed",
        "metric",
        "value",
        "latency_ms",
        "seed",
        "status",
    }
    assert payload["version"] == "v1"
    assert payload["metric"] == "signal_rate"
    assert payload["seed"] == 42
    assert payload["status"] == "success"


def test_error_output_matches_exact_schema(tmp_path: Path) -> None:
    bad_data = tmp_path / "bad.csv"
    bad_data.write_text("open,high,low,volume\n1,2,0,100\n", encoding="utf-8")

    result = run_pipeline(tmp_path, bad_data)
    assert result.returncode == 1

    payload = json.loads(result.stdout)
    assert set(payload.keys()) == {"version", "status", "error_message"}
    assert payload["version"] == "v1"
    assert payload["status"] == "error"
    assert isinstance(payload["error_message"], str)
