"""Serialization helpers for raw benchmark results and metadata."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from algorithms.common import RunResult, ensure_finite_state


CSV_FIELDS = (
    "iteration",
    "algorithm_time",
    "ifo_count",
    "feasible_objective",
    "split_objective",
    "optimality_gap",
    "primal_residual",
    "kkt_residual",
    "test_logistic_loss",
    "test_accuracy",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_run_csv(result: RunResult, output_dir: str | Path) -> Path:
    ensure_finite_state(
        result.name,
        result.seed,
        int(result.iteration[-1]) if result.iteration.size else -1,
        **{
            field: getattr(result, field)
            for field in CSV_FIELDS
            if field != "iteration"
        },
        x_final=result.x_final,
        y_final=result.y_final,
        dual_final=result.dual_final,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = result.name.lower().replace("-", "_")
    path = output_dir / f"{safe_name}_seed_{result.seed}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for index in range(result.iteration.size):
            writer.writerow(
                {field: getattr(result, field)[index] for field in CSV_FIELDS}
            )
    return path


def write_json(data: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            data,
            handle,
            ensure_ascii=False,
            indent=2,
            default=_json_default,
            allow_nan=False,
        )
    return path
