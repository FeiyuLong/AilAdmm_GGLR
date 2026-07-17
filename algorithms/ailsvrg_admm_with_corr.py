"""AILSVRG-ADMM ablation that restores the legacy x correction."""

from __future__ import annotations

from typing import Any
import numpy as np
from scipy import sparse

from algorithms.ailsvrg_admm import _run_ailsvrg_core
from algorithms.common import RunResult


def run_ailsvrg_admm_with_corr(
    X_train: sparse.spmatrix, labels_train: np.ndarray, X_test: sparse.spmatrix,
    labels_test: np.ndarray, D: sparse.spmatrix, f_star: float,
    config: dict[str, Any], seed: int, *, name: str = "AILSVRG-ADMM-WithCorr"
) -> RunResult:
    adjusted = dict(config)
    adjusted["enable_correction"] = True
    return _run_ailsvrg_core(X_train, labels_train, X_test, labels_test, D, f_star, adjusted, seed, name=name)
