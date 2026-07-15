"""AIL-SVRG-ADMM ablation that restores the legacy x correction."""

from __future__ import annotations

from typing import Any
import numpy as np
from scipy import sparse

from algorithms.ail_svrg_admm import _run_ail_svrg_core
from algorithms.common import RunResult


def run_ail_svrg_admm_with_corr(
    X_train: sparse.spmatrix, labels_train: np.ndarray, X_test: sparse.spmatrix,
    labels_test: np.ndarray, D: sparse.spmatrix, f_star: float,
    config: dict[str, Any], seed: int, *, name: str = "AIL-SVRG-ADMM-WithCorr"
) -> RunResult:
    adjusted = dict(config)
    adjusted["enable_correction"] = True
    return _run_ail_svrg_core(X_train, labels_train, X_test, labels_test, D, f_star, adjusted, seed, name=name)
