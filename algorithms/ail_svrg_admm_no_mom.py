"""AIL-SVRG-ADMM ablation without inertial extrapolation."""

from __future__ import annotations

from typing import Any
import numpy as np
from scipy import sparse

from algorithms.ail_svrg_admm import _run_ail_svrg_core
from algorithms.common import RunResult


def run_ail_svrg_admm_no_mom(
    X_train: sparse.spmatrix, labels_train: np.ndarray, X_test: sparse.spmatrix,
    labels_test: np.ndarray, D: sparse.spmatrix, f_star: float,
    config: dict[str, Any], seed: int, *, name: str = "AIL-SVRG-ADMM-NoMom"
) -> RunResult:
    adjusted = dict(config)
    adjusted["tau"] = 0.0
    return _run_ail_svrg_core(X_train, labels_train, X_test, labels_test, D, f_star, adjusted, seed, name=name)
