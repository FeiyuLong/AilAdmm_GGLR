"""STOC-ADMM for graph-guided logistic regression."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
from scipy import sparse

from algorithms.common import (
    Evaluator,
    HistoryRecorder,
    RunResult,
    baseline_admm_step,
    validate_algorithm_inputs,
)
from utils.metrics import batch_gradient


def run_stoc_admm(
    X_train: sparse.spmatrix,
    labels_train: np.ndarray,
    X_test: sparse.spmatrix,
    labels_test: np.ndarray,
    D: sparse.spmatrix,
    f_star: float,
    config: dict[str, Any],
    seed: int,
    *,
    name: str = "STOC-ADMM",
) -> RunResult:
    n, d, q, step, rho = validate_algorithm_inputs(X_train, labels_train, D, config)
    mu = float(config["mu"])
    max_iter = int(config["max_iter"])
    batch_size = min(int(config["batch_size"]), n)
    rng = np.random.default_rng(seed)
    x, y, dual = np.zeros(d), np.zeros(q), np.zeros(q)
    evaluator = Evaluator(X_train, labels_train, X_test, labels_test, D, mu, f_star)
    recorder = HistoryRecorder(name, seed, evaluator, int(config["eval_every"]))
    algorithm_time, ifo = 0.0, 0
    recorder.record(0, algorithm_time, ifo, x, y, dual)

    for iteration in range(1, max_iter + 1):
        started = perf_counter()
        indices = rng.choice(n, size=batch_size, replace=False)
        estimate = batch_gradient(X_train, labels_train, x, indices)
        x, y, dual = baseline_admm_step(
            x,
            y,
            dual,
            estimate,
            D,
            mu=mu,
            rho=rho,
            step_size=step,
            algorithm_name=name,
            seed=seed,
            iteration=iteration,
        )
        algorithm_time += perf_counter() - started
        ifo += batch_size
        if recorder.should_record(iteration, max_iter):
            recorder.record(iteration, algorithm_time, ifo, x, y, dual)
    return recorder.finalize(x, y, dual, metadata={"batch_size": batch_size})
