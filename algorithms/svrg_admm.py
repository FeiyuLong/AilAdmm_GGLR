"""Epoch-based SVRG-ADMM."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
from scipy import sparse

from algorithms.common import Evaluator, HistoryRecorder, RunResult, baseline_admm_step, validate_algorithm_inputs
from utils.metrics import batch_gradient


def run_svrg_admm(
    X_train: sparse.spmatrix,
    labels_train: np.ndarray,
    X_test: sparse.spmatrix,
    labels_test: np.ndarray,
    D: sparse.spmatrix,
    f_star: float,
    config: dict[str, Any],
    seed: int,
    *,
    name: str = "SVRG-ADMM",
) -> RunResult:
    n, d, q, step, rho = validate_algorithm_inputs(X_train, labels_train, D, config)
    mu, max_iter = float(config["mu"]), int(config["max_iter"])
    batch_size = min(int(config["batch_size"]), n)
    inner_iter = max(1, int(config["inner_iter"]))
    rng = np.random.default_rng(seed)
    x, y, dual = np.zeros(d), np.zeros(q), np.zeros(q)
    snapshot, full_gradient = x.copy(), np.zeros(d)
    evaluator = Evaluator(X_train, labels_train, X_test, labels_test, D, mu, f_star)
    recorder = HistoryRecorder(name, seed, evaluator, int(config["eval_every"]))
    algorithm_time, ifo = 0.0, 0
    recorder.record(0, algorithm_time, ifo, x, y, dual)

    for iteration in range(1, max_iter + 1):
        started = perf_counter()
        if (iteration - 1) % inner_iter == 0:
            snapshot = x.copy()
            full_gradient = batch_gradient(X_train, labels_train, snapshot)
            ifo += n
        indices = rng.choice(n, size=batch_size, replace=False)
        estimate = (
            batch_gradient(X_train, labels_train, x, indices)
            - batch_gradient(X_train, labels_train, snapshot, indices)
            + full_gradient
        )
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
        ifo += 2 * batch_size
        if recorder.should_record(iteration, max_iter):
            recorder.record(iteration, algorithm_time, ifo, x, y, dual)
    return recorder.finalize(
        x, y, dual, metadata={"batch_size": batch_size, "inner_iter": inner_iter}
    )
