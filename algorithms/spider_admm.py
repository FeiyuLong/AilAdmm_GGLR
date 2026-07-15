"""SPIDER-ADMM with periodic full-gradient refreshes."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
from scipy import sparse

from algorithms.common import Evaluator, HistoryRecorder, RunResult, baseline_admm_step, validate_algorithm_inputs
from utils.metrics import batch_gradient


def run_spider_admm(
    X_train: sparse.spmatrix,
    labels_train: np.ndarray,
    X_test: sparse.spmatrix,
    labels_test: np.ndarray,
    D: sparse.spmatrix,
    f_star: float,
    config: dict[str, Any],
    seed: int,
    *,
    name: str = "SPIDER-ADMM",
) -> RunResult:
    n, d, q_edges, step, rho = validate_algorithm_inputs(X_train, labels_train, D, config)
    mu, max_iter = float(config["mu"]), int(config["max_iter"])
    batch_size = min(int(config["batch_size"]), n)
    refresh_period = max(1, int(config["refresh_period"]))
    rng = np.random.default_rng(seed)
    x, x_previous = np.zeros(d), np.zeros(d)
    y, dual = np.zeros(q_edges), np.zeros(q_edges)
    estimate = np.zeros(d)
    evaluator = Evaluator(X_train, labels_train, X_test, labels_test, D, mu, f_star)
    recorder = HistoryRecorder(name, seed, evaluator, int(config["eval_every"]))
    algorithm_time, ifo = 0.0, 0
    recorder.record(0, algorithm_time, ifo, x, y, dual)

    for iteration in range(1, max_iter + 1):
        started = perf_counter()
        if (iteration - 1) % refresh_period == 0:
            estimate = batch_gradient(X_train, labels_train, x)
            ifo += n
        else:
            indices = rng.choice(n, size=batch_size, replace=False)
            estimate = (
                estimate
                + batch_gradient(X_train, labels_train, x, indices)
                - batch_gradient(X_train, labels_train, x_previous, indices)
            )
            ifo += 2 * batch_size
        old_x = x.copy()
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
        x_previous = old_x
        algorithm_time += perf_counter() - started
        if recorder.should_record(iteration, max_iter):
            recorder.record(iteration, algorithm_time, ifo, x, y, dual)
    return recorder.finalize(
        x,
        y,
        dual,
        metadata={"batch_size": batch_size, "refresh_period": refresh_period},
    )
