"""SAGA-ADMM with an unbiased single-sample estimator."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
from scipy import sparse

from algorithms.common import Evaluator, HistoryRecorder, RunResult, baseline_admm_step, validate_algorithm_inputs
from utils.metrics import per_sample_gradients, sample_gradient


def run_saga_admm(
    X_train: sparse.spmatrix,
    labels_train: np.ndarray,
    X_test: sparse.spmatrix,
    labels_test: np.ndarray,
    D: sparse.spmatrix,
    f_star: float,
    config: dict[str, Any],
    seed: int,
    *,
    name: str = "SAGA-ADMM",
) -> RunResult:
    n, d, q, step, rho = validate_algorithm_inputs(X_train, labels_train, D, config)
    mu, max_iter = float(config["mu"]), int(config["max_iter"])
    rng = np.random.default_rng(seed)
    x, y, dual = np.zeros(d), np.zeros(q), np.zeros(q)
    evaluator = Evaluator(X_train, labels_train, X_test, labels_test, D, mu, f_star)
    recorder = HistoryRecorder(name, seed, evaluator, int(config["eval_every"]))
    algorithm_time, ifo = 0.0, 0
    recorder.record(0, algorithm_time, ifo, x, y, dual)

    started = perf_counter()
    table = per_sample_gradients(X_train, labels_train, x)
    average = table.mean(axis=0)
    algorithm_time += perf_counter() - started
    ifo += n

    for iteration in range(1, max_iter + 1):
        started = perf_counter()
        index = int(rng.integers(n))
        new_gradient = sample_gradient(X_train[index], labels_train[index], x)
        old_gradient = table[index].copy()
        estimate = new_gradient - old_gradient + average
        average += (new_gradient - old_gradient) / n
        table[index] = new_gradient
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
        ifo += 1
        if recorder.should_record(iteration, max_iter):
            recorder.record(iteration, algorithm_time, ifo, x, y, dual)
    return recorder.finalize(x, y, dual, metadata={"gradient_table_dtype": "float64"})
