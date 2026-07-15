"""Accelerated SVRG-ADMM with the documented z-momentum update."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
from scipy import sparse

from algorithms.common import (
    Evaluator,
    HistoryRecorder,
    RunResult,
    ensure_finite_state,
    numerical_guard,
    validate_algorithm_inputs,
)
from utils.metrics import batch_gradient, soft_threshold


def run_asvrg_admm(
    X_train: sparse.spmatrix,
    labels_train: np.ndarray,
    X_test: sparse.spmatrix,
    labels_test: np.ndarray,
    D: sparse.spmatrix,
    f_star: float,
    config: dict[str, Any],
    seed: int,
    *,
    name: str = "ASVRG-ADMM",
) -> RunResult:
    n, d, q, step, rho = validate_algorithm_inputs(X_train, labels_train, D, config)
    mu, max_iter = float(config["mu"]), int(config["max_iter"])
    batch_size = min(int(config["batch_size"]), n)
    inner_iter = max(1, int(config["inner_iter"]))
    theta = float(config["theta"])
    if not 0.0 < theta <= 1.0:
        raise ValueError("ASVRG theta must belong to (0, 1].")

    rng = np.random.default_rng(seed)
    x, z = np.zeros(d), np.zeros(d)
    y, dual = np.zeros(q), np.zeros(q)
    snapshot, full_gradient = x.copy(), np.zeros(d)
    epoch_sum, epoch_count = np.zeros(d), 0
    evaluator = Evaluator(X_train, labels_train, X_test, labels_test, D, mu, f_star)
    recorder = HistoryRecorder(name, seed, evaluator, int(config["eval_every"]))
    algorithm_time, ifo = 0.0, 0
    recorder.record(0, algorithm_time, ifo, z, y, dual)

    for iteration in range(1, max_iter + 1):
        started = perf_counter()
        if (iteration - 1) % inner_iter == 0:
            if epoch_count > 0:
                snapshot = epoch_sum / epoch_count
            else:
                snapshot = x.copy()
            z = x.copy()
            full_gradient = batch_gradient(X_train, labels_train, snapshot)
            epoch_sum.fill(0.0)
            epoch_count = 0
            ifo += n

        with numerical_guard(name, seed, iteration):
            y = soft_threshold(D @ x + dual / rho, mu / rho)
            indices = rng.choice(n, size=batch_size, replace=False)
            estimate = (
                batch_gradient(X_train, labels_train, x, indices)
                - batch_gradient(X_train, labels_train, snapshot, indices)
                + full_gradient
            )
            z_gradient = estimate + rho * np.asarray(
                D.T @ (D @ z - y + dual / rho)
            ).reshape(-1)
            z = z - (step / theta) * z_gradient
            x = theta * z + (1.0 - theta) * snapshot
            dual = dual + rho * (np.asarray(D @ z).reshape(-1) - y)
            epoch_sum += x
            epoch_count += 1
        ensure_finite_state(
            name,
            seed,
            iteration,
            estimate=estimate,
            z_gradient=z_gradient,
            x=x,
            z=z,
            y=y,
            dual=dual,
            epoch_sum=epoch_sum,
        )

        algorithm_time += perf_counter() - started
        ifo += 2 * batch_size
        if recorder.should_record(iteration, max_iter):
            # z is the constrained primal variable used by the multiplier update.
            recorder.record(iteration, algorithm_time, ifo, z, y, dual)
    return recorder.finalize(
        z,
        y,
        dual,
        metadata={
            "batch_size": batch_size,
            "inner_iter": inner_iter,
            "theta": theta,
            "reported_primal": "z",
        },
    )
