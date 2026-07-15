"""AIL-SVRG-ADMM and its shared ablation core."""

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


def resolve_p_min(setting: Any, *, n: int, batch_size: int) -> float:
    """Resolve a dataset-dependent or explicit snapshot-probability floor."""
    if n <= 0:
        raise ValueError("p_min resolution requires n to be positive.")
    if batch_size <= 0:
        raise ValueError("p_min resolution requires batch_size to be positive.")

    if isinstance(setting, str):
        normalized = setting.strip().lower()
        if normalized == "inverse_n":
            value = 1.0 / n
        elif normalized == "batch_over_n":
            value = min(1.0, batch_size / n)
        else:
            raise ValueError(
                "p_min must be 'inverse_n', 'batch_over_n', or a numeric value in (0, 1]."
            )
    else:
        if isinstance(setting, (bool, np.bool_)):
            raise ValueError("p_min must not be a boolean value.")
        try:
            value = float(setting)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "p_min must be 'inverse_n', 'batch_over_n', or a numeric value in (0, 1]."
            ) from exc

    if not np.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError("p_min must be finite and belong to (0, 1].")
    return float(value)


def adaptive_probability(iteration_zero_based: int, *, varrho: float, p_min: float) -> float:
    if iteration_zero_based < 0:
        raise ValueError("iteration_zero_based must be nonnegative.")
    return max(p_min, varrho / (iteration_zero_based + 1.0 + varrho))


def cost_matched_probability(max_iter: int, *, varrho: float, p_min: float) -> float:
    probabilities = [
        adaptive_probability(t, varrho=varrho, p_min=p_min) for t in range(max_iter)
    ]
    return float(np.mean(probabilities))


def _run_ail_svrg_core(
    X_train: sparse.spmatrix,
    labels_train: np.ndarray,
    X_test: sparse.spmatrix,
    labels_test: np.ndarray,
    D: sparse.spmatrix,
    f_star: float,
    config: dict[str, Any],
    seed: int,
    *,
    name: str,
) -> RunResult:
    n, d, q, step, rho = validate_algorithm_inputs(X_train, labels_train, D, config)
    mu, max_iter = float(config["mu"]), int(config["max_iter"])
    batch_size = min(int(config["batch_size"]), n)
    tau = float(config.get("tau", 0.5))
    varrho = float(config.get("varrho", 0.3))
    beta_y = float(config.get("beta_y", 0.0))
    enable_correction = bool(config.get("enable_correction", False))
    p_min_setting = config.get("p_min", "inverse_n")
    p_min = resolve_p_min(p_min_setting, n=n, batch_size=batch_size)
    if not 0.0 <= tau < 1.0:
        raise ValueError("AIL-SVRG-ADMM tau must belong to [0, 1).")
    if varrho <= 0.0 or beta_y < 0.0:
        raise ValueError("AIL-SVRG-ADMM varrho must be positive and beta_y nonnegative.")

    fixed_setting = config.get("fixed_probability")
    if fixed_setting == "cost_matched":
        fixed_probability = cost_matched_probability(
            max_iter, varrho=varrho, p_min=p_min
        )
    elif fixed_setting is None:
        fixed_probability = None
    else:
        fixed_probability = float(fixed_setting)
        if not 0.0 < fixed_probability <= 1.0:
            raise ValueError("fixed_probability must belong to (0, 1].")

    rng = np.random.default_rng(seed)
    x_previous, x, snapshot = np.zeros(d), np.zeros(d), np.zeros(d)
    y, dual = np.zeros(q), np.zeros(q)
    evaluator = Evaluator(X_train, labels_train, X_test, labels_test, D, mu, f_star)
    recorder = HistoryRecorder(name, seed, evaluator, int(config["eval_every"]))
    algorithm_time, ifo, refreshes = 0.0, 0, 0
    recorder.record(0, algorithm_time, ifo, x, y, dual)

    started = perf_counter()
    full_gradient = batch_gradient(X_train, labels_train, snapshot)
    algorithm_time += perf_counter() - started
    ifo += n

    for iteration in range(1, max_iter + 1):
        started = perf_counter()
        with numerical_guard(name, seed, iteration):
            x_hat = x + tau * (x - x_previous)
            indices = rng.choice(n, size=batch_size, replace=False)
            estimate = (
                batch_gradient(X_train, labels_train, x_hat, indices)
                - batch_gradient(X_train, labels_train, snapshot, indices)
                + full_gradient
            )
            predictor_gradient = estimate + rho * np.asarray(
                D.T @ (D @ x - y + dual / rho)
            ).reshape(-1)
            x_prediction = x - step * predictor_gradient
            y_new = soft_threshold(
                (rho * np.asarray(D @ x_prediction).reshape(-1) + dual + beta_y * y)
                / (rho + beta_y),
                mu / (rho + beta_y),
            )
            if enable_correction:
                x_new = x_prediction + step * rho * np.asarray(
                    D.T @ (y_new - y)
                ).reshape(-1)
            else:
                x_new = x_prediction
            dual_new = dual + rho * (np.asarray(D @ x_new).reshape(-1) - y_new)

            probability = (
                fixed_probability
                if fixed_probability is not None
                else adaptive_probability(
                    iteration - 1, varrho=varrho, p_min=p_min
                )
            )
            if rng.random() < probability:
                snapshot = x_new.copy()
                full_gradient = batch_gradient(X_train, labels_train, snapshot)
                ifo += n
                refreshes += 1

        ensure_finite_state(
            name,
            seed,
            iteration,
            x_hat=x_hat,
            estimate=estimate,
            predictor_gradient=predictor_gradient,
            x_prediction=x_prediction,
            x=x_new,
            y=y_new,
            dual=dual_new,
            snapshot=snapshot,
            full_gradient=full_gradient,
        )

        x_previous, x = x, x_new
        y, dual = y_new, dual_new
        algorithm_time += perf_counter() - started
        ifo += 2 * batch_size
        if recorder.should_record(iteration, max_iter):
            recorder.record(iteration, algorithm_time, ifo, x, y, dual)

    return recorder.finalize(
        x,
        y,
        dual,
        metadata={
            "batch_size": batch_size,
            "tau": tau,
            "varrho": varrho,
            "beta_y": beta_y,
            "p_min_setting": p_min_setting,
            "p_min": p_min,
            "enable_correction": enable_correction,
            "fixed_probability": fixed_probability,
            "snapshot_refreshes": refreshes,
        },
    )


def run_ail_svrg_admm(
    X_train: sparse.spmatrix,
    labels_train: np.ndarray,
    X_test: sparse.spmatrix,
    labels_test: np.ndarray,
    D: sparse.spmatrix,
    f_star: float,
    config: dict[str, Any],
    seed: int,
    *,
    name: str = "AIL-SVRG-ADMM",
) -> RunResult:
    return _run_ail_svrg_core(
        X_train, labels_train, X_test, labels_test, D, f_star, config, seed, name=name
    )
