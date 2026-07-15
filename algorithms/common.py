"""Shared algorithm state, timing, evaluation, and ADMM operations."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np
from scipy import sparse

from utils.metrics import (
    classification_accuracy,
    feasible_objective,
    kkt_components,
    logistic_loss,
    soft_threshold,
    split_objective,
)


class NumericalDivergenceError(RuntimeError):
    """Raised as soon as an algorithm creates a non-finite numerical state."""


def _location(algorithm_name: str, seed: int, iteration: int) -> str:
    return f"{algorithm_name} seed={seed} iteration={iteration}"


def ensure_finite_state(
    algorithm_name: str,
    seed: int,
    iteration: int,
    **values: np.ndarray | float,
) -> None:
    invalid = [
        key for key, value in values.items() if not np.isfinite(np.asarray(value)).all()
    ]
    if invalid:
        names = ", ".join(invalid)
        raise NumericalDivergenceError(
            f"{_location(algorithm_name, seed, iteration)} produced non-finite values: {names}"
        )


@contextmanager
def numerical_guard(
    algorithm_name: str, seed: int, iteration: int
) -> Iterator[None]:
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            yield
    except NumericalDivergenceError:
        raise
    except (FloatingPointError, OverflowError) as exc:
        raise NumericalDivergenceError(
            f"{_location(algorithm_name, seed, iteration)} failed numerically: {exc}"
        ) from exc


@dataclass
class RunResult:
    name: str
    seed: int
    iteration: np.ndarray
    algorithm_time: np.ndarray
    ifo_count: np.ndarray
    feasible_objective: np.ndarray
    split_objective: np.ndarray
    optimality_gap: np.ndarray
    primal_residual: np.ndarray
    kkt_residual: np.ndarray
    test_logistic_loss: np.ndarray
    test_accuracy: np.ndarray
    x_final: np.ndarray
    y_final: np.ndarray
    dual_final: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def metric(self, name: str) -> np.ndarray:
        return np.asarray(getattr(self, name), dtype=np.float64)


class Evaluator:
    def __init__(
        self,
        X_train: sparse.spmatrix,
        labels_train: np.ndarray,
        X_test: sparse.spmatrix,
        labels_test: np.ndarray,
        D: sparse.spmatrix,
        mu: float,
        f_star: float,
        *,
        zero_tolerance: float = 1.0e-12,
    ) -> None:
        self.X_train = X_train
        self.labels_train = labels_train
        self.X_test = X_test
        self.labels_test = labels_test
        self.D = D
        self.mu = float(mu)
        self.f_star = float(f_star)
        self.zero_tolerance = float(zero_tolerance)

    def evaluate(self, x: np.ndarray, y: np.ndarray, dual: np.ndarray) -> dict[str, float]:
        feasible = feasible_objective(
            self.X_train, self.labels_train, self.D, x, self.mu
        )
        split = split_objective(self.X_train, self.labels_train, x, y, self.mu)
        components = kkt_components(
            self.X_train,
            self.labels_train,
            self.D,
            x,
            y,
            dual,
            self.mu,
            zero_tolerance=self.zero_tolerance,
        )
        metrics = {
            "feasible_objective": feasible,
            "split_objective": split,
            "optimality_gap": feasible - self.f_star,
            "primal_residual": components.primal_norm,
            "kkt_residual": components.squared_residual,
            "test_logistic_loss": logistic_loss(
                self.X_test, self.labels_test, x
            ),
            "test_accuracy": classification_accuracy(
                self.X_test, self.labels_test, x
            ),
        }
        return metrics


class HistoryRecorder:
    def __init__(self, name: str, seed: int, evaluator: Evaluator, eval_every: int) -> None:
        self.name = name
        self.seed = int(seed)
        self.evaluator = evaluator
        self.eval_every = max(1, int(eval_every))
        self._values: dict[str, list[float]] = {
            "iteration": [],
            "algorithm_time": [],
            "ifo_count": [],
            "feasible_objective": [],
            "split_objective": [],
            "optimality_gap": [],
            "primal_residual": [],
            "kkt_residual": [],
            "test_logistic_loss": [],
            "test_accuracy": [],
        }

    def should_record(self, iteration: int, max_iter: int) -> bool:
        return iteration == 0 or iteration == max_iter or iteration % self.eval_every == 0

    def record(
        self,
        iteration: int,
        algorithm_time: float,
        ifo_count: int,
        x: np.ndarray,
        y: np.ndarray,
        dual: np.ndarray,
    ) -> None:
        ensure_finite_state(self.name, self.seed, iteration, x=x, y=y, dual=dual)
        with numerical_guard(self.name, self.seed, iteration):
            metrics = self.evaluator.evaluate(x, y, dual)
        ensure_finite_state(self.name, self.seed, iteration, **metrics)
        self._values["iteration"].append(float(iteration))
        self._values["algorithm_time"].append(float(algorithm_time))
        self._values["ifo_count"].append(float(ifo_count))
        for key, value in metrics.items():
            self._values[key].append(float(value))

    def finalize(
        self,
        x: np.ndarray,
        y: np.ndarray,
        dual: np.ndarray,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> RunResult:
        final_iteration = int(self._values["iteration"][-1]) if self._values["iteration"] else -1
        ensure_finite_state(
            self.name, self.seed, final_iteration, x=x, y=y, dual=dual
        )
        return RunResult(
            name=self.name,
            seed=self.seed,
            iteration=np.asarray(self._values["iteration"], dtype=np.int64),
            algorithm_time=np.asarray(self._values["algorithm_time"], dtype=np.float64),
            ifo_count=np.asarray(self._values["ifo_count"], dtype=np.int64),
            feasible_objective=np.asarray(self._values["feasible_objective"]),
            split_objective=np.asarray(self._values["split_objective"]),
            optimality_gap=np.asarray(self._values["optimality_gap"]),
            primal_residual=np.asarray(self._values["primal_residual"]),
            kkt_residual=np.asarray(self._values["kkt_residual"]),
            test_logistic_loss=np.asarray(self._values["test_logistic_loss"]),
            test_accuracy=np.asarray(self._values["test_accuracy"]),
            x_final=np.asarray(x, dtype=np.float64).copy(),
            y_final=np.asarray(y, dtype=np.float64).copy(),
            dual_final=np.asarray(dual, dtype=np.float64).copy(),
            metadata=dict(metadata or {}),
        )


def baseline_admm_step(
    x: np.ndarray,
    y: np.ndarray,
    dual: np.ndarray,
    gradient_estimate: np.ndarray,
    D: sparse.spmatrix,
    *,
    mu: float,
    rho: float,
    step_size: float,
    algorithm_name: str = "ADMM",
    seed: int = -1,
    iteration: int = -1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The common y-x-lambda update with an unscaled multiplier."""
    ensure_finite_state(
        algorithm_name,
        seed,
        iteration,
        x=x,
        y=y,
        dual=dual,
        gradient_estimate=gradient_estimate,
    )
    with numerical_guard(algorithm_name, seed, iteration):
        y_new = soft_threshold(D @ x + dual / rho, mu / rho)
        augmented_gradient = gradient_estimate + rho * np.asarray(
            D.T @ (D @ x - y_new + dual / rho)
        ).reshape(-1)
        x_new = x - step_size * augmented_gradient
        dual_new = dual + rho * (np.asarray(D @ x_new).reshape(-1) - y_new)
    ensure_finite_state(
        algorithm_name,
        seed,
        iteration,
        augmented_gradient=augmented_gradient,
        x=x_new,
        y=y_new,
        dual=dual_new,
    )
    return x_new, y_new, dual_new


def validate_algorithm_inputs(
    X_train: sparse.spmatrix,
    labels_train: np.ndarray,
    D: sparse.spmatrix,
    config: dict[str, Any],
) -> tuple[int, int, int, float, float]:
    n_samples, n_features = X_train.shape
    if labels_train.shape != (n_samples,):
        raise ValueError("Training labels have an invalid shape.")
    if D.shape[1] != n_features:
        raise ValueError("Incidence matrix feature dimension does not match X_train.")
    max_iter = int(config["max_iter"])
    step_size = float(config["step_size"])
    rho = float(config["rho"])
    if (
        max_iter <= 0
        or step_size <= 0.0
        or rho <= 0.0
        or not np.isfinite([step_size, rho]).all()
    ):
        raise ValueError("max_iter, step_size, and rho must be positive.")
    return n_samples, n_features, D.shape[0], step_size, rho
