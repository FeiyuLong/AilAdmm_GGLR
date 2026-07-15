"""Numerically stable objectives, gradients, residuals, and certification metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.special import expit


@dataclass(frozen=True)
class KKTComponents:
    primal_norm: float
    x_stationarity_norm: float
    y_stationarity_norm: float
    squared_residual: float
    normalized_residual: float


def logistic_loss(X: sparse.spmatrix, labels: np.ndarray, x: np.ndarray) -> float:
    margins = labels * np.asarray(X @ x).reshape(-1)
    return float(np.mean(np.logaddexp(0.0, -margins)))


def sample_gradient(row: sparse.spmatrix, label: float, x: np.ndarray) -> np.ndarray:
    margin = float(label * (row @ x)[0])
    return np.asarray(row.toarray()).reshape(-1) * (-label * expit(-margin))


def batch_gradient(
    X: sparse.spmatrix,
    labels: np.ndarray,
    x: np.ndarray,
    indices: np.ndarray | list[int] | None = None,
) -> np.ndarray:
    if indices is None:
        X_batch = X
        labels_batch = labels
    else:
        X_batch = X[indices]
        labels_batch = labels[np.asarray(indices, dtype=np.int64)]
    margins = labels_batch * np.asarray(X_batch @ x).reshape(-1)
    coefficients = -labels_batch * expit(-margins)
    return np.asarray(X_batch.T @ coefficients).reshape(-1) / len(labels_batch)


def per_sample_gradients(
    X: sparse.spmatrix, labels: np.ndarray, x: np.ndarray, *, block_size: int = 4096
) -> np.ndarray:
    n_samples, n_features = X.shape
    table = np.empty((n_samples, n_features), dtype=np.float64)
    for start in range(0, n_samples, block_size):
        stop = min(start + block_size, n_samples)
        X_block = X[start:stop]
        labels_block = labels[start:stop]
        margins = labels_block * np.asarray(X_block @ x).reshape(-1)
        coefficients = -labels_block * expit(-margins)
        table[start:stop] = X_block.multiply(coefficients[:, None]).toarray()
    return table


def soft_threshold(values: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(values) * np.maximum(np.abs(values) - threshold, 0.0)


def split_objective(
    X: sparse.spmatrix,
    labels: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    mu: float,
) -> float:
    return logistic_loss(X, labels, x) + float(mu * np.linalg.norm(y, 1))


def feasible_objective(
    X: sparse.spmatrix,
    labels: np.ndarray,
    D: sparse.spmatrix,
    x: np.ndarray,
    mu: float,
) -> float:
    return logistic_loss(X, labels, x) + float(mu * np.linalg.norm(D @ x, 1))


def classification_accuracy(X: sparse.spmatrix, labels: np.ndarray, x: np.ndarray) -> float:
    scores = np.asarray(X @ x).reshape(-1)
    predictions = np.where(scores >= 0.0, 1.0, -1.0)
    return float(np.mean(predictions == labels))


def l1_subgradient_distance(
    y: np.ndarray, dual: np.ndarray, mu: float, *, zero_tolerance: float = 1.0e-12
) -> np.ndarray:
    nonzero = np.abs(y) > zero_tolerance
    residual = np.empty_like(y, dtype=np.float64)
    residual[nonzero] = mu * np.sign(y[nonzero]) - dual[nonzero]
    residual[~nonzero] = np.maximum(np.abs(dual[~nonzero]) - mu, 0.0)
    return residual


def kkt_components(
    X: sparse.spmatrix,
    labels: np.ndarray,
    D: sparse.spmatrix,
    x: np.ndarray,
    y: np.ndarray,
    dual: np.ndarray,
    mu: float,
    *,
    zero_tolerance: float = 1.0e-12,
) -> KKTComponents:
    primal = np.asarray(D @ x).reshape(-1) - y
    gradient = batch_gradient(X, labels, x)
    dual_x = np.asarray(D.T @ dual).reshape(-1)
    x_stationarity = gradient + dual_x
    y_stationarity = l1_subgradient_distance(
        y, dual, mu, zero_tolerance=zero_tolerance
    )

    primal_norm = float(np.linalg.norm(primal))
    x_norm = float(np.linalg.norm(x_stationarity))
    y_norm = float(np.linalg.norm(y_stationarity))
    squared = primal_norm**2 + x_norm**2 + y_norm**2

    primal_scale = 1.0 + float(np.linalg.norm(D @ x)) + float(np.linalg.norm(y))
    x_scale = 1.0 + float(np.linalg.norm(gradient)) + float(np.linalg.norm(dual_x))
    y_scale = 1.0 + mu * np.sqrt(y.size) + float(np.linalg.norm(dual))
    normalized = float(
        np.sqrt(
            (primal_norm / primal_scale) ** 2
            + (x_norm / x_scale) ** 2
            + (y_norm / y_scale) ** 2
        )
    )
    return KKTComponents(primal_norm, x_norm, y_norm, squared, normalized)


def relative_primal_residual(D: sparse.spmatrix, x: np.ndarray, y: np.ndarray) -> float:
    Dx = np.asarray(D @ x).reshape(-1)
    return float(np.linalg.norm(Dx - y) / (1.0 + np.linalg.norm(Dx) + np.linalg.norm(y)))

