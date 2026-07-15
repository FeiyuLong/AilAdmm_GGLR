"""Sparse data loading, preprocessing, and feature-graph construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from sklearn.datasets import load_svmlight_file, load_svmlight_files
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler

from config import DATASET_FILES


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    X_train: sparse.csr_matrix
    y_train: np.ndarray
    X_test: sparse.csr_matrix
    y_test: np.ndarray
    used_external_test: bool
    train_path: Path
    test_path: Path | None


@dataclass(frozen=True)
class FeatureGraph:
    incidence: sparse.csr_matrix
    edges: tuple[tuple[int, int], ...]
    similarities: np.ndarray
    metadata: dict[str, Any]


def _labels_to_pm_one(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    return np.where(labels > 0.0, 1.0, -1.0)


def resolve_dataset_files(data_dir: str | Path, dataset_name: str) -> tuple[Path, Path | None]:
    if dataset_name not in DATASET_FILES:
        raise KeyError(f"Unsupported dataset: {dataset_name!r}")
    data_dir = Path(data_dir)
    spec = DATASET_FILES[dataset_name]
    train_path = data_dir / spec["train"]
    test_path = data_dir / spec["test"]
    if not train_path.is_file():
        raise FileNotFoundError(f"Training file not found: {train_path}")
    return train_path, test_path if test_path.is_file() else None


def load_dataset(
    data_dir: str | Path,
    dataset_name: str,
    *,
    test_size: float = 0.20,
    split_seed: int = 2026,
) -> DatasetBundle:
    """Load LIBSVM/SVMLight data and fit scaling on training data only."""
    train_path, test_path = resolve_dataset_files(data_dir, dataset_name)

    if test_path is not None:
        X_train, y_train, X_test, y_test = load_svmlight_files(
            [str(train_path), str(test_path)], dtype=np.float64
        )
        used_external_test = True
    else:
        X_all, y_all = load_svmlight_file(str(train_path), dtype=np.float64)
        indices = np.arange(X_all.shape[0])
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=split_seed,
            stratify=_labels_to_pm_one(y_all),
        )
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y_all[train_idx], y_all[test_idx]
        used_external_test = False

    X_train = sparse.csr_matrix(X_train, dtype=np.float64)
    X_test = sparse.csr_matrix(X_test, dtype=np.float64)
    y_train = _labels_to_pm_one(y_train)
    y_test = _labels_to_pm_one(y_test)

    scaler = MaxAbsScaler(copy=True)
    X_train = sparse.csr_matrix(scaler.fit_transform(X_train), dtype=np.float64)
    X_test = sparse.csr_matrix(scaler.transform(X_test), dtype=np.float64)
    X_train.eliminate_zeros()
    X_test.eliminate_zeros()

    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError("Train/test feature dimensions differ after loading.")
    if not np.isfinite(X_train.data).all() or not np.isfinite(X_test.data).all():
        raise ValueError("Non-finite feature value detected after scaling.")

    return DatasetBundle(
        name=dataset_name,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        used_external_test=used_external_test,
        train_path=train_path,
        test_path=test_path,
    )


def _absolute_correlation(X: sparse.csr_matrix) -> np.ndarray:
    n_samples, n_features = X.shape
    if n_samples < 2 or n_features < 2:
        raise ValueError("At least two samples and two features are required to build a graph.")

    means = np.asarray(X.mean(axis=0)).reshape(-1)
    second_moment = np.asarray((X.T @ X).toarray(), dtype=np.float64) / n_samples
    covariance = second_moment - np.outer(means, means)
    variance = np.maximum(np.diag(covariance), 0.0)
    scale = np.sqrt(np.outer(variance, variance))
    correlation = np.divide(
        covariance,
        scale,
        out=np.zeros_like(covariance),
        where=scale > np.finfo(np.float64).eps,
    )
    correlation = np.clip(np.abs(correlation), 0.0, 1.0)
    np.fill_diagonal(correlation, 0.0)
    return correlation


def build_correlation_graph(
    X_train: sparse.csr_matrix,
    *,
    k: int = 5,
    zero_tolerance: float = 1.0e-14,
) -> FeatureGraph:
    """Build a deterministic symmetric kNN graph plus a maximum spanning tree."""
    X_train = sparse.csr_matrix(X_train, dtype=np.float64)
    similarities = _absolute_correlation(X_train)
    n_features = X_train.shape[1]
    k = int(max(1, min(k, n_features - 1)))

    edges: set[tuple[int, int]] = set()
    all_indices = np.arange(n_features)
    for source in range(n_features):
        order = np.lexsort((all_indices, -similarities[source]))
        neighbors = [idx for idx in order if idx != source][:k]
        for target in neighbors:
            edges.add((min(source, int(target)), max(source, int(target))))

    # A complete positive-cost graph makes the MST deterministic even for zero correlations.
    epsilon = max(float(zero_tolerance), np.finfo(np.float64).eps)
    costs = 1.0 - similarities + epsilon
    np.fill_diagonal(costs, 0.0)
    mst = minimum_spanning_tree(sparse.csr_matrix(costs)).tocoo()
    for source, target in zip(mst.row, mst.col):
        edges.add((min(int(source), int(target)), max(int(source), int(target))))

    ordered_edges = tuple(sorted(edges))
    row_idx = np.repeat(np.arange(len(ordered_edges)), 2)
    col_idx = np.asarray([v for edge in ordered_edges for v in edge], dtype=np.int64)
    values = np.tile(np.asarray([1.0, -1.0]), len(ordered_edges))
    incidence = sparse.csr_matrix(
        (values, (row_idx, col_idx)), shape=(len(ordered_edges), n_features)
    )

    adjacency = sparse.csr_matrix(
        (
            np.ones(2 * len(ordered_edges)),
            (
                [u for u, v in ordered_edges] + [v for u, v in ordered_edges],
                [v for u, v in ordered_edges] + [u for u, v in ordered_edges],
            ),
        ),
        shape=(n_features, n_features),
    )
    components, _ = connected_components(adjacency, directed=False)
    if components != 1:
        raise RuntimeError("Feature graph construction failed to produce a connected graph.")

    metadata = {
        "method": "absolute-correlation-symmetric-knn-plus-max-spanning-tree",
        "k": k,
        "n_features": n_features,
        "n_edges": len(ordered_edges),
        "connected_components": components,
    }
    return FeatureGraph(incidence, ordered_edges, similarities, metadata)

