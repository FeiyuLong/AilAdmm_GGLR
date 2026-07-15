from __future__ import annotations

import unittest

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from utils.data_utils import build_correlation_graph


class FeatureGraphTests(unittest.TestCase):
    def test_graph_is_deterministic_connected_incidence(self) -> None:
        rng = np.random.default_rng(7)
        X = sparse.csr_matrix(rng.normal(size=(30, 8)))
        first = build_correlation_graph(X, k=3)
        second = build_correlation_graph(X, k=3)
        self.assertEqual(first.edges, second.edges)
        np.testing.assert_allclose(first.incidence.toarray(), second.incidence.toarray())

        incidence = first.incidence
        self.assertTrue(np.all(np.diff(incidence.indptr) == 2))
        for row in incidence.toarray():
            self.assertEqual(set(row[row != 0.0]), {-1.0, 1.0})
        adjacency = incidence.T @ incidence
        adjacency.setdiag(0.0)
        adjacency.eliminate_zeros()
        components, _ = connected_components(adjacency, directed=False)
        self.assertEqual(components, 1)


if __name__ == "__main__":
    unittest.main()

