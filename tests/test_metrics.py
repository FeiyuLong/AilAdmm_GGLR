from __future__ import annotations

import unittest

import numpy as np
from scipy import sparse

from algorithms.common import Evaluator
from utils.metrics import (
    batch_gradient,
    kkt_components,
    logistic_loss,
    soft_threshold,
)


class MetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.X = sparse.csr_matrix(
            [[1.0, 0.5], [-1.0, 0.25], [0.2, -0.7], [-0.2, 0.7]]
        )
        self.labels = np.asarray([1.0, -1.0, 1.0, -1.0])

    def test_full_gradient_matches_finite_difference(self) -> None:
        x = np.asarray([0.2, -0.3])
        analytic = batch_gradient(self.X, self.labels, x)
        epsilon = 1.0e-6
        numerical = np.empty_like(x)
        for index in range(x.size):
            direction = np.zeros_like(x)
            direction[index] = epsilon
            numerical[index] = (
                logistic_loss(self.X, self.labels, x + direction)
                - logistic_loss(self.X, self.labels, x - direction)
            ) / (2.0 * epsilon)
        np.testing.assert_allclose(analytic, numerical, rtol=1.0e-6, atol=1.0e-8)

    def test_soft_threshold(self) -> None:
        values = np.asarray([-2.0, -0.2, 0.0, 0.2, 2.0])
        expected = np.asarray([-1.5, 0.0, 0.0, 0.0, 1.5])
        np.testing.assert_allclose(soft_threshold(values, 0.5), expected)

    def test_kkt_residual_is_zero_at_symmetric_origin(self) -> None:
        X = sparse.csr_matrix([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        labels = np.ones(4)
        D = sparse.csr_matrix([[1.0, -1.0]])
        components = kkt_components(
            X, labels, D, np.zeros(2), np.zeros(1), np.zeros(1), 1.0e-2
        )
        self.assertLess(components.squared_residual, 1.0e-28)
        self.assertLess(components.normalized_residual, 1.0e-14)

    def test_evaluator_reports_finite_test_logistic_loss(self) -> None:
        D = sparse.csr_matrix([[1.0, -1.0]])
        evaluator = Evaluator(
            self.X,
            self.labels,
            self.X,
            self.labels,
            D,
            1.0e-2,
            0.0,
        )
        x = np.asarray([0.2, -0.3])
        metrics = evaluator.evaluate(x, np.zeros(1), np.zeros(1))
        expected = logistic_loss(self.X, self.labels, x)
        self.assertTrue(np.isfinite(metrics["test_logistic_loss"]))
        self.assertAlmostEqual(metrics["test_logistic_loss"], expected)


if __name__ == "__main__":
    unittest.main()
