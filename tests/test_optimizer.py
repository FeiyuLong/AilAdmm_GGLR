from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy import sparse

from config import CLARABEL_SETTINGS, ECOS_SETTINGS, REFERENCE_CERTIFICATION
from utils.optimizer import (
    _cross_solver_discrepancy,
    _solver_attempts,
    ensure_clarabel_available,
    ensure_reference_solvers_available,
    solve_reference_problem,
)


class OptimizerTests(unittest.TestCase):
    @staticmethod
    def _tiny_problem() -> tuple[sparse.csr_matrix, np.ndarray, sparse.csr_matrix]:
        X = sparse.csr_matrix(
            [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
        )
        return X, np.ones(4), sparse.csr_matrix([[1.0, -1.0]])

    def test_tiny_reference_is_certified_and_cached(self) -> None:
        ensure_clarabel_available()
        X, labels, D = self._tiny_problem()
        with tempfile.TemporaryDirectory() as temporary:
            first = solve_reference_problem(
                X,
                labels,
                D,
                1.0e-2,
                dataset_name="tiny",
                cache_dir=temporary,
                solver_settings=CLARABEL_SETTINGS,
                certification=REFERENCE_CERTIFICATION,
                recompute=True,
            )
            second = solve_reference_problem(
                X,
                labels,
                D,
                1.0e-2,
                dataset_name="tiny",
                cache_dir=temporary,
                solver_settings=CLARABEL_SETTINGS,
                certification=REFERENCE_CERTIFICATION,
                recompute=False,
            )
            metadata = json.loads(
                (Path(temporary) / "tiny_reference.json").read_text(encoding="utf-8")
            )
        self.assertEqual(first.status, "optimal")
        self.assertAlmostEqual(first.f_star, np.log(2.0), places=7)
        self.assertLess(first.normalized_kkt_residual, 1.0e-6)
        self.assertEqual(first.problem_hash, second.problem_hash)
        self.assertEqual(first.solver, "CLARABEL")
        self.assertIn(metadata["solver_profile_used"], {"strict-auto", "strict-faer"})

    def test_ecos_tiny_reference_is_certified(self) -> None:
        ensure_reference_solvers_available()
        X, labels, D = self._tiny_problem()
        with tempfile.TemporaryDirectory() as temporary:
            solution = solve_reference_problem(
                X,
                labels,
                D,
                1.0e-2,
                dataset_name="tiny_ecos",
                cache_dir=temporary,
                solver_settings=CLARABEL_SETTINGS,
                ecos_settings=ECOS_SETTINGS,
                certification=REFERENCE_CERTIFICATION,
                solver_order=("ECOS",),
                recompute=True,
            )
            metadata = json.loads(
                (Path(temporary) / "tiny_ecos_reference.json").read_text(
                    encoding="utf-8"
                )
            )
        self.assertEqual(solution.solver, "ECOS")
        self.assertEqual(solution.status, "optimal")
        self.assertLess(solution.normalized_kkt_residual, 1.0e-6)
        self.assertEqual(metadata["solver"], "ECOS")
        self.assertEqual(metadata["solver_profile"], "strict-ecos")

    def test_ecos_fallback_after_forced_clarabel_failure(self) -> None:
        X, labels, D = self._tiny_problem()
        forced_clarabel_profile = [("forced-limit", {"max_iter": 1})]
        with tempfile.TemporaryDirectory() as temporary:
            with patch(
                "utils.optimizer._solver_attempts", return_value=forced_clarabel_profile
            ), patch(
                "utils.optimizer._cross_solver_discrepancy", return_value=(None, None)
            ):
                solution = solve_reference_problem(
                    X,
                    labels,
                    D,
                    1.0e-2,
                    dataset_name="tiny_fallback",
                    cache_dir=temporary,
                    solver_settings=CLARABEL_SETTINGS,
                    ecos_settings=ECOS_SETTINGS,
                    certification=REFERENCE_CERTIFICATION,
                    recompute=True,
                )
        self.assertEqual(solution.solver, "ECOS")
        self.assertIn("CLARABEL strict profiles", solution.fallback_reason or "")

    def test_solver_profiles_relax_only_to_full_accuracy(self) -> None:
        attempts = _solver_attempts(CLARABEL_SETTINGS)
        self.assertEqual([name for name, _ in attempts], [
            "strict-auto",
            "strict-faer",
            "full-accuracy-faer",
        ])
        self.assertTrue(all(settings["max_iter"] >= 2_000 for _, settings in attempts))
        self.assertEqual(attempts[1][1]["direct_solve_method"], "faer")
        self.assertEqual(attempts[1][1]["iterative_refinement_max_iter"], 20)
        self.assertEqual(attempts[2][1]["tol_gap_abs"], 1.0e-8)
        self.assertGreaterEqual(
            min(settings["tol_gap_abs"] for _, settings in attempts), 1.0e-9
        )

    def test_cross_solver_objective_check(self) -> None:
        discrepancy, error = _cross_solver_discrepancy(
            0.5 + 1.0e-9, 0.5, REFERENCE_CERTIFICATION
        )
        self.assertAlmostEqual(discrepancy or 0.0, 1.0e-9, places=14)
        self.assertIsNone(error)
        _, error = _cross_solver_discrepancy(0.6, 0.5, REFERENCE_CERTIFICATION)
        self.assertIn("cross-solver objective discrepancy", error or "")


if __name__ == "__main__":
    unittest.main()
