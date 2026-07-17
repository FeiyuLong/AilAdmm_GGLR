from __future__ import annotations

import unittest

import numpy as np
from scipy import sparse

from algorithms.ailsvrg_admm import (
    adaptive_probability,
    cost_matched_probability,
    resolve_p_min,
)
from algorithms.common import NumericalDivergenceError, baseline_admm_step
from config import ALGORITHM_PARAMS
from main import (
    ALGORITHM_REGISTRY,
    _resolve_algorithm_config,
    _spectral_squared_norm,
)
from utils.metrics import batch_gradient, feasible_objective, soft_threshold


class AlgorithmFormulaTests(unittest.TestCase):
    def setUp(self) -> None:
        self.X = sparse.csr_matrix(
            [
                [1.0, 0.0, 0.5],
                [-1.0, 0.2, 0.0],
                [0.0, 1.0, -0.5],
                [0.0, -1.0, 0.5],
                [0.5, 0.5, 0.0],
                [-0.5, -0.5, 0.0],
            ]
        )
        self.labels = np.asarray([1.0, 1.0, -1.0, -1.0, 1.0, -1.0])
        self.D = sparse.csr_matrix([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])
        self.f_star = feasible_objective(self.X, self.labels, self.D, np.zeros(3), 1.0e-2)

    def _config(self, name: str) -> dict[str, object]:
        config: dict[str, object] = {
            "max_iter": 2,
            "eval_every": 1,
            "step_size": 1.0e-2,
            "rho": 1.0,
            "mu": 1.0e-2,
            "batch_size": 2,
        }
        if name in {"SAG-ADMM", "SAGA-ADMM"}:
            config["batch_size"] = 1
        if name in {"SVRG-ADMM", "ASVRG-ADMM"}:
            config["inner_iter"] = 2
        if name == "ASVRG-ADMM":
            config["theta"] = 0.5
        if name == "SPIDER-ADMM":
            config["refresh_period"] = 2
        if name.startswith("AILSVRG-ADMM"):
            config.update(
                {
                    "tau": 0.5,
                    "varrho": 0.3,
                    "beta_y": 0.1,
                    "p_min": "inverse_n",
                }
            )
        if name == "AILSVRG-ADMM-NoMom":
            config["tau"] = 0.0
        if name == "AILSVRG-ADMM-Fixed-p":
            config["fixed_probability"] = "cost_matched"
        if name == "AILSVRG-ADMM-WithCorr":
            config["enable_correction"] = True
        return config

    def test_every_algorithm_produces_finite_deterministic_checkpoints(self) -> None:
        for name, function in ALGORITHM_REGISTRY.items():
            with self.subTest(algorithm=name):
                result = function(
                    self.X,
                    self.labels,
                    self.X,
                    self.labels,
                    self.D,
                    self.f_star,
                    self._config(name),
                    11,
                    name=name,
                )
                self.assertEqual(result.iteration.tolist(), [0, 1, 2])
                self.assertTrue(np.isfinite(result.kkt_residual).all())
                self.assertTrue(np.isfinite(result.test_logistic_loss).all())
                self.assertTrue(np.all(np.diff(result.algorithm_time) >= 0.0))
                self.assertTrue(np.all(np.diff(result.ifo_count) >= 0))
                if name == "AILSVRG-ADMM-NoMom":
                    self.assertEqual(result.metadata["tau"], 0.0)
                if name == "AILSVRG-ADMM-Fixed-p":
                    self.assertIsNotNone(result.metadata["fixed_probability"])
                if name == "AILSVRG-ADMM-WithCorr":
                    self.assertTrue(result.metadata["enable_correction"])
                if name.startswith("AILSVRG-ADMM") and name != "AILSVRG-ADMM-WithCorr":
                    self.assertFalse(result.metadata["enable_correction"])
                if name.startswith("AILSVRG-ADMM"):
                    self.assertEqual(result.metadata["p_min_setting"], "inverse_n")
                    self.assertAlmostEqual(result.metadata["p_min"], 1.0 / self.X.shape[0])

    def test_every_algorithm_stays_finite_with_resolved_safe_step(self) -> None:
        logistic_lipschitz = _spectral_squared_norm(self.X) / (4.0 * self.X.shape[0])
        graph_norm = _spectral_squared_norm(self.D)
        base_step = 1.0 / (1.0 + logistic_lipschitz + graph_norm)
        for name, function in ALGORITHM_REGISTRY.items():
            with self.subTest(algorithm=name):
                config = _resolve_algorithm_config(
                    name,
                    self.X.shape[0],
                    base_step,
                    max_iter=25,
                    eval_every=5,
                    mu=1.0e-2,
                    rho=1.0,
                )
                result = function(
                    self.X,
                    self.labels,
                    self.X,
                    self.labels,
                    self.D,
                    self.f_star,
                    config,
                    11,
                    name=name,
                )
                self.assertTrue(np.isfinite(result.optimality_gap).all())
                self.assertTrue(np.isfinite(result.kkt_residual).all())
                self.assertTrue(np.isfinite(result.x_final).all())

    def test_ailsvrg_ablation_common_parameters_match_main_algorithm(self) -> None:
        shared_keys = (
            "batch_size",
            "step_multiplier",
            "varrho",
            "beta_y",
            "p_min",
        )
        main = ALGORITHM_PARAMS["AILSVRG-ADMM"]
        for name in (
            "AILSVRG-ADMM-NoMom",
            "AILSVRG-ADMM-Fixed-p",
            "AILSVRG-ADMM-WithCorr",
        ):
            with self.subTest(algorithm=name):
                for key in shared_keys:
                    self.assertEqual(ALGORITHM_PARAMS[name][key], main[key])
        self.assertEqual(main["tau"], 0.0)
        self.assertEqual(ALGORITHM_PARAMS["AILSVRG-ADMM-NoMom"]["tau"], 0.0)
        self.assertEqual(ALGORITHM_PARAMS["AILSVRG-ADMM-Fixed-p"]["tau"], main["tau"])
        self.assertEqual(ALGORITHM_PARAMS["AILSVRG-ADMM-WithCorr"]["tau"], main["tau"])
        self.assertNotIn("fixed_probability", main)
        self.assertNotIn("enable_correction", main)
        self.assertEqual(
            ALGORITHM_PARAMS["AILSVRG-ADMM-Fixed-p"]["fixed_probability"],
            "cost_matched",
        )
        self.assertTrue(ALGORITHM_PARAMS["AILSVRG-ADMM-WithCorr"]["enable_correction"])

    def test_nonfinite_update_fails_with_context(self) -> None:
        with self.assertRaisesRegex(
            NumericalDivergenceError, "Injected-ADMM seed=7 iteration=3"
        ):
            baseline_admm_step(
                np.zeros(3),
                np.zeros(2),
                np.zeros(2),
                np.full(3, np.inf),
                self.D,
                mu=1.0e-2,
                rho=1.0,
                step_size=1.0e-2,
                algorithm_name="Injected-ADMM",
                seed=7,
                iteration=3,
            )

    def test_p_min_resolution_and_probability_floor(self) -> None:
        self.assertAlmostEqual(resolve_p_min("inverse_n", n=10, batch_size=4), 0.1)
        self.assertAlmostEqual(resolve_p_min("batch_over_n", n=10, batch_size=4), 0.4)
        self.assertAlmostEqual(resolve_p_min(0.2, n=10, batch_size=4), 0.2)

        p_min = 0.1
        self.assertAlmostEqual(
            adaptive_probability(0, varrho=0.3, p_min=p_min),
            0.3 / 1.3,
        )
        self.assertEqual(adaptive_probability(100, varrho=0.3, p_min=p_min), p_min)
        expected = np.mean(
            [adaptive_probability(t, varrho=0.3, p_min=p_min) for t in range(8)]
        )
        self.assertAlmostEqual(
            cost_matched_probability(8, varrho=0.3, p_min=p_min),
            expected,
        )

    def test_p_min_rejects_invalid_settings(self) -> None:
        invalid_settings = [True, False, 0.0, -0.1, 1.1, np.nan, np.inf, None, "unknown"]
        for setting in invalid_settings:
            with self.subTest(setting=setting):
                with self.assertRaisesRegex(ValueError, "p_min"):
                    resolve_p_min(setting, n=10, batch_size=4)

    def _expected_ailsvrg_first_step(
        self, config: dict[str, object], *, enable_correction: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x0 = np.zeros(self.X.shape[1])
        y0 = np.zeros(self.D.shape[0])
        dual0 = np.zeros(self.D.shape[0])
        gradient = batch_gradient(self.X, self.labels, x0)
        x_prediction = x0 - config["step_size"] * (
            gradient
            + config["rho"]
            * np.asarray(self.D.T @ (self.D @ x0 - y0 + dual0 / config["rho"])).reshape(-1)
        )
        y1 = soft_threshold(
            (
                config["rho"] * np.asarray(self.D @ x_prediction).reshape(-1)
                + dual0
                + config["beta_y"] * y0
            )
            / (config["rho"] + config["beta_y"]),
            config["mu"] / (config["rho"] + config["beta_y"]),
        )
        x1 = x_prediction
        if enable_correction:
            x1 = x_prediction + config["step_size"] * config["rho"] * np.asarray(
                self.D.T @ (y1 - y0)
            ).reshape(-1)
        dual1 = dual0 + config["rho"] * (np.asarray(self.D @ x1).reshape(-1) - y1)
        return x1, y1, dual1

    def test_uncorrected_ailsvrg_variants_match_modified_iteration(self) -> None:
        for name in (
            "AILSVRG-ADMM",
            "AILSVRG-ADMM-NoMom",
            "AILSVRG-ADMM-Fixed-p",
        ):
            with self.subTest(algorithm=name):
                config = self._config(name)
                config.update(
                    {
                        "max_iter": 1,
                        "batch_size": self.X.shape[0],
                        "step_size": 1.0e-2,
                        "tau": 0.3,
                        "varrho": 3.0,
                        "beta_y": 0.1,
                        "p_min": 1.0,
                    }
                )
                if name == "AILSVRG-ADMM-NoMom":
                    config["tau"] = 0.0
                result = ALGORITHM_REGISTRY[name](
                    self.X,
                    self.labels,
                    self.X,
                    self.labels,
                    self.D,
                    self.f_star,
                    config,
                    11,
                    name=name,
                )
                x1, y1, dual1 = self._expected_ailsvrg_first_step(
                    config, enable_correction=False
                )

                np.testing.assert_allclose(result.x_final, x1, rtol=0.0, atol=1.0e-14)
                np.testing.assert_allclose(result.y_final, y1, rtol=0.0, atol=1.0e-14)
                np.testing.assert_allclose(result.dual_final, dual1, rtol=0.0, atol=1.0e-14)

    def test_with_corr_ablation_matches_previous_correction(self) -> None:
        config = self._config("AILSVRG-ADMM-WithCorr")
        config.update(
            {
                "max_iter": 1,
                "batch_size": self.X.shape[0],
                "step_size": 1.0e-2,
                "tau": 0.3,
                "varrho": 3.0,
                "beta_y": 0.1,
                "p_min": 1.0,
            }
        )
        result = ALGORITHM_REGISTRY["AILSVRG-ADMM-WithCorr"](
            self.X,
            self.labels,
            self.X,
            self.labels,
            self.D,
            self.f_star,
            config,
            11,
            name="AILSVRG-ADMM-WithCorr",
        )
        x1, y1, dual1 = self._expected_ailsvrg_first_step(config, enable_correction=True)

        np.testing.assert_allclose(result.x_final, x1, rtol=0.0, atol=1.0e-14)
        np.testing.assert_allclose(result.y_final, y1, rtol=0.0, atol=1.0e-14)
        np.testing.assert_allclose(result.dual_final, dual1, rtol=0.0, atol=1.0e-14)

    def test_cost_matched_ablation_uses_configured_p_min(self) -> None:
        config = self._config("AILSVRG-ADMM-Fixed-p")
        config.update({"max_iter": 8, "p_min": 0.2})
        result = ALGORITHM_REGISTRY["AILSVRG-ADMM-Fixed-p"](
            self.X,
            self.labels,
            self.X,
            self.labels,
            self.D,
            self.f_star,
            config,
            11,
            name="AILSVRG-ADMM-Fixed-p",
        )
        self.assertAlmostEqual(result.metadata["p_min"], 0.2)
        self.assertAlmostEqual(
            result.metadata["fixed_probability"],
            cost_matched_probability(8, varrho=0.3, p_min=0.2),
        )


if __name__ == "__main__":
    unittest.main()
