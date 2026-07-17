from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from algorithms.common import NumericalDivergenceError, RunResult
from utils.plot_utils import (
    _draw_median_curve,
    _ifo_values,
    _iteration_values,
    _time_values,
    plot_comparison_curves,
)
from utils.results import write_run_csv
import matplotlib.pyplot as plt


def artificial_run(name: str, seed: int, scale: float) -> RunResult:
    iterations = np.asarray([0, 1, 2, 3])
    decay = scale * np.asarray([1.0, 0.5, 0.25, 0.125])
    return RunResult(
        name=name,
        seed=seed,
        iteration=iterations,
        algorithm_time=np.asarray([0.0, 0.1, 0.2, 0.3]) * scale,
        ifo_count=iterations * 2,
        feasible_objective=1.0 + decay,
        split_objective=1.0 + decay,
        optimality_gap=decay,
        primal_residual=decay,
        kkt_residual=decay**2,
        test_logistic_loss=np.asarray([0.7, 0.6, 0.55, 0.5]) * scale,
        test_accuracy=np.asarray([0.5, 0.6, 0.7, 0.8]),
        x_final=np.zeros(2),
        y_final=np.zeros(1),
        dual_final=np.zeros(1),
    )


class PlottingTests(unittest.TestCase):
    def test_canonical_algorithm_name_is_preserved_as_curve_label(self) -> None:
        run = artificial_run("AILSVRG-ADMM", 1, 1.0)
        x_values, y_values = _iteration_values([run], "primal_residual")
        figure, axis = plt.subplots()
        try:
            _draw_median_curve(
                axis, x_values, y_values, label=run.name,
                color=(0.0, 0.0, 1.0), linestyle="-", multiplier=1.0,
                log_scale=False,
            )
            self.assertEqual(axis.lines[0].get_label(), "AILSVRG-ADMM")
        finally:
            plt.close(figure)

    def test_default_outputs_exclude_ifo_without_svg_raster_images(self) -> None:
        results = {
            "Method-A": [artificial_run("Method-A", 1, 1.0), artificial_run("Method-A", 2, 1.1)],
            "Method-B": [artificial_run("Method-B", 1, 1.2), artificial_run("Method-B", 2, 1.3)],
        }
        with tempfile.TemporaryDirectory() as temporary:
            paths = plot_comparison_curves(
                results,
                temporary,
                filename_prefix="gglr_tiny",
                time_grid_points=10,
            )
            self.assertEqual(len(paths), 20)
            expected_names = {
                f"gglr_tiny_{metric}_vs_{axis}.{extension}"
                for metric in (
                    "optimality_gap",
                    "primal_residual",
                    "kkt_residual",
                    "test_logistic_loss",
                    "test_accuracy",
                )
                for axis in ("iterations", "time")
                for extension in ("pdf", "svg")
            }
            self.assertEqual({path.name for path in paths}, expected_names)
            self.assertEqual({path.name for path in Path(temporary).iterdir()}, expected_names)
            for path in paths:
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)
                self.assertEqual(path.parent, Path(temporary))
                self.assertTrue(path.name.startswith("gglr_tiny_"))
                self.assertIn(path.suffix, {".pdf", ".svg"})
                if path.suffix == ".svg":
                    content = Path(path).read_text(encoding="utf-8")
                    self.assertNotIn("<image", content)
                    if "_vs_time." in path.name:
                        self.assertIn("Time (s)", content)
                        self.assertNotIn("Algorithm Time (s)", content)

    def test_ifo_outputs_are_created_when_explicitly_enabled(self) -> None:
        results = {
            "Method-A": [artificial_run("Method-A", 1, 1.0)],
            "Method-B": [artificial_run("Method-B", 1, 1.2)],
        }
        with tempfile.TemporaryDirectory() as temporary:
            paths = plot_comparison_curves(
                results,
                temporary,
                filename_prefix="gglr_tiny",
                time_grid_points=10,
                include_ifo_plots=True,
            )
            expected_names = {
                f"gglr_tiny_{metric}_vs_{axis}.{extension}"
                for metric in (
                    "optimality_gap",
                    "primal_residual",
                    "kkt_residual",
                    "test_logistic_loss",
                    "test_accuracy",
                )
                for axis in ("iterations", "time", "ifo")
                for extension in ("pdf", "svg")
            }
            self.assertEqual(len(paths), 30)
            self.assertEqual({path.name for path in paths}, expected_names)
            self.assertTrue(any("_vs_ifo." in path.name for path in paths))

    def test_disabled_ifo_plots_skip_ifo_validation(self) -> None:
        results = {
            "Method-A": [
                replace(artificial_run("Method-A", 1, 1.0), ifo_count=np.zeros(4, dtype=int))
            ]
        }
        with tempfile.TemporaryDirectory() as temporary:
            paths = plot_comparison_curves(
                results,
                temporary,
                filename_prefix="gglr_tiny",
                time_grid_points=10,
            )
            self.assertEqual(len(paths), 20)
            self.assertFalse(any("_vs_ifo." in path.name for path in paths))
            with self.assertRaisesRegex(ValueError, "IFO counts must be positive"):
                plot_comparison_curves(
                    results,
                    temporary,
                    filename_prefix="gglr_tiny_with_ifo",
                    time_grid_points=10,
                    include_ifo_plots=True,
                )

    def test_metrics_use_one_median_curve_without_fill(self) -> None:
        runs = [artificial_run("Method-A", seed, 1.0) for seed in (1, 2, 3)]
        low = np.asarray([1.0, 1.0e-12, 1.0e-12, 1.0e-12])
        high = np.asarray([1.0, 1.0, 1.0, 1.0])
        runs[0] = replace(runs[0], primal_residual=low)
        runs[1] = replace(runs[1], primal_residual=low)
        runs[2] = replace(runs[2], primal_residual=high)

        iterations, values = _iteration_values(runs, "primal_residual")
        self.assertGreater(values.std(axis=0)[1], values.mean(axis=0)[1])
        for log_scale in (False, True):
            figure, axis = plt.subplots()
            try:
                _draw_median_curve(
                    axis,
                    iterations,
                    values,
                    label="Method-A",
                    color=(0.0, 0.0, 1.0),
                    linestyle="-",
                    multiplier=1.0,
                    log_scale=log_scale,
                )
                self.assertEqual(len(axis.lines), 1)
                self.assertEqual(len(axis.collections), 0)
                expected = np.median(values, axis=0)
                if log_scale:
                    expected = np.maximum(expected, np.finfo(np.float64).eps)
                np.testing.assert_allclose(axis.lines[0].get_ydata(), expected)
            finally:
                plt.close(figure)

        grid = np.asarray([0.0, 0.15, 0.30])
        interpolated = _time_values(runs, "primal_residual", grid)
        self.assertEqual(interpolated.shape, (3, 3))
        for index, run in enumerate(runs):
            np.testing.assert_allclose(
                interpolated[index],
                np.interp(grid, np.maximum.accumulate(run.algorithm_time), run.primal_residual),
            )
        np.testing.assert_allclose(
            np.median(interpolated, axis=0),
            np.asarray([1.0, 1.0e-12, 1.0e-12]),
        )

        ifo_runs = [
            replace(run, ifo_count=np.asarray([0, 2, 4, 6]) * (index + 1))
            for index, run in enumerate(runs)
        ]
        ifo_grid = np.asarray([0.0, 3.0, 6.0])
        ifo_interpolated = _ifo_values(ifo_runs, "primal_residual", ifo_grid)
        self.assertEqual(ifo_interpolated.shape, (3, 3))
        for index, run in enumerate(ifo_runs):
            np.testing.assert_allclose(
                ifo_interpolated[index],
                np.interp(ifo_grid, run.ifo_count, run.primal_residual),
            )


class ResultSerializationTests(unittest.TestCase):
    def test_csv_writer_handles_iteration_checkpoint_field(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = write_run_csv(artificial_run("AILSVRG-ADMM", 1, 1.0), temporary)
            self.assertTrue(path.is_file())
            self.assertEqual(path.name, "ailsvrg_admm_seed_1.csv")
            rows = path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(rows), 5)
        self.assertTrue(rows[0].startswith("iteration,"))
        self.assertIn("test_logistic_loss", rows[0])

    def test_csv_writer_rejects_nonfinite_metric(self) -> None:
        invalid = replace(
            artificial_run("Method-A", 1, 1.0),
            optimality_gap=np.asarray([1.0, np.nan, 0.5, 0.25]),
        )
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(NumericalDivergenceError):
                write_run_csv(invalid, temporary)


if __name__ == "__main__":
    unittest.main()
