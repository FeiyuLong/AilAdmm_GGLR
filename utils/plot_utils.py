"""Editable vector convergence plots for iterations, time, and optional IFO count."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from algorithms.common import RunResult


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


METRICS = {
    "optimality_gap": ("Optimality Gap", True, 1.0),
    "primal_residual": (r"Primal Residual $\|D \mathbf{x}-\mathbf{y}\|$", True, 1.0),
    "kkt_residual": (r"KKT Residual $K_t$", True, 1.0),
    "test_logistic_loss": ("Test Logistic Loss", False, 1.0),
    "test_accuracy": ("Test Accuracy (%)", False, 100.0),
}


COLORS = plt.get_cmap("tab10").colors
LINESTYLES = ("-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--")


def _save_vector_figure(figure: plt.Figure, base_path: Path) -> list[Path]:
    paths: list[Path] = []
    for extension in ("pdf", "svg"):
        path = base_path.with_suffix(f".{extension}")
        figure.savefig(path, bbox_inches="tight")
        paths.append(path)
    return paths


def _iteration_values(
    runs: list[RunResult], metric: str
) -> tuple[np.ndarray, np.ndarray]:
    iterations = runs[0].iteration
    for run in runs[1:]:
        if not np.array_equal(iterations, run.iteration):
            raise ValueError("Iteration checkpoints differ across trials.")
    values = np.vstack([run.metric(metric) for run in runs])
    if not np.isfinite(values).all():
        raise ValueError(f"Cannot plot non-finite {metric} values.")
    return iterations, values


def _time_values(
    runs: list[RunResult], metric: str, grid: np.ndarray
) -> np.ndarray:
    interpolated = []
    for run in runs:
        times = np.maximum.accumulate(run.algorithm_time)
        values = run.metric(metric)
        if not np.isfinite(times).all() or not np.isfinite(values).all():
            raise ValueError(f"Cannot plot non-finite {metric} values or times.")
        interpolated.append(np.interp(grid, times, values))
    values = np.vstack(interpolated)
    return values


def _ifo_values(
    runs: list[RunResult], metric: str, grid: np.ndarray
) -> np.ndarray:
    interpolated = []
    for run in runs:
        ifo_counts = np.maximum.accumulate(np.asarray(run.ifo_count, dtype=np.float64))
        values = run.metric(metric)
        if not np.isfinite(ifo_counts).all() or not np.isfinite(values).all():
            raise ValueError(f"Cannot plot non-finite {metric} values or IFO counts.")
        interpolated.append(np.interp(grid, ifo_counts, values))
    return np.vstack(interpolated)


def _draw_median_curve(
    axis: plt.Axes,
    x_values: np.ndarray,
    values: np.ndarray,
    *,
    label: str,
    color: tuple[float, float, float],
    linestyle: str,
    multiplier: float,
    log_scale: bool,
) -> None:
    values = values * multiplier
    median = np.median(values, axis=0)
    if log_scale:
        median = np.maximum(median, np.finfo(np.float64).eps)
    axis.plot(
        x_values,
        median,
        label=label,
        color=color,
        linestyle=linestyle,
        linewidth=1.9,
    )


def plot_comparison_curves(
    results: dict[str, list[RunResult]],
    output_dir: str | Path,
    *,
    filename_prefix: str,
    time_grid_points: int = 200,
    include_ifo_plots: bool = False,
) -> list[Path]:
    if not results or any(not runs for runs in results.values()):
        raise ValueError("At least one completed run is required for every algorithm.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    common_time_limit = min(run.algorithm_time[-1] for runs in results.values() for run in runs)
    if common_time_limit <= 0.0:
        raise ValueError("Algorithm times must be positive at the final checkpoint.")
    time_grid = np.linspace(0.0, common_time_limit, max(2, int(time_grid_points)))
    horizontal_axes = ("iterations", "time")
    if include_ifo_plots:
        common_ifo_limit = min(run.ifo_count[-1] for runs in results.values() for run in runs)
        if common_ifo_limit <= 0:
            raise ValueError("IFO counts must be positive at the final checkpoint.")
        ifo_grid = np.linspace(0.0, common_ifo_limit, max(2, int(time_grid_points)))
        horizontal_axes += ("ifo",)

    output_paths: list[Path] = []
    for metric, (ylabel, log_scale, multiplier) in METRICS.items():
        for horizontal in horizontal_axes:
            figure, axis = plt.subplots(figsize=(7.2, 4.8))
            for index, (name, runs) in enumerate(results.items()):
                if horizontal == "iterations":
                    x_values, values = _iteration_values(runs, metric)
                elif horizontal == "time":
                    x_values = time_grid
                    values = _time_values(runs, metric, time_grid)
                else:
                    x_values = ifo_grid
                    values = _ifo_values(runs, metric, ifo_grid)
                _draw_median_curve(
                    axis,
                    x_values,
                    values,
                    label=name,
                    color=COLORS[index % len(COLORS)],
                    linestyle=LINESTYLES[index % len(LINESTYLES)],
                    multiplier=multiplier,
                    log_scale=log_scale,
                )
            if log_scale:
                axis.set_yscale("log")
            xlabel = {
                "iterations": "Iterations",
                "time": "Time (s)",
                "ifo": "IFO Count",
            }[horizontal]
            axis.set_xlabel(xlabel)
            axis.set_ylabel(ylabel)
            axis.grid(True, which="both", alpha=0.25)
            axis.legend(fontsize=7.5, ncol=2, frameon=True)
            figure.tight_layout()
            output_paths.extend(
                _save_vector_figure(figure, output_dir / f"{filename_prefix}_{metric}_vs_{horizontal}")
            )
            plt.close(figure)
    return output_paths
