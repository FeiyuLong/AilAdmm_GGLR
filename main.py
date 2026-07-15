"""Command-line entry point for the complete GGLR benchmark."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Callable

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import ArpackNoConvergence, eigsh

from algorithms import (
    run_ail_svrg_admm,
    run_ail_svrg_admm_fixed_p,
    run_ail_svrg_admm_with_corr,
    run_ail_svrg_admm_no_mom,
    run_asvrg_admm,
    run_sag_admm,
    run_saga_admm,
    run_spider_admm,
    run_stoc_admm,
    run_svrg_admm,
)
from algorithms.common import NumericalDivergenceError, RunResult
from config import (
    ALGORITHM_ORDER,
    ALGORITHM_PARAMS,
    CLARABEL_SETTINGS,
    DATASET_FILES,
    ECOS_SETTINGS,
    GLOBAL_SETTINGS,
    PLOT_SETTINGS,
    REFERENCE_CERTIFICATION,
)
from utils.data_utils import build_correlation_graph, load_dataset, resolve_dataset_files
from utils.optimizer import (
    ReferenceSolveError,
    ensure_reference_solvers_available,
    solve_reference_problem,
)
from utils.plot_utils import plot_comparison_curves


Algorithm = Callable[..., RunResult]
RESULTS_DIR = Path("results")
PLOT_FORMATS = ("pdf", "svg")
ALGORITHM_REGISTRY: dict[str, Algorithm] = {
    "STOC-ADMM": run_stoc_admm,
    "SAG-ADMM": run_sag_admm,
    "SAGA-ADMM": run_saga_admm,
    "SVRG-ADMM": run_svrg_admm,
    "ASVRG-ADMM": run_asvrg_admm,
    "SPIDER-ADMM": run_spider_admm,
    "AIL-SVRG-ADMM": run_ail_svrg_admm,
    "AIL-SVRG-ADMM-NoMom": run_ail_svrg_admm_no_mom,
    "AIL-SVRG-ADMM-Fixed-p": run_ail_svrg_admm_fixed_p,
    "AIL-SVRG-ADMM-WithCorr": run_ail_svrg_admm_with_corr,
}


def _spectral_squared_norm(
    matrix: sparse.spmatrix, *, relative_safety: float = 1.0e-6
) -> float:
    """Return a slightly inflated estimate of ``||matrix||_2^2`` using ARPACK."""
    matrix = sparse.csc_matrix(matrix, dtype=np.float64)
    n_features = matrix.shape[1]
    if n_features <= 0:
        raise ValueError("A spectral norm requires at least one matrix column.")
    if relative_safety < 0.0:
        raise ValueError("relative_safety must be nonnegative.")
    if not np.isfinite(matrix.data).all():
        raise ValueError("Cannot compute a spectral norm from non-finite data.")
    if matrix.nnz == 0:
        return 0.0

    gram = sparse.csc_matrix(matrix.T @ matrix)
    if n_features == 1:
        eigenvalue = float(gram[0, 0])
    else:
        try:
            eigenvalues = eigsh(
                gram,
                k=1,
                which="LA",
                return_eigenvectors=False,
                tol=1.0e-10,
                maxiter=max(1_000, 20 * n_features),
                v0=np.linspace(1.0, 2.0, n_features, dtype=np.float64),
            )
        except ArpackNoConvergence as exc:
            raise RuntimeError("ARPACK failed to compute the largest spectral value.") from exc
        eigenvalue = float(eigenvalues[0])

    if not np.isfinite(eigenvalue) or eigenvalue <= 0.0:
        raise RuntimeError(
            "A nonzero matrix produced a nonpositive or non-finite squared spectral norm."
        )
    return eigenvalue * (1.0 + relative_safety)


def _resolve_algorithm_config(
    name: str,
    n_samples: int,
    base_step: float,
    *,
    max_iter: int,
    eval_every: int,
    mu: float,
    rho: float,
) -> dict[str, Any]:
    config = deepcopy(ALGORITHM_PARAMS[name])
    if config.get("batch_size") == "sqrt_n":
        config["batch_size"] = max(1, int(math.ceil(math.sqrt(n_samples))))
    if config.get("refresh_period") == "sqrt_n":
        config["refresh_period"] = max(1, int(math.ceil(math.sqrt(n_samples))))
    if config.get("inner_iter") == "auto":
        config["inner_iter"] = max(1, int(math.ceil(n_samples / int(config["batch_size"]))))
    multiplier = float(config.pop("step_multiplier"))
    config.update(
        {
            "step_size": base_step * multiplier,
            "max_iter": max_iter,
            "eval_every": eval_every,
            "mu": mu,
            "rho": rho,
        }
    )
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets"))
    parser.add_argument("--datasets", nargs="+", choices=list(DATASET_FILES), default=list(DATASET_FILES))
    parser.add_argument("--algorithms", nargs="+", choices=ALGORITHM_ORDER, default=ALGORITHM_ORDER)
    parser.add_argument("--seeds", nargs="+", type=int, default=GLOBAL_SETTINGS["seeds"])
    parser.add_argument("--max-iter", type=int, default=GLOBAL_SETTINGS["max_iter"])
    parser.add_argument("--eval-every", type=int, default=GLOBAL_SETTINGS["eval_every"])
    parser.add_argument("--preflight", action="store_true", help="Check dependencies and dataset dimensions only.")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved global configuration and exit.")
    parser.add_argument("--reference-only", action="store_true", help="Compute and certify F* without running algorithms.")
    parser.add_argument(
        "--recompute-reference",
        action="store_true",
        help="Retained for compatibility; reference caches are temporary for every run.",
    )
    return parser


def _configuration_summary(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "data_dir": str(args.data_dir),
        "results_dir": str(RESULTS_DIR),
        "datasets": args.datasets,
        "algorithms": args.algorithms,
        "seeds": args.seeds,
        "max_iter": args.max_iter,
        "eval_every": args.eval_every,
        "plot_formats": list(PLOT_FORMATS),
        "include_ifo_plots": PLOT_SETTINGS["include_ifo_plots"],
        "mu": GLOBAL_SETTINGS["mu"],
        "rho": GLOBAL_SETTINGS["rho"],
        "clarabel": CLARABEL_SETTINGS,
        "ecos": ECOS_SETTINGS,
    }


def run_preflight(args: argparse.Namespace) -> int:
    versions = ensure_reference_solvers_available()
    print(
        f"CVXPY {versions['CVXPY']}; CLARABEL {versions['CLARABEL']}; "
        f"ECOS {versions['ECOS']}"
    )
    for dataset_name in args.datasets:
        train_path, test_path = resolve_dataset_files(args.data_dir, dataset_name)
        bundle = load_dataset(
            args.data_dir,
            dataset_name,
            test_size=GLOBAL_SETTINGS["split_test_size"],
            split_seed=GLOBAL_SETTINGS["split_seed"],
        )
        print(
            f"{dataset_name}: train={bundle.X_train.shape}, test={bundle.X_test.shape}, "
            f"train_file={train_path}, test_file={test_path or 'generated split'}"
        )
    return 0


def _clear_results_dir(results_dir: Path = RESULTS_DIR) -> None:
    """Create the fixed output directory and remove artifacts from prior runs."""
    results_dir.mkdir(parents=True, exist_ok=True)
    for child in results_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def run_dataset(args: argparse.Namespace, dataset_name: str) -> int:
    bundle = load_dataset(
        args.data_dir,
        dataset_name,
        test_size=GLOBAL_SETTINGS["split_test_size"],
        split_seed=GLOBAL_SETTINGS["split_seed"],
    )
    graph = build_correlation_graph(
        bundle.X_train,
        k=GLOBAL_SETTINGS["graph_k"],
        zero_tolerance=GLOBAL_SETTINGS["graph_zero_tolerance"],
    )
    D = graph.incidence
    with tempfile.TemporaryDirectory(prefix=f"gglr_{dataset_name}_") as temporary:
        reference = solve_reference_problem(
            bundle.X_train,
            bundle.y_train,
            D,
            GLOBAL_SETTINGS["mu"],
            dataset_name=dataset_name,
            cache_dir=Path(temporary) / "reference",
            solver_settings=CLARABEL_SETTINGS,
            ecos_settings=ECOS_SETTINGS,
            certification=REFERENCE_CERTIFICATION,
            zero_tolerance=GLOBAL_SETTINGS["kkt_zero_tolerance"],
            recompute=True,
        )
        print(
            f"{dataset_name}: certified F*={reference.f_star:.12g}, "
            f"normalized KKT={reference.normalized_kkt_residual:.3e}"
        )
        if args.reference_only:
            return 0

        logistic_lipschitz = _spectral_squared_norm(bundle.X_train) / (4.0 * bundle.X_train.shape[0])
        graph_norm = _spectral_squared_norm(D)
        rho = float(GLOBAL_SETTINGS["rho"])
        base_step = 1.0 / (1.0 + logistic_lipschitz + rho * graph_norm)
        results: dict[str, list[RunResult]] = {name: [] for name in args.algorithms}
        failed_runs = 0

        for name in args.algorithms:
            algorithm_config = _resolve_algorithm_config(
                name,
                bundle.X_train.shape[0],
                base_step,
                max_iter=args.max_iter,
                eval_every=args.eval_every,
                mu=GLOBAL_SETTINGS["mu"],
                rho=rho,
            )
            for seed in args.seeds:
                print(f"{dataset_name} | {name} | seed={seed}")
                try:
                    result = ALGORITHM_REGISTRY[name](
                        bundle.X_train,
                        bundle.y_train,
                        bundle.X_test,
                        bundle.y_test,
                        D,
                        reference.f_star,
                        algorithm_config,
                        seed,
                        name=name,
                    )
                except NumericalDivergenceError as exc:
                    failed_runs += 1
                    print(f"ERROR [{dataset_name} | {name} | seed={seed}]: {exc}", file=sys.stderr)
                    continue
                results[name].append(result)

        completed_results = {name: runs for name, runs in results.items() if runs}
        if not completed_results:
            raise RuntimeError("All algorithm runs failed; no figures were generated.")
        plot_comparison_curves(
            completed_results,
            RESULTS_DIR,
            filename_prefix=f"gglr_{dataset_name}",
            time_grid_points=GLOBAL_SETTINGS["time_grid_points"],
            include_ifo_plots=PLOT_SETTINGS["include_ifo_plots"],
        )
        return failed_runs


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_iter <= 0 or args.eval_every <= 0:
        raise SystemExit("--max-iter and --eval-every must be positive.")
    if args.dry_run:
        print(json.dumps(_configuration_summary(args), ensure_ascii=False, indent=2))
        return 0
    if args.preflight:
        return run_preflight(args)

    if not args.reference_only:
        _clear_results_dir()
    failures = 0
    for dataset_name in args.datasets:
        try:
            failures += run_dataset(args, dataset_name)
        except (ReferenceSolveError, FileNotFoundError, ValueError, RuntimeError) as exc:
            failures += 1
            print(f"ERROR [{dataset_name}]: {exc}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
