"""Certified high-accuracy CLARABEL/ECOS reference solver for GGLR."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable
import warnings

import numpy as np
from scipy import sparse

from utils.metrics import (
    feasible_objective,
    kkt_components,
    relative_primal_residual,
    split_objective,
)


class ReferenceSolveError(RuntimeError):
    """Raised when no reference solution can be certified."""


@dataclass(frozen=True)
class ReferenceSolution:
    f_star: float
    solver_objective: float
    split_objective: float
    x: np.ndarray
    y: np.ndarray
    dual: np.ndarray
    status: str
    num_iters: int | None
    solve_time: float | None
    primal_relative_residual: float
    normalized_kkt_residual: float
    objective_discrepancy: float
    problem_hash: str
    solver: str = "CLARABEL"
    solver_profile: str | None = None
    fallback_reason: str | None = None
    cross_solver_objective_discrepancy: float | None = None

    def metadata(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("x", "y", "dual"):
            data.pop(key)
        return data


def ensure_reference_solvers_available(
    solvers: Iterable[str] = ("CLARABEL", "ECOS"),
) -> dict[str, str]:
    """Check the open-source conic solvers used by the reference workflow."""
    requested = tuple(dict.fromkeys(str(solver).upper() for solver in solvers))
    unsupported = set(requested) - {"CLARABEL", "ECOS"}
    if unsupported:
        raise ReferenceSolveError(
            f"Unsupported reference solver request: {sorted(unsupported)}"
        )
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise ReferenceSolveError("CVXPY is not installed. Run: python -m pip install cvxpy") from exc

    versions = {"CVXPY": cp.__version__}
    for solver in requested:
        if solver not in cp.installed_solvers():
            raise ReferenceSolveError(
                f"CVXPY does not report {solver} as an installed solver."
            )
        try:
            if solver == "CLARABEL":
                import clarabel

                versions[solver] = getattr(clarabel, "__version__", "unknown")
            else:
                from importlib.metadata import version

                versions[solver] = version("ecos")
        except Exception as exc:
            raise ReferenceSolveError(f"{solver} is not importable: {exc}") from exc
    return versions


def ensure_clarabel_available() -> tuple[str, str]:
    """Backward-compatible CLARABEL-only availability check."""
    versions = ensure_reference_solvers_available(("CLARABEL",))
    return versions["CVXPY"], versions["CLARABEL"]


def _update_hash(digest: Any, matrix: sparse.spmatrix | np.ndarray) -> None:
    if sparse.issparse(matrix):
        csr = sparse.csr_matrix(matrix)
        digest.update(np.asarray(csr.shape, dtype=np.int64).tobytes())
        digest.update(csr.data.tobytes())
        digest.update(csr.indices.tobytes())
        digest.update(csr.indptr.tobytes())
    else:
        array = np.asarray(matrix)
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(array).tobytes())


def problem_hash(
    X: sparse.spmatrix, labels: np.ndarray, D: sparse.spmatrix, mu: float
) -> str:
    digest = hashlib.sha256()
    _update_hash(digest, X)
    _update_hash(digest, labels)
    _update_hash(digest, D)
    digest.update(np.asarray([mu], dtype=np.float64).tobytes())
    return digest.hexdigest()


def _cache_paths(cache_dir: Path, dataset_name: str) -> tuple[Path, Path, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return (
        cache_dir / f"{dataset_name}_reference.npz",
        cache_dir / f"{dataset_name}_reference.json",
        cache_dir / f"{dataset_name}_reference_failure.json",
    )


def _load_cache(npz_path: Path, json_path: Path, expected_hash: str) -> ReferenceSolution | None:
    if not npz_path.is_file() or not json_path.is_file():
        return None
    with json_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("problem_hash") != expected_hash:
        return None
    arrays = np.load(npz_path, allow_pickle=False)
    return ReferenceSolution(
        f_star=float(metadata["f_star"]),
        solver_objective=float(metadata["solver_objective"]),
        split_objective=float(metadata["split_objective"]),
        x=arrays["x"],
        y=arrays["y"],
        dual=arrays["dual"],
        status=str(metadata["status"]),
        num_iters=metadata.get("num_iters"),
        solve_time=metadata.get("solve_time"),
        primal_relative_residual=float(metadata["primal_relative_residual"]),
        normalized_kkt_residual=float(metadata["normalized_kkt_residual"]),
        objective_discrepancy=float(metadata["objective_discrepancy"]),
        problem_hash=expected_hash,
        solver=str(metadata.get("solver", "CLARABEL")),
        solver_profile=metadata.get("solver_profile") or metadata.get("solver_profile_used"),
        fallback_reason=metadata.get("fallback_reason"),
        cross_solver_objective_discrepancy=metadata.get(
            "cross_solver_objective_discrepancy"
        ),
    )


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)


def _solver_attempts(
    solver_settings: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Build deterministic CLARABEL profiles that still require full accuracy."""
    primary = dict(solver_settings)
    primary["max_iter"] = max(2_000, int(primary.get("max_iter", 2_000)))
    primary.setdefault("direct_solve_method", "auto")

    faer = dict(primary)
    faer["direct_solve_method"] = "faer"
    faer["iterative_refinement_enable"] = True
    faer["iterative_refinement_max_iter"] = max(
        20, int(faer.get("iterative_refinement_max_iter", 20))
    )

    relaxed = dict(faer)
    for key in ("tol_gap_abs", "tol_gap_rel", "tol_feas"):
        relaxed[key] = max(float(relaxed[key]), 1.0e-8)

    profiles = [("strict-auto", primary), ("strict-faer", faer)]
    if relaxed != faer:
        profiles.append(("full-accuracy-faer", relaxed))
    return profiles


def _ecos_attempts(ecos_settings: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    settings = dict(ecos_settings)
    settings["max_iters"] = max(10_000, int(settings.get("max_iters", 10_000)))
    for key in ("abstol", "reltol", "feastol"):
        settings[key] = min(float(settings.get(key, 1.0e-9)), 1.0e-9)
    return [("strict-ecos", settings)]


def _solver_stats_metadata(problem: Any) -> dict[str, Any]:
    stats = problem.solver_stats
    return {
        "solver_name": getattr(stats, "solver_name", None),
        "num_iters": getattr(stats, "num_iters", None),
        "solve_time": getattr(stats, "solve_time", None),
        "setup_time": getattr(stats, "setup_time", None),
    }


def _feasibility_tolerance(settings: dict[str, Any], zero_tolerance: float) -> float:
    return max(
        zero_tolerance,
        10.0 * float(settings.get("tol_feas", settings.get("feastol", zero_tolerance))),
    )


def _certify_current_solution(
    cp: Any,
    problem: Any,
    equality: Any,
    x_var: Any,
    y_var: Any,
    X: sparse.spmatrix,
    labels: np.ndarray,
    D: sparse.spmatrix,
    mu: float,
    certification: dict[str, float],
    zero_tolerance: float,
    settings: dict[str, Any],
    problem_digest: str,
    solver: str,
    profile: str,
) -> tuple[ReferenceSolution | None, list[str], dict[str, Any]]:
    attempt_info: dict[str, Any] = {
        "status": problem.status,
        "solver_objective": _finite_or_none(problem.value),
        "solver_stats": _solver_stats_metadata(problem),
    }
    if problem.status != cp.OPTIMAL:
        return None, [f"status {problem.status!r}"], attempt_info
    if x_var.value is None or y_var.value is None or equality.dual_value is None:
        return None, ["incomplete primal-dual values"], attempt_info

    x = np.asarray(x_var.value, dtype=np.float64).reshape(-1)
    y = np.asarray(y_var.value, dtype=np.float64).reshape(-1)
    dual = np.asarray(equality.dual_value, dtype=np.float64).reshape(-1)
    solver_objective = float(problem.value)
    split_value = split_objective(X, labels, x, y, mu)
    f_star = feasible_objective(X, labels, D, x, mu)
    objective_discrepancy = abs(split_value - solver_objective)
    primal_relative = relative_primal_residual(D, x, y)
    active_zero_tolerance = _feasibility_tolerance(settings, zero_tolerance)
    kkt = kkt_components(
        X,
        labels,
        D,
        x,
        y,
        dual,
        mu,
        zero_tolerance=active_zero_tolerance,
    )
    attempt_info.update(
        {
            "primal_relative_residual": primal_relative,
            "normalized_kkt_residual": kkt.normalized_residual,
            "objective_discrepancy": objective_discrepancy,
            "active_zero_tolerance": active_zero_tolerance,
        }
    )

    finite_values = np.concatenate(
        [x, y, dual, [solver_objective, split_value, f_star, kkt.normalized_residual]]
    )
    errors: list[str] = []
    if not np.isfinite(finite_values).all():
        errors.append("non-finite primal, dual, objective, or residual value")
    if primal_relative > certification["primal_relative_tolerance"]:
        errors.append(f"relative primal residual {primal_relative:.3e}")
    if kkt.normalized_residual > certification["normalized_kkt_tolerance"]:
        errors.append(f"normalized KKT residual {kkt.normalized_residual:.3e}")
    objective_limit = certification["objective_relative_tolerance"] * (
        1.0 + abs(solver_objective)
    )
    if objective_discrepancy > objective_limit:
        errors.append(
            f"objective discrepancy {objective_discrepancy:.3e} > {objective_limit:.3e}"
        )
    if errors:
        return None, errors, attempt_info

    stats = problem.solver_stats
    return (
        ReferenceSolution(
            f_star=f_star,
            solver_objective=solver_objective,
            split_objective=split_value,
            x=x,
            y=y,
            dual=dual,
            status=str(problem.status),
            num_iters=getattr(stats, "num_iters", None),
            solve_time=getattr(stats, "solve_time", None),
            primal_relative_residual=primal_relative,
            normalized_kkt_residual=kkt.normalized_residual,
            objective_discrepancy=objective_discrepancy,
            problem_hash=problem_digest,
            solver=solver,
            solver_profile=profile,
        ),
        [],
        attempt_info,
    )


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    scalar = float(value)
    return scalar if np.isfinite(scalar) else None


def _cross_solver_discrepancy(
    candidate_objective: float,
    clarabel_objective: float | None,
    certification: dict[str, float],
) -> tuple[float | None, str | None]:
    if clarabel_objective is None:
        return None, None
    discrepancy = abs(candidate_objective - clarabel_objective)
    limit = certification["objective_relative_tolerance"] * (
        1.0 + abs(clarabel_objective)
    )
    if discrepancy > limit:
        return discrepancy, (
            f"cross-solver objective discrepancy {discrepancy:.3e} > {limit:.3e}"
        )
    return discrepancy, None


def _save_solution(
    solution: ReferenceSolution,
    npz_path: Path,
    json_path: Path,
    failure_path: Path,
    settings: dict[str, Any],
    zero_tolerance: float,
) -> None:
    np.savez_compressed(npz_path, x=solution.x, y=solution.y, dual=solution.dual)
    metadata = solution.metadata()
    metadata["solver_profile_used"] = solution.solver_profile
    metadata["solver_settings_used"] = settings
    metadata["active_zero_tolerance"] = _feasibility_tolerance(settings, zero_tolerance)
    _write_json(json_path, metadata)
    if failure_path.exists():
        failure_path.unlink()


def solve_reference_problem(
    X: sparse.spmatrix,
    labels: np.ndarray,
    D: sparse.spmatrix,
    mu: float,
    *,
    dataset_name: str,
    cache_dir: str | Path,
    solver_settings: dict[str, Any],
    certification: dict[str, float],
    ecos_settings: dict[str, Any] | None = None,
    solver_order: tuple[str, ...] = ("CLARABEL", "ECOS"),
    zero_tolerance: float = 1.0e-12,
    recompute: bool = False,
) -> ReferenceSolution:
    """Solve and independently certify the exact split model with open-source IPMs."""
    import cvxpy as cp

    requested_solvers = tuple(dict.fromkeys(solver.upper() for solver in solver_order))
    if not requested_solvers or any(
        solver not in {"CLARABEL", "ECOS"} for solver in requested_solvers
    ):
        raise ValueError("solver_order must contain only CLARABEL and/or ECOS.")
    ensure_reference_solvers_available(requested_solvers)
    ecos_settings = dict(ecos_settings or {
        "abstol": 1.0e-9,
        "reltol": 1.0e-9,
        "feastol": 1.0e-9,
        "max_iters": 10_000,
    })
    X = sparse.csc_matrix(X, dtype=np.float64)
    D = sparse.csc_matrix(D, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    digest = problem_hash(X, labels, D, mu)
    npz_path, json_path, failure_path = _cache_paths(Path(cache_dir), dataset_name)
    if not recompute:
        cached = _load_cache(npz_path, json_path, digest)
        if cached is not None:
            objective_limit = certification["objective_relative_tolerance"] * (
                1.0 + abs(cached.solver_objective)
            )
            cache_is_certified = (
                cached.solver in requested_solvers
                and cached.status == cp.OPTIMAL
                and cached.primal_relative_residual
                <= certification["primal_relative_tolerance"]
                and cached.normalized_kkt_residual
                <= certification["normalized_kkt_tolerance"]
                and cached.objective_discrepancy <= objective_limit
                and np.isfinite(
                    np.concatenate(
                        [
                            cached.x,
                            cached.y,
                            cached.dual,
                            [cached.f_star, cached.solver_objective],
                        ]
                    )
                ).all()
            )
            if cache_is_certified:
                return cached

    n_samples, n_features = X.shape
    n_edges = D.shape[0]
    x_var = cp.Variable(n_features, name="x")
    y_var = cp.Variable(n_edges, name="y")
    equality = D @ x_var - y_var == 0
    logistic = cp.sum(cp.logistic(-cp.multiply(labels, X @ x_var))) / n_samples
    problem = cp.Problem(cp.Minimize(logistic + mu * cp.norm1(y_var)), [equality])

    diagnostics: dict[str, Any] = {
        "dataset": dataset_name,
        "problem_hash": digest,
        "shape": {"samples": n_samples, "features": n_features, "edges": n_edges},
        "solver_order": list(requested_solvers),
        "attempts": [],
    }
    clarabel_errors: list[str] = []
    clarabel_inaccurate_objective: float | None = None
    fallback_reason: str | None = None
    try:
        for solver in requested_solvers:
            profiles = (
                _solver_attempts(solver_settings)
                if solver == "CLARABEL"
                else _ecos_attempts(ecos_settings)
            )
            diagnostics[f"{solver.lower()}_settings"] = [
                {"profile": profile, "settings": settings}
                for profile, settings in profiles
            ]
            solver_errors: list[str] = []
            for attempt_number, (profile, settings) in enumerate(profiles, start=1):
                attempt_info: dict[str, Any] = {
                    "solver": solver,
                    "attempt": attempt_number,
                    "profile": profile,
                    "settings": settings,
                }
                try:
                    with warnings.catch_warnings():
                        if solver == "CLARABEL":
                            warnings.filterwarnings(
                                "ignore",
                                message="Solution may be inaccurate.*",
                                category=UserWarning,
                            )
                        problem.solve(
                            solver=getattr(cp, solver),
                            verbose=False,
                            warm_start=False,
                            **settings,
                        )
                    solution, errors, certification_info = _certify_current_solution(
                        cp,
                        problem,
                        equality,
                        x_var,
                        y_var,
                        X,
                        labels,
                        D,
                        mu,
                        certification,
                        zero_tolerance,
                        settings,
                        digest,
                        solver,
                        profile,
                    )
                    attempt_info.update(certification_info)
                except Exception as exc:
                    solution = None
                    errors = [f"solver exception {type(exc).__name__}: {exc}"]
                    attempt_info.update({"status": "exception", "solver_objective": None})
                diagnostics["attempts"].append(attempt_info)

                if solver == "CLARABEL" and attempt_info.get("status") == cp.OPTIMAL_INACCURATE:
                    clarabel_inaccurate_objective = attempt_info.get("solver_objective")
                if solution is None:
                    solver_errors.append(
                        f"{solver} {profile}: " + "; ".join(errors)
                    )
                    continue

                if solver == "ECOS":
                    discrepancy, disagreement = _cross_solver_discrepancy(
                        solution.solver_objective,
                        clarabel_inaccurate_objective,
                        certification,
                    )
                    attempt_info["cross_solver_objective_discrepancy"] = discrepancy
                    if disagreement is not None:
                        solver_errors.append(f"{solver} {profile}: {disagreement}")
                        continue
                    solution = replace(
                        solution,
                        fallback_reason=fallback_reason,
                        cross_solver_objective_discrepancy=discrepancy,
                    )
                _save_solution(
                    solution,
                    npz_path,
                    json_path,
                    failure_path,
                    settings,
                    zero_tolerance,
                )
                return solution

            if solver == "CLARABEL":
                clarabel_errors = solver_errors
                fallback_reason = (
                    "CLARABEL strict profiles did not certify: "
                    + " | ".join(clarabel_errors)
                )
                diagnostics["fallback_reason"] = fallback_reason
            else:
                diagnostics["ecos_errors"] = solver_errors

        all_errors = clarabel_errors + diagnostics.get("ecos_errors", [])
        raise ReferenceSolveError(
            "No open-source reference solver passed strict certification: "
            + " | ".join(all_errors)
        )
    except Exception as exc:
        diagnostics["error_type"] = type(exc).__name__
        diagnostics["error"] = str(exc)
        _write_json(failure_path, diagnostics)
        if isinstance(exc, ReferenceSolveError):
            raise
        raise ReferenceSolveError(f"Reference solve failed: {exc}") from exc
