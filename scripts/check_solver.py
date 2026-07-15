"""Verify the free CVXPY/CLARABEL/ECOS reference-solver stack."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

import numpy as np
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CLARABEL_SETTINGS, ECOS_SETTINGS, REFERENCE_CERTIFICATION  # noqa: E402
from utils.optimizer import (  # noqa: E402
    ensure_reference_solvers_available,
    solve_reference_problem,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    versions = ensure_reference_solvers_available()
    print(f"CVXPY version: {versions['CVXPY']}")
    print(f"CLARABEL version: {versions['CLARABEL']}")
    print(f"ECOS version: {versions['ECOS']}")

    X = sparse.csr_matrix(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
    )
    labels = np.ones(4)
    D = sparse.csr_matrix([[1.0, -1.0]])
    if args.verbose:
        print(f"CLARABEL settings: {CLARABEL_SETTINGS}")
        print(f"ECOS settings: {ECOS_SETTINGS}")

    with tempfile.TemporaryDirectory() as temporary:
        clarabel_solution = solve_reference_problem(
            X,
            labels,
            D,
            1.0e-2,
            dataset_name="clarabel_check",
            cache_dir=temporary,
            solver_settings=CLARABEL_SETTINGS,
            ecos_settings=ECOS_SETTINGS,
            certification=REFERENCE_CERTIFICATION,
            recompute=True,
        )
        ecos_solution = solve_reference_problem(
            X,
            labels,
            D,
            1.0e-2,
            dataset_name="ecos_check",
            cache_dir=temporary,
            solver_settings=CLARABEL_SETTINGS,
            ecos_settings=ECOS_SETTINGS,
            certification=REFERENCE_CERTIFICATION,
            solver_order=("ECOS",),
            recompute=True,
        )
    for solution in (clarabel_solution, ecos_solution):
        print(f"{solution.solver} status: {solution.status}")
        print(f"{solution.solver} certified objective: {solution.f_star:.12g}")
        print(f"{solution.solver} normalized KKT residual: {solution.normalized_kkt_residual:.3e}")
    print("CLARABEL and ECOS checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
