"""Central configuration for the GGLR benchmark."""

from __future__ import annotations


DATASET_FILES = {
    "a9a": {"train": "a9a", "test": "a9a.t"},
    "w8a": {"train": "w8a", "test": "w8a.t"},
    "ijcnn1": {"train": "ijcnn1.bz2", "test": "ijcnn1.t.bz2"},
    "madelon": {"train": "madelon", "test": "madelon.t"},
}


GLOBAL_SETTINGS = {
    "mu": 1.0e-2,
    "rho": 1.0,
    "max_iter": 4_000,
    "eval_every": 20,
    "seeds": [2018, 2025, 2026],
    "split_test_size": 0.20,
    "split_seed": 2026,
    "graph_k": 5,
    "graph_zero_tolerance": 1.0e-14,
    "kkt_zero_tolerance": 1.0e-12,
    "time_grid_points": 200,
}


PLOT_SETTINGS = {
    # Set to True only when IFO-budget comparisons are needed.
    "include_ifo_plots": False,
}


CLARABEL_SETTINGS = {
    "tol_gap_abs": 1.0e-9,
    "tol_gap_rel": 1.0e-9,
    "tol_feas": 1.0e-9,
    "max_iter": 2_000,
    "equilibrate_enable": True,
    "presolve_enable": True,
    "iterative_refinement_enable": True,
}


ECOS_SETTINGS = {
    "abstol": 1.0e-9,
    "reltol": 1.0e-9,
    "feastol": 1.0e-9,
    "max_iters": 10_000,
}


REFERENCE_CERTIFICATION = {
    "primal_relative_tolerance": 1.0e-7,
    "normalized_kkt_tolerance": 1.0e-6,
    "objective_relative_tolerance": 1.0e-7,
}


# ``step_multiplier`` multiplies the data-dependent safe step calculated in main.py.
# AIL-SVRG-ADMM ``p_min`` accepts "inverse_n", "batch_over_n", or a numeric value in (0, 1].
ALGORITHM_PARAMS = {
    "STOC-ADMM": {"batch_size": 32, "step_multiplier": 1.0},
    "SAG-ADMM": {"batch_size": 1, "step_multiplier": 0.75},
    "SAGA-ADMM": {"batch_size": 1, "step_multiplier": 0.75},
    "SVRG-ADMM": {
        "batch_size": 32,
        "step_multiplier": 1.0,
        "inner_iter": "auto"
    },
    "ASVRG-ADMM": {
        "batch_size": 32,
        "step_multiplier": 0.5,
        "inner_iter": "auto",
        "theta": 0.5,
    },
    # "SPIDER-ADMM": {
    #     "batch_size": "sqrt_n",
    #     "step_multiplier": 1.0,
    #     "refresh_period": "sqrt_n",
    # },
    "SPIDER-ADMM": {
        "batch_size": 64,
        "step_multiplier": 1.0,
        "refresh_period": 100,
    },
    "AIL-SVRG-ADMM": {
        "batch_size": 64,
        "step_multiplier": 1.35,
        "tau": 0.10,
        "varrho": 3,
        "beta_y": 0.0,
        "p_min": 0.002,
    },
    "AIL-SVRG-ADMM-NoMom": {
        "batch_size": 64,
        "step_multiplier": 1.0,
        "tau": 0.0,
        "varrho": 3,
        "beta_y": 0.0,
        "p_min": 0.002,
    },
    "AIL-SVRG-ADMM-Fixed-p": {
        "batch_size": 64,
        "step_multiplier": 1.0,
        "tau": 0.0,
        "varrho": 3,
        "beta_y": 0.0,
        "p_min": 0.002,
        "fixed_probability": "cost_matched",
    },
    "AIL-SVRG-ADMM-WithCorr": {
        "batch_size": 64,
        "step_multiplier": 1.0,
        "tau": 0.0,
        "varrho": 3,
        "beta_y": 0.0,
        "p_min": 0.002,
        "enable_correction": True,
    },
}


ALGORITHM_ORDER = list(ALGORITHM_PARAMS)
