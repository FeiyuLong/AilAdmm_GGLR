"""Algorithm entry points for the GGLR benchmark."""

from .ail_svrg_admm import run_ail_svrg_admm
from .ail_svrg_admm_fixed_p import run_ail_svrg_admm_fixed_p
from .ail_svrg_admm_no_mom import run_ail_svrg_admm_no_mom
from .ail_svrg_admm_with_corr import run_ail_svrg_admm_with_corr
from .asvrg_admm import run_asvrg_admm
from .sag_admm import run_sag_admm
from .saga_admm import run_saga_admm
from .spider_admm import run_spider_admm
from .stoc_admm import run_stoc_admm
from .svrg_admm import run_svrg_admm

__all__ = [
    "run_stoc_admm",
    "run_sag_admm",
    "run_saga_admm",
    "run_svrg_admm",
    "run_asvrg_admm",
    "run_spider_admm",
    "run_ail_svrg_admm",
    "run_ail_svrg_admm_no_mom",
    "run_ail_svrg_admm_fixed_p",
    "run_ail_svrg_admm_with_corr",
]
