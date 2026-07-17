"""Algorithm entry points for the GGLR benchmark."""

from .ailsvrg_admm import run_ailsvrg_admm
from .ailsvrg_admm_fixed_p import run_ailsvrg_admm_fixed_p
from .ailsvrg_admm_no_mom import run_ailsvrg_admm_no_mom
from .ailsvrg_admm_with_corr import run_ailsvrg_admm_with_corr
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
    "run_ailsvrg_admm",
    "run_ailsvrg_admm_no_mom",
    "run_ailsvrg_admm_fixed_p",
    "run_ailsvrg_admm_with_corr",
]
