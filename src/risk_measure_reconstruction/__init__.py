"""Convex, cash-additive reconstruction from finitely many risk quotes.

Imports are intentionally lazy: the CLI sets BLAS thread defaults before
loading NumPy. All public classes are also available from their own modules.
"""

from importlib import import_module

__version__ = "0.1.0"
__all__ = [
    "ConicReconstructor",
    "RandomDualReconstructor",
    "GBM",
    "Heston",
    "FiniteModel",
    "FiniteGeometry",
    "CallGeometry",
    "PopulationCallReconstructor",
    "population_call_risk",
    "common_lipschitz_budget",
    "weighted_es_loss",
]
_MODULES = {
    "ConicReconstructor": "risk_reconstruction",
    "RandomDualReconstructor": "risk_reconstruction",
    "weighted_es_loss": "risk_reconstruction",
    "GBM": "gbm_risk",
    "CallGeometry": "gbm_risk",
    "PopulationCallReconstructor": "gbm_risk",
    "population_call_risk": "gbm_risk",
    "common_lipschitz_budget": "gbm_risk",
    "Heston": "heston_risk",
    "FiniteModel": "finite_risk",
    "FiniteGeometry": "finite_risk",
}


def __getattr__(name):
    if name in _MODULES:
        value = getattr(import_module(f".{_MODULES[name]}", __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
