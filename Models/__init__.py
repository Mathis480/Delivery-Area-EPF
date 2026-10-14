"""Models package for EPF Delivery Area forecasting."""
from Models.linear import train_lasso, predict_lasso, get_lasso_weights
from Models.csvr import train_csvr, predict_csvr, CSVRModel
from Models.ensemble import compute_rolling_weighted_average, dm_test


def __getattr__(name: str):
    if name in ("MAMLManager", "build_maml_net"):
        import Models.maml_nn as _maml
        val = getattr(_maml, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "train_lasso",
    "predict_lasso",
    "get_lasso_weights",
    "train_csvr",
    "predict_csvr",
    "CSVRModel",
    "MAMLManager",
    "build_maml_net",
    "compute_rolling_weighted_average",
    "dm_test",
]
