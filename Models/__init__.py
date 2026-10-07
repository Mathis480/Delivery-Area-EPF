"""
Models package for EPF Delivery Area forecasting.
Includes:
  - linear: Ordinary Least Squares (LR) and LASSO with holdout CV
  - csvr: Corrected Support Vector Regression with Laplacian kernel (Puć & Janczura, 2024)
  - maml_nn: Model-Agnostic Meta-Learning Neural Network with Linear Bypass (Finn et al., 2017)
  - ensemble: Rolling Inverse-MAE Weighted Forecast Averaging (Puć & Janczura, 2024; Bates & Granger, 1969)
"""
from Models.linear import train_lr, predict_lr, get_lr_weights, train_lasso, predict_lasso, get_lasso_weights
from Models.csvr import train_csvr, predict_csvr, CSVRModel
from Models.ensemble import compute_rolling_weighted_average, compute_rolling_intelligent_ensemble, dm_test


def __getattr__(name: str):
    if name in ("MAMLManager", "build_maml_net"):
        import Models.maml_nn as _maml
        val = getattr(_maml, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
    "train_lr",
    "predict_lr",
    "get_lr_weights",
    "train_lasso",
    "predict_lasso",
    "get_lasso_weights",
    "train_csvr",
    "predict_csvr",
    "CSVRModel",
    "MAMLManager",
    "build_maml_net",
    "compute_rolling_weighted_average",
    "compute_rolling_intelligent_ensemble",
    "dm_test",
]

