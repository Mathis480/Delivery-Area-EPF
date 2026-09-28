"""
Models package for EPF Delivery Area forecasting.
Includes:
  - linear: Ordinary Least Squares (LR) and LASSO with holdout CV
  - csvr: Corrected Support Vector Regression with Laplacian kernel (Puć & Janczura, 2024)
  - maml_nn: Model-Agnostic Meta-Learning Neural Network with Linear Bypass and GNCL (Buschjäger et al., 2020)
  - ensemble: Rolling Inverse-MAE Weighted Forecast Averaging (Puć & Janczura, 2024; Bates & Granger, 1969)
"""
from Models.linear import train_lr, predict_lr, get_lr_weights, train_lasso, predict_lasso, get_lasso_weights
from Models.csvr import train_csvr, predict_csvr, CSVRModel
from Models.maml_nn import MAMLManager, build_maml_net
from Models.ensemble import compute_rolling_weighted_average, compute_rolling_intelligent_ensemble

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
]
