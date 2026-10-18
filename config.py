"""
Central configuration for the EPF Delivery Area forecasting framework.
All paths, hyperparameters, and scenario settings in one place.
"""
import os

# PATHS
PROJECT_ROOT       = os.path.dirname(os.path.abspath(__file__))
MASTER_PARQUET     = os.path.join(PROJECT_ROOT, "Data", "master_dataset.parquet")
FEATURES_CSV       = os.path.join(PROJECT_ROOT, "features.csv")
SCALERS_DIR        = os.path.join(PROJECT_ROOT, "Models", "scalers")
TRAINED_MODELS_DIR = os.path.join(PROJECT_ROOT, "Models", "trained_models")
RESULTS_DIR        = os.path.join(PROJECT_ROOT, "Results")

os.makedirs(SCALERS_DIR, exist_ok=True)
os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# DATA CONFIGURATION
ZONES          = ["DE1", "DE2", "DE3", "DE4"]
N_QH           = 96

TARGET_COL_PATTERN    = "{zone}_VWAP_0to30"
BENCHMARK_COL_PATTERN = "{zone}_VWAP_90to105"

# TRAINING & EVALUATION WINDOW
TEST_START     = "2024-01-01"
TEST_END       = "2024-12-31"
LOOKBACK_DAYS  = 822          # Pre-2024 history: 2021-10-01 to 2023-12-31 (~2.25 years)
VAL_DAYS       = 0            # No artificial validation holdout; training window runs through T-1

# FEATURE ENGINEERING & TIMING RULES
SHIFT_QH                  = 8   # 2 hours = 8 quarter-hours lag for ex-post fundamentals
EFFECTIVE_CUTOFF_MINUTES  = 90  # 60 min gate closure + 30 min reporting buffer

# Neighboring contract price trajectories (k relative to target QH)
NEIGHBOR_QH_OFFSETS       = [-4, -3, -2, -1, 1, 2]
NEIGHBOR_SAFE_VWAP_WINDOWS = {
    -4: "30to45",
    -3: "45to60",
    -2: "60to75",
    -1: "75to90",
    1: "105to120",
    2: "120to135",
}

# Feature Sets (4 German Delivery Area Partitions)
FEATURE_SET_MACRO       = "set1_macro"
FEATURE_SET_NEIGHBOR    = "set2_neighbor"
FEATURE_SET_FUNDAMENTAL = "set3_fundamental"
FEATURE_SET_BALANCE     = "set4_balance"
S4_USE_STRUCTURED_RATIOS = True        # Use structured 16-feature flow-to-load and momentum ratios for S4

# MODEL HYPERPARAMETERS

# cSVR (Puć & Janczura, 2024, Eq. 12)
CSVR_EPSILON                 = 0.1
CSVR_C                       = 1.0
CSVR_Q_KERNEL                = 0.75         # Quantile for feature Laplace kernel threshold
CSVR_Q_DATA                  = 0.5          # Quantile for feature distance normalization
CSVR_NORM                    = 2            # L2 norm for cdist
CSVR_USE_GAUSSIAN_CORRECTION = True         # Enable multiplicative Gaussian naive correction (Eq. 12)
CSVR_Q_KERNEL_NAIVE          = 0.75         # Quantile for naive Gaussian kernel threshold
CSVR_Q_DATA_NAIVE            = 0.75         # Quantile for naive price difference normalization
CSVR_CORR_THRESH_S1          = 0.80         # Correlation threshold for S1 Macro
CSVR_CORR_THRESH_S2          = 0.95         # Correlation threshold for S2 Neighbor

# LASSO
LASSO_MAX_ITER = 3000
LASSO_TOL      = 1e-3
LASSO_CV_DAYS  = 14           # Temporal CV window for alpha selection (last 14 days as holdout folds)

# MAML-NN (Finn et al., 2017)
MAML_HIDDEN_SIZES         = [128, 64]
MAML_ACTIVATION           = "tanh"
MAML_L1_REG               = 1e-4
MAML_META_LR              = 5e-4            # Adam learning rate for backbone pre-training
MAML_INNER_LR             = 0.0025          # Fast inner loop adaptation learning rate
MAML_INNER_STEPS          = 8               # Gradient steps during fast adaptation
MAML_PROX_SHRINK          = 1e-5            # Proximal soft-thresholding shrinkage operator
MAML_BYPASS_MODES         = ["nobypass", "optlasso"] # Active MAML bypass variants ('nobypass', 'optlasso')
MAML_TRAINING_MODES       = ["warmstart", "maml"]    # Active MAML training variants ('warmstart', 'maml')
MAML_BYPASS_SATURATION_M  = 0.15            # Smooth tanh saturation scale for linear bypass: p_safe = m * tanh(p_lin / m)
MAML_SUPPORT_K            = 28              # Number of support days selected for inner adaptation
MAML_META_EPOCHS          = 6               # Warm-start shared backbone training epochs
MAML_BILEVEL_EPOCHS       = 5               # Meta-epochs for bilevel MAML training (E_meta in paper)
MAML_OUTER_CLIP           = 0.5             # Meta-gradient clipping threshold for bilevel MAML (Antoniou et al., 2019)
MAML_BACKBONE_BATCH       = 256             # Batch size for backbone pre-training
MAML_META_BATCH           = 16              # FOMAML tasks (QH, query day) averaged per meta-update
MAML_N_FOLDS              = 5               # Temporal folds over the training window (fold 0 = warm-start/support only)
MAML_WARMSTART_SCOPE      = "fold0"         # 'fold0': maml-mode warm-start sees only fold 0 (query days unseen); 'all': legacy, all days (warmstart mode always uses all days)
RANDOM_SEED               = 42              # Central deterministic random seed across framework

# Rolling Inverse-MAE Ensemble (Puć & Janczura, 2024; Marcjasz et al., 2018)
ENSEMBLE_CALIB_DAYS       = 14              # Rolling calibration window in days (W)
ENSEMBLE_POWER            = 1.0             # Ensemble power parameter (linear weighting)
ENSEMBLE_QH_ADAPTIVE      = True            # Quarter-hour specific adaptive weighting (prevents D-1 late-night look-ahead)

# RUNTIME
N_WORKERS                 = 4               # Parallel CPU workers for multi-zone execution
