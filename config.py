"""
Central configuration for the EPF Delivery Area forecasting framework.
All paths, hyperparameters, and scenario settings in one place.
"""
import os

# ==============================================================================
# PATHS
# ==============================================================================
PROJECT_ROOT       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR           = os.path.join(PROJECT_ROOT, "Data")
MASTER_PARQUET     = os.path.join(DATA_DIR, "master_dataset.parquet")
MASTER_CSV         = os.path.join(DATA_DIR, "master_dataset_2021_2024.csv")
FEATURES_CSV       = os.path.join(PROJECT_ROOT, "features.csv")

MODELS_DIR         = os.path.join(PROJECT_ROOT, "Models")
SCALERS_DIR        = os.path.join(MODELS_DIR, "scalers")
TRAINED_MODELS_DIR = os.path.join(MODELS_DIR, "trained_models")
RESULTS_DIR        = os.path.join(PROJECT_ROOT, "Results")

os.makedirs(SCALERS_DIR, exist_ok=True)
os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ==============================================================================
# DATA CONFIGURATION
# ==============================================================================
SCENARIO       = "scenario1"          # Feature scenario to use (from features.csv)
ZONES          = ["DE1", "DE2", "DE3", "DE4"]
N_QH           = 96                   # Number of quarter-hour positions per day

# Canonical target column pattern: {zone}_VWAP_0to30
# Benchmark (naive): {zone}_VWAP_90to105
TARGET_COL_PATTERN    = "{zone}_VWAP_0to30"
BENCHMARK_COL_PATTERN = "{zone}_VWAP_90to105"

# ==============================================================================
# TRAINING WINDOW
# ==============================================================================
TEST_START     = "2024-01-01"
TEST_END       = "2024-12-31"
LOOKBACK_DAYS  = 822          # Full pre-2024 history: 2021-10-01 to 2023-12-31 (approx 2.25 years)
VAL_DAYS       = 30           # Last 30 days of lookback window used as validation set

# ==============================================================================
# FEATURE ENGINEERING & TIMING RULES
# ==============================================================================
# Ex-post feature lag (fundamentals, balancing reserves)
SHIFT_QH                  = 8            # 2 hours = 8 quarter-hours

# Prediction cutoff & operational reporting delay (Puć et al. rounded to 15-min grid)
PREDICTION_CUTOFF_MINUTES = 60           # Nominal cutoff before delivery
REPORTING_DELAY_MINUTES   = 30           # Publishing/processing delay buffer
EFFECTIVE_CUTOFF_MINUTES  = PREDICTION_CUTOFF_MINUTES + REPORTING_DELAY_MINUTES  # 90 minutes

# Neighboring contract price trajectories (k relative to target QH)
# Leakage-safe rule: window ending A minutes before neighbor delivery requires A >= 90 + 15*k
NEIGHBOR_QH_OFFSETS       = [-4, -3, -2, -1, 1, 2]
NEIGHBOR_SAFE_VWAP_WINDOWS = {
    -4: "30to45",
    -3: "45to60",
    -2: "60to75",
    -1: "75to90",
    1: "105to120",
    2: "120to135",
}

# ==============================================================================
# MODEL HYPERPARAMETERS
# ==============================================================================

# cSVR (Marcjasz et al. kernel-based SVR)
CSVR_EPSILON   = 0.1
CSVR_C         = 1.0
CSVR_Q_KERNEL  = 0.75         # Quantile for intermediate kernel threshold
CSVR_Q_DATA    = 0.5          # Quantile for data normalization in cSVR
CSVR_NORM      = 2            # L2 norm for cdist

# LASSO
LASSO_MAX_ITER = 3000
LASSO_TOL      = 1e-3
LASSO_CV_DAYS  = 14           # Temporal CV window for alpha selection (last 14 days as holdout folds)

# MAML-NN
MAML_HIDDEN_SIZES      = [128, 64]   # MLP layer sizes
MAML_META_LR           = 1e-3        # Outer loop (meta) learning rate
MAML_INNER_LR          = 1e-2        # Inner loop (task) learning rate
MAML_INNER_STEPS       = 5           # Gradient steps during adaptation
MAML_META_EPOCHS       = 100         # Warm-start shared backbone training epochs
MAML_BACKBONE_BATCH    = 512         # Batch size for backbone pre-training
MAML_ADAPT_EPOCHS      = 1           # Inner adaptation epochs per test day
MAML_DROPOUT           = 0.1

# ==============================================================================
# BACKTEST CONFIGURATION
# ==============================================================================
N_WORKERS        = 4          # Parallel processes for workers if needed
CHECKPOINT_EVERY = 10         # Flush results CSV every N test days
