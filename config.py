"""
Central configuration for the EPF Delivery Area forecasting framework.
All paths, hyperparameters, and scenario settings in one place.
"""
import os

# PATHS
PROJECT_ROOT       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR           = os.path.join(PROJECT_ROOT, "Data")
MASTER_PARQUET     = os.path.join(DATA_DIR, "master_dataset.parquet")
FEATURES_CSV       = os.path.join(PROJECT_ROOT, "features.csv")

MODELS_DIR         = os.path.join(PROJECT_ROOT, "Models")
SCALERS_DIR        = os.path.join(MODELS_DIR, "scalers")
TRAINED_MODELS_DIR = os.path.join(MODELS_DIR, "trained_models")
RESULTS_DIR        = os.path.join(PROJECT_ROOT, "Results")

os.makedirs(SCALERS_DIR, exist_ok=True)
os.makedirs(TRAINED_MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# DATA CONFIGURATION
ZONES          = ["DE1", "DE2", "DE3", "DE4"]
N_QH           = 96

TARGET_COL_PATTERN    = "{zone}_VWAP_0to30"
BENCHMARK_COL_PATTERN = "{zone}_VWAP_90to105"


# TRAINING WINDOW
TEST_START     = "2024-01-01"
TEST_END       = "2024-12-31"
LOOKBACK_DAYS  = 822          # Full pre-2024 history: 2021-10-01 to 2023-12-31 (approx 2.25 years)
VAL_DAYS       = 30           # Last 30 days of lookback window used as validation set

# FEATURE ENGINEERING & TIMING RULES

SHIFT_QH                  = 8            # 2 hours = 8 quarter-hours - ex-post feature lag (fundamentals, balancing reserves)
PREDICTION_CUTOFF_MINUTES = 60           # Our Point of prediction
REPORTING_DELAY_MINUTES   = 30           # Delay
EFFECTIVE_CUTOFF_MINUTES  = PREDICTION_CUTOFF_MINUTES + REPORTING_DELAY_MINUTES 

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

# MODEL HYPERPARAMETERS

# cSVR (Corrected Support Vector Regression, Puć & Janczura, 2024, Eq. 12)
CSVR_EPSILON                 = 0.1
CSVR_C                       = 1.0
CSVR_Q_KERNEL                = 0.75         # Quantile for feature Laplace kernel threshold
CSVR_Q_DATA                  = 0.5          # Quantile for feature data normalization in cSVR
CSVR_NORM                    = 2            # L2 norm for cdist
CSVR_USE_GAUSSIAN_CORRECTION = True         # Enable multiplicative Gaussian naive correction (Puć & Janczura, 2024, Eq. 12)
CSVR_Q_KERNEL_NAIVE          = 0.75         # Quantile for naive Gaussian kernel threshold
CSVR_Q_DATA_NAIVE            = 0.75         # Quantile for naive price difference normalization
CSVR_USE_CORR_FILTER         = True         # Canonical benchmark default (Puć & Janczura 2024: True | Ablation without filter: False)
CSVR_CORR_THRESH_S1          = 0.80         # Correlation threshold for S1 Macro
CSVR_CORR_THRESH_S2          = 0.95         # Correlation threshold for S2 Neighbor

# LASSO
LASSO_MAX_ITER = 3000
LASSO_TOL      = 1e-3
LASSO_CV_DAYS  = 14           # Temporal CV window for alpha selection (last 14 days as holdout folds)

# MAML-NN Architecture & Hyperparameters
MAML_HIDDEN_SIZES         = [128, 64]   
MAML_DROPOUT              = 0.0         # No dropout during meta-training
MAML_ACTIVATION           = "tanh"      
MAML_L1_REG               = 1e-4        
MAML_META_LR              = 5e-4        # Outer loop (meta) learning rate
MAML_INNER_LR             = 0.0025      # Fast inner loop adaptation learning rate
MAML_INNER_STEPS          = 8           # Gradient steps during fast adaptation
MAML_SUPPORT_WEIGHTING    = "soft_kernel" # Support weighting: 'uniform' or 'soft_kernel' -> 
MAML_KERNEL_TAU           = 4.0         # Temperature scale factor for soft kernel support weighting
MAML_PROX_SHRINK          = 1e-5        # Proximal soft-thresholding shrinkage operator
MAML_CLIP_RESIDUAL        = None        # Unconstrained NN residual
MAML_USE_LINEAR_BYPASS    = False       
MAML_BYPASS_SATURATION_M  = 0.15        # Smooth tanh saturation scale for Scenario 3: p_safe = m * tanh(p_lin / m)
MAML_SUPPORT_SELECTION    = "regime_l1" # L1 distance nearest regimes within pool window
MAML_SUPPORT_K            = 28          # Number of support days selected for inner adaptation

MAML_META_EPOCHS          = 6           # Warm-start shared backbone training epochs
MAML_BACKBONE_BATCH       = 256         # Batch size for backbone pre-training
MAML_SEEDS                = [42, 123, 999] # Multi-seed deep ensembling seeds per feature set (Lago et al., 2021)

# Rolling Weighted Average Ensemble (Puć & Janczura, 2024, Eq. 25; Marcjasz et al., 2018)
ENSEMBLE_CALIB_DAYS       = 14          # Rolling calibration window in days (W in Eq. 25)
ENSEMBLE_POWER            = 1.0         
ENSEMBLE_QH_ADAPTIVE      = True        # Quarter-hour specific adaptive weighting (Puć & Janczura, 2024; prevents D-1 late-night look-ahead)

# BACKTEST CONFIGURATION
N_WORKERS                 = 4           # Parallel CPU workers for multi-zone execution
