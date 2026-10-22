"""
Central configuration for the EPF delivery area forecasting framework.
"""
import os

# PATHS
PROJECT_ROOT   = os.path.dirname(os.path.abspath(__file__))
MASTER_PARQUET = os.path.join(PROJECT_ROOT, "Data", "master_dataset.parquet")
FEATURES_CSV   = os.path.join(PROJECT_ROOT, "features.csv")
RESULTS_DIR    = os.path.join(PROJECT_ROOT, "Results")

# DATA
ZONES = ["DE1", "DE2", "DE3", "DE4"]
N_QH  = 96

TARGET_COL_PATTERN    = "{zone}_VWAP_0to30"
BENCHMARK_COL_PATTERN = "{zone}_VWAP_90to105"

# BACKTEST WINDOW
DATA_START    = "2021-10-01"  # first delivery day used for calibration
TEST_START    = "2024-01-01"
TEST_END      = "2024-12-31"
LOOKBACK_DAYS = 822           # rolling calibration window (2021-10-01 to 2023-12-31 for the first test day)
RUN_DIR       = os.path.join(RESULTS_DIR, f"backtest_{TEST_START[:4]}")

# INFORMATION SET
# Ex-post data must be published by the forecast time (60 min before delivery); the publication delay counts from the
# end of a quarter-hour, so lag = (60 + delay + 15) / 15 quarter-hours relative to the target delivery (features.csv)
LAG_QH_BALANCING = 6  # netztransparenz.de operational aFRR/mFRR: published 15 min after the quarter-hour
LAG_QH_ENTSOE    = 9  # ENTSO-E actual load/generation (and cross-border trading): up to 60 min after the quarter-hour
EFFECTIVE_CUTOFF_MINUTES = 90  # forecast 60 min before delivery + 30 min publication buffer

# Neighboring contracts: latest admissible window of contract s+k (ends 90 min before delivery of s)
NEIGHBOR_SAFE_VWAP_WINDOWS = {
    -4: "30to45",
    -3: "45to60",
    -2: "60to75",
    -1: "75to90",
    1: "105to120",
    2: "120to135",
}

# FEATURE SETS
FEATURE_SET_MACRO       = "set1_macro"
FEATURE_SET_NEIGHBOR    = "set2_neighbor"
FEATURE_SET_FUNDAMENTAL = "set3_fundamental"
FEATURE_SET_BALANCE     = "set4_balance"
FEATURE_SETS = [FEATURE_SET_MACRO, FEATURE_SET_NEIGHBOR, FEATURE_SET_FUNDAMENTAL, FEATURE_SET_BALANCE]
SET_TAGS     = {FEATURE_SET_MACRO: "s1", FEATURE_SET_NEIGHBOR: "s2", FEATURE_SET_FUNDAMENTAL: "s3", FEATURE_SET_BALANCE: "s4"}

# Correlation filter of Puć & Janczura (2024), applied to S1 and S2 (LASSO, cSVR replication)
CORR_THRESHOLDS = {FEATURE_SET_MACRO: 0.80, FEATURE_SET_NEIGHBOR: 0.95}
# Exogenous variables exempt from the correlation filter, as in Puć & Janczura (2024): auction prices (incl. lags), weekday dummies
CORR_FILTER_EXEMPT_PREFIXES = ("Dayahead_Auction_hourly", "Intraday Auction Price", "Weekday_")
# Exact duplicates within a calibration window (e.g. forecast deviation = actual when the solar forecast is zero)
DUPLICATE_CORR = 1.0 - 1e-10

# cSVR (Puć & Janczura, 2024, Eq. 12)
CSVR_EPSILON        = 0.1
CSVR_C              = 1.0
CSVR_Q_KERNEL       = 0.75  # Laplace kernel: theoretical quantile
CSVR_Q_DATA         = 0.5   # Laplace kernel: empirical quantile of feature distances
CSVR_Q_KERNEL_NAIVE = 0.75  # Gaussian correction: theoretical quantile
CSVR_Q_DATA_NAIVE   = 0.75  # Gaussian correction: empirical quantile of squared naive differences
CSVR_NORM           = 2     # L2 norm in the Laplace kernel

# LASSO
LASSO_MAX_ITER = 3000
LASSO_CV_DAYS  = 14         # expanding-window CV over the last 14 calibration days

# FA-NN (fast local adaptation; Finn et al., 2017 for the FOMAML ablation)
FANN_HIDDEN_SIZES = [128, 64]
FANN_ACTIVATION   = "tanh"
FANN_LR           = 5e-4    # Adam learning rate (backbone pre-training and FOMAML outer loop)
FANN_EPOCHS       = 6       # backbone pre-training epochs
FANN_BATCH        = 256     # backbone pre-training batch size
FANN_SUPPORT_K    = 28      # support days selected for local adaptation
FANN_INNER_LR     = 0.0025  # local adaptation learning rate
FANN_INNER_STEPS  = 8       # local adaptation gradient steps

# Training modes: "fann" (supervised backbone), "fomaml" (fann + FOMAML), "fomaml_scratch" (FOMAML from random init)
FANN_TRAINING_MODES = ["fann", "fomaml", "fomaml_scratch"]
# Optional linear bypass: "none" or "lasso" (network fits the residual of a saturated LASSO forecast)
FANN_BYPASS_MODES   = ["none", "lasso"]
FANN_BYPASS_SATURATION_M = 0.15

FOMAML_EPOCHS         = 5   # meta-epochs after the supervised backbone
FOMAML_SCRATCH_EPOCHS = 8   # compute-matched to backbone pre-training + FOMAML
FOMAML_META_BATCH     = 16  # tasks per meta-update
FOMAML_LAYER_NORM     = True  # layer normalization before each activation (validated on 2023, 14 of 16 blocks)

RANDOM_SEED = 42

# FORECAST AVERAGING (Marcjasz et al., 2018; Puć & Janczura, 2024): per-product inverse-MAE weights
ENSEMBLE_CALIB_DAYS = 14
ENSEMBLE_POWER      = 1.0

# RUNTIME
N_WORKERS = 4  # parallel zones
