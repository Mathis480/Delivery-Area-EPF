"""
Rolling per-QH backtest for the EPF Delivery Area framework.

Outer loop: test days (e.g. 2024-01-01 → 2024-12-31)
Inner loop: 4 zones × 96 QH = 384 model calls per test day

For each (zone, qh_idx, test_date):
  1. QHDataLoader.get_window() → leakage-free scaled arrays (ad hoc, in RAM)
  2. LR  → trains on X_trainval
  3. LASSO → selects alpha on val split, fits on X_trainval
  4. cSVR → trains on X_trainval
  5. MAML-NN → inner-loop adapt on X_train, monitor on X_val
  6. All predictions backtransformed: y_eur = y_pred_diff + benchmark_eur
  7. Results appended to CSV; resumes from last completed row if interrupted.
"""
from __future__ import annotations

import argparse
import gc
import logging
import os
import time
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (
    CHECKPOINT_EVERY,
    LOOKBACK_DAYS,
    N_QH,
    RESULTS_DIR,
    TEST_END,
    TEST_START,
    TRAINED_MODELS_DIR,
    VAL_DAYS,
    ZONES,
)
from data_loader import QHDataLoader
from Models.linear import (
    get_lr_weights,
    predict_lasso,
    predict_lr,
    save_trained_model,
    train_lasso,
    train_lr,
)
from Models.csvr import predict_csvr, train_csvr
from Models.maml_nn import MAMLManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


def _mae(y_true: float, y_pred: float) -> float:
    return abs(y_true - y_pred)


def _backtransform(y_diff_pred: float, benchmark_eur: float) -> float:
    return y_diff_pred + benchmark_eur


def _results_path(zone: str, run_tag: str) -> str:
    return os.path.join(RESULTS_DIR, f"results_{zone}_{run_tag}.csv")


def _already_done(df_existing: pd.DataFrame, zone: str, qh: int, date: str) -> set:
    if df_existing is None or len(df_existing) == 0:
        return set()
    mask = (
        (df_existing["zone"] == zone)
        & (df_existing["qh_idx"] == qh)
        & (df_existing["test_date"] == date)
    )
    sub = df_existing[mask]
    if len(sub) == 0:
        return set()
    return set(sub["model"].tolist())


def _collect_backbone_data(
    loader: QHDataLoader,
    zone: str,
    pretrain_date: str = TEST_START,
    lookback: int = LOOKBACK_DAYS,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, float]:
    """Collect 96 QH windows on a representative pretrain date for MAML backbone."""
    log.info(f"[Backbone] Collecting data for {zone} from {pretrain_date}")
    qh_windows = []
    X_all_list, y_all_list = [], []

    for qh_idx in range(N_QH):
        w = loader.get_window(
            zone, qh_idx, pretrain_date, lookback_days=lookback,
            filter_zero_var=False,  # MAML needs fixed dimension across all 96 QH
        )
        if w is not None and len(w.X_trainval) > 10:
            qh_windows.append((w.X_train, w.y_train))
            X_all_list.append(w.X_trainval)
            y_all_list.append(w.y_trainval)
        else:
            qh_windows.append((np.zeros((1, loader.n_features(zone))), np.zeros(1)))

    if X_all_list:
        X_all = np.vstack(X_all_list)
        y_all = np.concatenate(y_all_list)
        # Train global linear model to initialize the bypass layer
        lr_global = train_lr(X_all, y_all)
        lr_coef, lr_intercept = get_lr_weights(lr_global)
    else:
        n_feat = loader.n_features(zone)
        lr_coef = np.zeros(n_feat)
        lr_intercept = 0.0

    return qh_windows, lr_coef, lr_intercept


def run_backtest(
    zone: str,
    test_start: str = TEST_START,
    test_end: str = TEST_END,
    lookback_days: int = LOOKBACK_DAYS,
    val_days: int = VAL_DAYS,
    use_lr: bool = True,
    use_lasso: bool = True,
    use_csvr: bool = True,
    use_maml: bool = True,
    save_scalers: bool = False,
    save_models: bool = False,
    run_tag: str | None = None,
    maml_backbone_date: str = TEST_START,
    verbose: bool = False,
):
    if run_tag is None:
        run_tag = datetime.now().strftime("%Y%m%d_%H%M")

    results_path = _results_path(zone, run_tag)
    log.info(f"=== Backtest: {zone} | {test_start} → {test_end} | lookback={lookback_days}d ===")
    log.info(f"Results → {results_path}")

    df_existing = None
    if os.path.exists(results_path):
        df_existing = pd.read_csv(results_path)
        log.info(f"Resuming: {len(df_existing)} rows already in results file")

    test_dates = pd.date_range(test_start, test_end, freq="D")
    loader = QHDataLoader(verbose=verbose)
    n_features = loader.n_features(zone)
    log.info(f"Features for {zone}: {n_features}")

    maml_mgr: Optional[MAMLManager] = None
    if use_maml:
        cache_path = os.path.join(TRAINED_MODELS_DIR, f"maml_cache_{zone}")
        maml_mgr = MAMLManager(n_features=n_features, cache_path=cache_path)
        qh_windows, lr_coef, lr_intercept = _collect_backbone_data(
            loader, zone, pretrain_date=maml_backbone_date, lookback=lookback_days
        )
        maml_mgr.pretrain_backbone(qh_windows, lr_coef, lr_intercept, verbose=True)
        maml_mgr.load_qh_states()

    rows_buffer = []
    fail_counts: dict[str, int] = {}  # Track failures per model
    FLUSH_EVERY = CHECKPOINT_EVERY
    t_total = time.time()

    for day_idx, test_day in enumerate(tqdm(test_dates, desc=f"{zone} backtest")):
        test_date_str = str(test_day.date())

        for qh_idx in range(N_QH):
            done_models = _already_done(df_existing, zone, qh_idx, test_date_str)
            needed_models = set()
            if use_lr    and "lr"    not in done_models: needed_models.add("lr")
            if use_lasso and "lasso" not in done_models: needed_models.add("lasso")
            if use_csvr  and "csvr"  not in done_models: needed_models.add("csvr")
            if use_maml  and "maml"  not in done_models: needed_models.add("maml")
            if not needed_models:
                continue

            # Filtered window for LR/LASSO/cSVR (zero-variance columns removed)
            linear_models_needed = needed_models & {"lr", "lasso", "csvr"}
            w = None
            if linear_models_needed:
                w = loader.get_window(
                    zone, qh_idx, test_day,
                    lookback_days=lookback_days,
                    val_days=val_days,
                    save_scaler=save_scalers,
                    filter_zero_var=True,
                )

            # Unfiltered window for MAML (needs fixed dimension across all 96 QH)
            w_maml = None
            if "maml" in needed_models and maml_mgr is not None:
                w_maml = loader.get_window(
                    zone, qh_idx, test_day,
                    lookback_days=lookback_days,
                    val_days=val_days,
                    filter_zero_var=False,
                )

            # Use whichever window is available for row_base metadata
            w_ref = w or w_maml
            if w_ref is None:
                continue

            benchmark_eur = w_ref.benchmark_eur
            y_true_diff   = w_ref.y_test
            y_true_eur    = _backtransform(y_true_diff, benchmark_eur)

            row_base = {
                "zone": zone,
                "qh_idx": qh_idx,
                "test_date": test_date_str,
                "benchmark_eur": benchmark_eur,
                "y_true_diff": y_true_diff,
                "y_true_eur": y_true_eur,
                "n_train": len(w_ref.X_train),
                "n_val": len(w_ref.X_val),
            }

            # -- LR --
            if "lr" in needed_models and w is not None:
                try:
                    m_lr = train_lr(w.X_trainval, w.y_trainval)
                    pred_diff = float(predict_lr(m_lr, w.X_test)[0])
                    pred_eur  = _backtransform(pred_diff, benchmark_eur)
                    rows_buffer.append({
                        **row_base, "model": "lr",
                        "pred_diff": pred_diff, "pred_eur": pred_eur,
                        "mae_eur": _mae(y_true_eur, pred_eur),
                    })
                    if save_models:
                        save_trained_model(m_lr, "lr", zone, qh_idx, test_date_str)
                except Exception as e:
                    fail_counts["lr"] = fail_counts.get("lr", 0) + 1
                    if verbose: log.warning(f"LR failed {zone} QH{qh_idx} {test_date_str}: {e}")

            # -- LASSO --
            if "lasso" in needed_models and w is not None:
                try:
                    m_lasso, best_alpha = train_lasso(
                        w.X_train, w.y_train,
                        w.X_val,   w.y_val,
                        w.X_trainval, w.y_trainval,
                    )
                    pred_diff = float(predict_lasso(m_lasso, w.X_test)[0])
                    pred_eur  = _backtransform(pred_diff, benchmark_eur)
                    rows_buffer.append({
                        **row_base, "model": "lasso",
                        "pred_diff": pred_diff, "pred_eur": pred_eur,
                        "mae_eur": _mae(y_true_eur, pred_eur),
                        "lasso_alpha": best_alpha,
                    })
                    if save_models:
                        save_trained_model(m_lasso, "lasso", zone, qh_idx, test_date_str)
                except Exception as e:
                    fail_counts["lasso"] = fail_counts.get("lasso", 0) + 1
                    if verbose: log.warning(f"LASSO failed {zone} QH{qh_idx} {test_date_str}: {e}")

            # -- cSVR --
            if "csvr" in needed_models and w is not None:
                try:
                    m_csvr = train_csvr(w.X_trainval, w.y_trainval)
                    pred_diff = float(predict_csvr(m_csvr, w.X_test)[0])
                    pred_eur  = _backtransform(pred_diff, benchmark_eur)
                    rows_buffer.append({
                        **row_base, "model": "csvr",
                        "pred_diff": pred_diff, "pred_eur": pred_eur,
                        "mae_eur": _mae(y_true_eur, pred_eur),
                    })
                    if save_models:
                        save_trained_model(m_csvr, "csvr", zone, qh_idx, test_date_str)
                except Exception as e:
                    fail_counts["csvr"] = fail_counts.get("csvr", 0) + 1
                    if verbose: log.warning(f"cSVR failed {zone} QH{qh_idx} {test_date_str}: {e}")

            # -- MAML-NN (uses unfiltered window for fixed dimension) --
            if "maml" in needed_models and maml_mgr is not None and w_maml is not None:
                try:
                    pred_diff = maml_mgr.adapt_and_predict(
                        qh_idx,
                        w_maml.X_train, w_maml.y_train,
                        w_maml.X_val,   w_maml.y_val,
                        w_maml.X_test,
                    )
                    pred_eur = _backtransform(pred_diff, benchmark_eur)
                    rows_buffer.append({
                        **row_base, "model": "maml",
                        "pred_diff": pred_diff, "pred_eur": pred_eur,
                        "mae_eur": _mae(y_true_eur, pred_eur),
                    })
                except Exception as e:
                    fail_counts["maml"] = fail_counts.get("maml", 0) + 1
                    if verbose: log.warning(f"MAML failed {zone} QH{qh_idx} {test_date_str}: {e}")

        # Flush periodically
        if (day_idx + 1) % FLUSH_EVERY == 0 and rows_buffer:
            df_new = pd.DataFrame(rows_buffer)
            write_header = not os.path.exists(results_path)
            df_new.to_csv(results_path, mode="a", header=write_header, index=False)
            rows_buffer.clear()
            gc.collect()

        if use_maml and maml_mgr is not None and (day_idx + 1) % 30 == 0:
            maml_mgr.save_qh_states()

    # Final flush
    if rows_buffer:
        df_new = pd.DataFrame(rows_buffer)
        write_header = not os.path.exists(results_path)
        df_new.to_csv(results_path, mode="a", header=write_header, index=False)

    if use_maml and maml_mgr is not None:
        maml_mgr.save_qh_states()

    loader.close()
    elapsed = (time.time() - t_total) / 60
    if fail_counts:
        log.warning(f"Model failures during backtest: {fail_counts}")
    log.info(f"Backtest done: {zone} — {elapsed:.1f} min → {results_path}")
    return results_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-QH EPF Backtest")
    parser.add_argument("--zone",         default="DE1", choices=ZONES)
    parser.add_argument("--test_start",   default=TEST_START)
    parser.add_argument("--test_end",     default=TEST_END)
    parser.add_argument("--lookback",     default=LOOKBACK_DAYS, type=int)
    parser.add_argument("--val_days",     default=VAL_DAYS,      type=int)
    parser.add_argument("--no_lr",        action="store_true")
    parser.add_argument("--no_lasso",     action="store_true")
    parser.add_argument("--no_csvr",      action="store_true")
    parser.add_argument("--no_maml",      action="store_true")
    parser.add_argument("--save_scalers", action="store_true", help="Save fitted scalers to Models/scalers")
    parser.add_argument("--save_models",  action="store_true", help="Save models to Models/trained_models")
    parser.add_argument("--tag",          default=None, help="Results file suffix")
    parser.add_argument("--verbose",      action="store_true")
    args = parser.parse_args()

    run_backtest(
        zone=args.zone,
        test_start=args.test_start,
        test_end=args.test_end,
        lookback_days=args.lookback,
        val_days=args.val_days,
        use_lr=not args.no_lr,
        use_lasso=not args.no_lasso,
        use_csvr=not args.no_csvr,
        use_maml=not args.no_maml,
        save_scalers=args.save_scalers,
        save_models=args.save_models,
        run_tag=args.tag,
        verbose=args.verbose,
    )
