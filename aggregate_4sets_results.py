"""
Aggregate final 4-sets results for DE1, DE2, DE3, DE4 into full_4sets_summary_metrics_2024.csv
"""
import os
import numpy as np
import pandas as pd
from config import RESULTS_DIR, ZONES
from Models.ensemble import compute_rolling_intelligent_ensemble

def aggregate_results():
    results_dir = os.path.join(RESULTS_DIR, "annual_run_2024")
    summaries = []

    for z in ZONES:
        npz_path = os.path.join(results_dir, f"results_{z}_full4sets_2024.npz")
        if not os.path.exists(npz_path):
            print(f"Waiting on {z} (not finished yet)...")
            continue
        
        data = np.load(npz_path)
        y_true = data["y_true"]
        bm = data["benchmark"]
        mae_naive = float(np.mean(np.abs(y_true - bm)))

        res = {
            "zone": z,
            "mae_naive": mae_naive,
            # Set 1 (Macro)
            "rmae_lasso_s1": float(np.mean(np.abs(y_true - data["pred_lasso_s1"])) / mae_naive),
            "rmae_csvr_s1": float(np.mean(np.abs(y_true - data["pred_csvr_s1"])) / mae_naive),
            "rmae_maml_s1": float(np.mean(np.abs(y_true - data["pred_maml_s1"])) / mae_naive),
            # Set 2 (Neighbor)
            "rmae_lasso_s2": float(np.mean(np.abs(y_true - data["pred_lasso_s2"])) / mae_naive),
            "rmae_csvr_s2": float(np.mean(np.abs(y_true - data["pred_csvr_s2"])) / mae_naive),
            "rmae_maml_s2": float(np.mean(np.abs(y_true - data["pred_maml_s2"])) / mae_naive),
            # Set 3 (Fundamental)
            "rmae_lasso_s3": float(np.mean(np.abs(y_true - data["pred_lasso_s3"])) / mae_naive),
            "rmae_csvr_s3": float(np.mean(np.abs(y_true - data["pred_csvr_s3"])) / mae_naive),
            "rmae_maml_s3": float(np.mean(np.abs(y_true - data["pred_maml_s3"])) / mae_naive),
            # Set 4 (Balance)
            "rmae_lasso_s4": float(np.mean(np.abs(y_true - data["pred_lasso_s4"])) / mae_naive),
            "rmae_csvr_s4": float(np.mean(np.abs(y_true - data["pred_csvr_s4"])) / mae_naive),
            "rmae_maml_s4": float(np.mean(np.abs(y_true - data["pred_maml_s4"])) / mae_naive),
            # Ensembles
            "rmae_csvr_ens_pure": float(np.mean(np.abs(y_true - data["pred_csvr_ens_pure"])) / mae_naive),
            "rmae_csvr_ens_wn": float(np.mean(np.abs(y_true - data["pred_csvr_ens_wn"])) / mae_naive),
            "rmae_csvr_ens_adapt": float(np.mean(np.abs(y_true - data["pred_csvr_ens_adapt"])) / mae_naive) if "pred_csvr_ens_adapt" in data else np.nan,
            "rmae_maml_ens_pure": float(np.mean(np.abs(y_true - data["pred_maml_ens_pure"])) / mae_naive),
            "rmae_maml_ens_wn": float(np.mean(np.abs(y_true - data["pred_maml_ens_wn"])) / mae_naive),
            "rmae_maml_ens_adapt": float(np.mean(np.abs(y_true - data["pred_maml_ens_adapt"])) / mae_naive) if "pred_maml_ens_adapt" in data else np.nan,
            "rmae_hybrid_adapt": float(np.mean(np.abs(y_true - data["pred_ens"])) / mae_naive),
        }
        summaries.append(res)

    if summaries:
        df = pd.DataFrame(summaries)
        out_csv = os.path.join(results_dir, "full_4sets_summary_metrics_2024.csv")
        df.to_csv(out_csv, index=False)
        print("\n=== AGGREGATED SUMMARY ===")
        print(df.to_string(index=False))
    else:
        print("No completed runs found yet.")

if __name__ == "__main__":
    aggregate_results()
