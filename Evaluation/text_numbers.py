"""
Numbers quoted in the results section of the paper that are not in the generated tables:

    python Evaluation/text_numbers.py
"""
import glob, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import pandas as pd
from config import DST_PRODUCTS, MASTER_PARQUET, N_QH, RESULTS_DIR, RUN_DIR, TIMEZONE, ZONES
from Models.ensemble import daily_loss, dm_test, gw_test, inverse_mae_average

D = {z: np.load(os.path.join(RUN_DIR, f"results_{z}.npz")) for z in ZONES}
mae = lambda z, p: np.nanmean(np.abs(p - D[z]["y_true"]))
rmse = lambda z, p: np.sqrt(np.nanmean((p - D[z]["y_true"]) ** 2))
sets = lambda z, m, ids=(1, 2, 3, 4): {f"s{i}": D[z][f"pred_{m}_s{i}"] for i in ids}
withn = lambda z, m: inverse_mae_average({**sets(z, m), "naive": D[z]["naive"]}, D[z]["y_true"])
pb = lambda z, a, b: dm_test(daily_loss(b, D[z]["y_true"]), daily_loss(a, D[z]["y_true"]))[1]
pb_gw = lambda z, a, b: gw_test(daily_loss(b, D[z]["y_true"]), daily_loss(a, D[z]["y_true"]))[1]


def timestamps(dates: np.ndarray, n_slots: int) -> pd.DatetimeIndex:
    """UTC delivery start of every (day, slot); NaT where the local product does not exist."""
    out = []
    for day in pd.to_datetime(dates.astype(str)):
        for q in range(n_slots):
            if q < N_QH:
                ts = (day + pd.Timedelta(minutes=15 * q)).tz_localize(TIMEZONE, ambiguous=True, nonexistent="NaT")
            else:  # second delivery of 02:00-02:45 on the autumn DST day
                ts = (day + pd.Timedelta(minutes=15 * (q - N_QH + DST_PRODUCTS[0]))).tz_localize(
                    TIMEZONE, ambiguous=False, nonexistent="NaT")
            out.append(ts.tz_convert("UTC") if ts is not pd.NaT else pd.NaT)
    return pd.DatetimeIndex(out)


print("Mean MAE over zones, without / with naive member; mean naive weight (median over days, all zones)")
for m in ("lasso", "lasso_n", "csvr_corr", "csvr", "fann", "fann_bypass", "fann_static", "fomaml_pre", "fomaml"):
    a = np.mean([mae(z, D[z][f"pred_{m}"]) for z in ZONES])
    if m in ("lasso", "csvr_corr", "fann", "fomaml"):
        r = {z: withn(z, m) for z in ZONES}
        w = np.concatenate([r[z][1][:, 4] for z in ZONES])
        print(f"{m:16s} {a:.3f} {np.mean([mae(z, r[z][0]) for z in ZONES]):.3f} w={np.median(w):.3f} [{np.quantile(w, .05):.3f}, {np.quantile(w, .95):.3f}]")
    else:
        print(f"{m:16s} {a:.3f}")

print("\nPairs (p first better / p second better)")
for z in ZONES:
    d = D[z]
    print(z, "fann v csvr DM", *np.round([pb(z, d["pred_fann"], d["pred_csvr_corr"]), pb(z, d["pred_csvr_corr"], d["pred_fann"])], 3),
          "GW", *np.round([pb_gw(z, d["pred_fann"], d["pred_csvr_corr"]), pb_gw(z, d["pred_csvr_corr"], d["pred_fann"])], 3),
          "| fann v fomaml GW", round(pb_gw(z, d["pred_fann"], d["pred_fomaml"]), 3))
    print(z, "bypass v csvr", *np.round([pb(z, d["pred_fann_bypass"], d["pred_csvr_corr"]), pb(z, d["pred_csvr_corr"], d["pred_fann_bypass"])], 3),
          "| fomaml_pre v fann", *np.round([pb(z, d["pred_fomaml_pre"], d["pred_fann"]), pb(z, d["pred_fann"], d["pred_fomaml_pre"])], 3),
          "| nofilter v filter", *np.round([pb(z, d["pred_csvr"], d["pred_csvr_corr"]), pb(z, d["pred_csvr_corr"], d["pred_csvr"])], 3),
          "| static v fann", *np.round([pb(z, d["pred_fann"], d["pred_fann_static"]), mae(z, d["pred_fann_static"]) - mae(z, d["pred_fann"])], 3))

print("\nrRMSE by zone:")
for m in ("lasso", "lasso_n", "csvr_corr", "fomaml", "fann"):
    print(f"{m:12s}", {z: round(rmse(z, D[z][f"pred_{m}"]) / rmse(z, D[z]["naive"]), 4) for z in ZONES})

print("\nS4: average of S1-S3 vs. all four (rMAE change, p S4 helps, p S4 hurts)")
for m in ("lasso", "csvr_corr", "fann", "fomaml"):
    for z in ZONES:
        without = inverse_mae_average(sets(z, m, (1, 2, 3)), D[z]["y_true"])[0]
        full = D[z][f"pred_{m}"]
        print(f"{m:15s} {z} {(mae(z, full) - mae(z, without)) / mae(z, D[z]['naive']):+.4f} {pb(z, full, without):.3f} {pb(z, without, full):.3f}")

print("\nMembers: rMAE per set (FA-NN, cSVR) and mean weight")
for m in ("fann", "csvr_corr"):
    print(m, {z: [round(mae(z, D[z][f"pred_{m}_s{i}"]) / mae(z, D[z]["naive"]), 4) for i in range(1, 5)] for z in ZONES})
    print(m, {z: np.round(D[z][f"weights_{m}"].mean(axis=0), 3).tolist() for z in ZONES})

master = pd.read_parquet(MASTER_PARQUET, columns=["Date"] + [f"{z}_VWAP_total_volume_0to30" for z in ZONES])
master["Date"] = pd.to_datetime(master["Date"], utc=True)
m = master.set_index("Date").tz_convert("Europe/Berlin").loc["2024-01-01":"2024-12-31"]
print("\nNo trades in target window:")
for z in ZONES:
    print(z, int((m[f"{z}_VWAP_total_volume_0to30"].fillna(0) == 0).sum()), len(m))

print("\nExcluding quarter-hours without trades in the target window: largest absolute rMAE change")
vol = master.set_index("Date")
models = ("lasso", "lasso_n", "csvr_corr", "csvr", "fann", "fann_bypass", "fomaml_pre", "fomaml")
worst = 0.0
for z in ZONES:
    y = D[z]["y_true"]
    ts = timestamps(D[z]["dates"], y.shape[1])
    v = pd.Series(np.nan, index=range(ts.size))
    ok = ~ts.isna()
    v[ok] = vol[f"{z}_VWAP_total_volume_0to30"].reindex(ts[ok]).fillna(0).to_numpy()
    traded = (v.to_numpy() > 0).reshape(y.shape)
    for m in models:
        a = np.abs(D[z][f"pred_{m}"] - y); n = np.abs(D[z]["naive"] - y)
        worst = max(worst, abs(np.nanmean(a) / np.nanmean(n) - np.nanmean(a[traded]) / np.nanmean(n[traded])))
print(round(worst, 5))

print("\nSeeds (base + 4): zone MAE, rMAE range, one-sided DM p vs. cSVR (FA-NN better / cSVR better), p vs. naive")
for z in ZONES:
    rows = [D[z]["pred_fann"]] + [np.load(os.path.join(RUN_DIR, f"seed_{s}", f"results_{z}.npz"))["pred_fann"] for s in range(1, 5)]
    y, c, n = D[z]["y_true"], D[z]["pred_csvr_corr"], D[z]["naive"]
    print(z, np.round([mae(z, r) / mae(z, n) for r in rows], 4), "better", np.round([pb(z, r, c) for r in rows], 3),
          "worse", np.round([pb(z, c, r) for r in rows], 3), "naive", np.round([pb(z, r, n) for r in rows], 4))

print("\n2023 validation, ln_pre vs. current (Evaluation/variant_pretest.py): blocks where ln_pre is better, mean MAE change")
for prefix in ("", "fomaml_"):
    diff = {}
    for z in ZONES:
        for start, end in (("01-01", "03-31"), ("04-01", "06-30"), ("07-01", "09-30"), ("10-01", "12-31")):
            r = {v: np.load(os.path.join(RESULTS_DIR, "variant_pretest", f"{prefix}{v}_{z}_2023-{start}_2023-{end}.npz"))
                 for v in ("current", "ln_pre")}
            mae_v = {v: np.nanmean(np.abs(r[v]["pred"] - r[v]["y_true"])) for v in r}
            diff[z, start] = mae_v["ln_pre"] - mae_v["current"]
    d = np.array(list(diff.values()))
    print(f"{prefix or 'fann_'}ln_pre better in {(d < 0).sum()} of {d.size}, mean {d.mean():+.3f};",
          {z: int(sum(diff[z, s] < 0 for zz, s in diff if zz == z)) for z in ZONES})

print("\nConventional per-forecast retraining vs. FA-NN (Evaluation/retrain_baseline.py, DE2, sampled days of 2023)")
rb = pd.read_csv(os.path.join(RESULTS_DIR, "retrain_baseline", "summary.csv"))
daily = rb.groupby("day")[["naive", "retrain", "static_once", "fann"]].mean()
print(f"{daily.shape[0]} days; mean MAE over sets: " + ", ".join(f"{c} {daily[c].mean():.2f}" for c in daily)
      + "; ratio to naive: " + ", ".join(f"{c} {daily[c].mean() / daily['naive'].mean():.4f}" for c in daily))
print(f"FA-NN better than retraining on {(daily['fann'] < daily['retrain']).sum()} of {daily.shape[0]} days, "
      f"retraining better than naive on {(daily['retrain'] < daily['naive']).sum()}; one-sided DM p (FA-NN better) "
      f"{dm_test(daily['retrain'].to_numpy(), daily['fann'].to_numpy(), h=1)[1]:.4f}")
sec = np.concatenate([np.load(f)["seconds"] for f in sorted(glob.glob(os.path.join(RESULTS_DIR, "retrain_baseline", "*.npz")))])
sec = sec[sec > 0]
print(f"Per fit under parallel load {sec.mean():.2f} s ({sec.mean() * 35136 * 4 / 3600:.1f} h per control area and test year)")
sec = np.concatenate([np.load(f)["seconds"] for f in glob.glob(os.path.join(RESULTS_DIR, "retrain_baseline", "timing", "*.npz"))])
sec = sec[sec > 0]
print(f"Unloaded single process (retrain_baseline.py --timing): {sec.mean():.2f} s per fit, extrapolated test year per "
      f"control area (35,136 QH x 4 sets, 1 thread): {sec.mean() * 35136 * 4 / 3600:.1f} h")
