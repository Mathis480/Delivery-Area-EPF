"""
Result tables for the paper: python Evaluation/make_tables.py [results_dir]  ->  Contribution/tables/*.tex
"""
import os
import re
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import RUN_DIR, ZONES
from Models.ensemble import daily_loss, dm_test, gw_test, inverse_mae_average

RESULTS_DIR = sys.argv[1] if len(sys.argv) > 1 else RUN_DIR
D = {z: np.load(os.path.join(RESULTS_DIR, f"results_{z}.npz")) for z in ZONES}
STAGE1 = {"LASSO": "lasso", "LASSO+Naive": "lasso_n", "cSVR": "csvr_corr", "FOMAML": "fomaml", "FA-NN": "fann"}
WITH_NAIVE = {"LASSO (N)": "lasso+n", "cSVR (N)": "csvr_corr+n", "FOMAML (N)": "fomaml+n",
              "FA-NN (N)": "fann+n"}
_with_naive = {}
STAGE2 = {"FOMAML (after pre-training)": ("fomaml_pre", "fann"), "cSVR without filter": ("csvr", "csvr_corr"),
          "FA-NN with LASSO bypass": ("fann_bypass", "fann")}


def pred(z, m):
    if m == "naive":
        return D[z]["naive"]
    if not m.endswith("+n"):
        return D[z][f"pred_{m}"]
    if (z, m) not in _with_naive:  # inverse-MAE average over the four feature sets and the naive forecast
        members = {f"s{i}": D[z][f"pred_{m[:-2]}_s{i}"] for i in range(1, 5)}
        _with_naive[z, m] = inverse_mae_average({**members, "naive": D[z]["naive"]}, D[z]["y_true"])[0]
    return _with_naive[z, m]


def err(z, m, p=1):
    """MAE (p = 1) or RMSE (p = 2)."""
    return np.nanmean(np.abs(pred(z, m) - D[z]["y_true"]) ** p) ** (1 / p)


def p_better(z, a, b, test=dm_test):
    """One-sided p-value for H1: model a is more accurate than model b (daily MAE)."""
    return test(daily_loss(pred(z, b), D[z]["y_true"]), daily_loss(pred(z, a), D[z]["y_true"]))[1]


def stars(p):
    return "$^{***}$" if p < 0.01 else "$^{**}$" if p < 0.05 else "$^{*}$" if p < 0.10 else ""


def save(table, name, caption, label):
    df = pd.DataFrame(table).T
    tex = df.to_latex(escape=False, caption=caption, label=label, position="htbp",
                      column_format="l" * df.index.nlevels + "c" * df.shape[1])
    os.makedirs(os.path.join(ROOT, "Contribution", "tables"), exist_ok=True)
    with open(os.path.join(ROOT, "Contribution", "tables", f"{name}.tex"), "w") as f:
        tex = re.sub(r"\\multicolumn\{(\d+)\}\{r\}", r"\\multicolumn{\1}{c}", tex)  # center group headers
        f.write(tex.replace("\\cline{1-6}\n\\bottomrule", "\\bottomrule").replace("\\cline{1-6}", "\\midrule")
                .replace("\\begin{tabular}", "\\centering\\small\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}"))


save({(metric, lab): {z: f"{err(z, m, p) / err(z, 'naive', p):.4f}" + (stars(p_better(z, m, "naive")) if p == 1 else "")
                      for z in ZONES}
      for metric, p in (("rMAE", 1), ("rRMSE", 2)) for lab, m in {**STAGE1, **WITH_NAIVE}.items()},
     "tab_main", "Forecast accuracy in 2024 relative to the naive benchmark (naive MAE: "
     + ", ".join(f"{z} {err(z, 'naive'):.2f}" for z in ZONES) + " EUR/MWh). Stars: one-sided DM test against the naive "
     r"benchmark, $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$. FOMAML without pre-training. (N): naive forecast as fifth member of the average.", "tab:MainResults")

pairs = [("FA-NN", "fann", "cSVR", "csvr_corr"), ("cSVR", "csvr_corr", "FA-NN", "fann"),
         ("FA-NN", "fann", "FOMAML", "fomaml"), ("FOMAML", "fomaml", "FA-NN", "fann"),
         ("FA-NN (N)", "fann+n", "cSVR (N)", "csvr_corr+n"), ("cSVR (N)", "csvr_corr+n", "FA-NN (N)", "fann+n")]
save({f"{la} $\\succ$ {lb}": {(z, name): f"{p_better(z, a, b, test):.3f}" for z in ZONES
                               for name, test in (("DM", dm_test), ("GW", gw_test))} for la, a, lb, b in pairs},
     "tab_dm_gw", r"One-sided $p$-values of the Diebold--Mariano (DM) \cite{Diebold.1995} and Giacomini--White (GW) "
     r"\cite{Giacomini.2006} tests on daily MAE in 2024, H1: the first model is more accurate than the second (FOMAML "
     "without pre-training). DM: Newey--West variance with seven lags. GW: conditional test with a constant and the "
     r"lagged loss differential as instruments, signed by the mean loss differential as in the epftoolbox \cite{Lago.2021}, "
     "so that $p = 1$ if the mean loss favors the second model. (N): naive forecast as fifth member of the average.",
     "tab:DMGW")

save({lab: {z: f"{err(z, m) / err(z, 'naive'):.4f}"
            + ("$^{+}$" if p_better(z, m, b) < 0.05 else "$^{-}$" if p_better(z, b, m) < 0.05 else "") for z in ZONES}
      for lab, (m, b) in STAGE2.items()},
     "tab_stage2", "Model variants and post-hoc modifications in 2024 (rMAE). $^{+}$/$^{-}$: significantly more/less accurate "
     "than the base model (FA-NN, cSVR with filter, FA-NN), one-sided DM test, $p<0.05$. The modifications were "
     "developed after the main evaluation. Their accuracy is an optimistic estimate.", "tab:Stage2")

# Runtime (Evaluation/runtime_benchmark.py): DE2, one thread, sample extrapolated to the full test year
rt = pd.read_csv(os.path.join(ROOT, "Results", "runtime_benchmark", "runtime_DE2.csv")).set_index("component")
RUNTIME = {"LASSO": (None, "lasso"), "cSVR": (None, "csvr_corr"), "FA-NN": ("fann_stage1", "fann_stage2"),
           "FOMAML": ("fomaml_stage1", "fomaml_stage2")}
rows = {"Data access (all models)": {"Training (min)": "--", "Per forecast (ms)": f"{rt.at['data', 'ms_per_forecast']:.1f}",
                                     "Test year (min)": f"{rt.at['data', 'min_per_zone_year']:.1f}"}}
for lab, (s1, s2) in RUNTIME.items():
    once = rt.at[s1, "min_per_zone_year"] if s1 else 0.0
    rows[lab] = {"Training (min)": f"{once:.2f}" if s1 else "--",
                 "Per forecast (ms)": f"{rt.at[s2, 'ms_per_forecast']:.1f}",
                 "Test year (min)": f"{once + rt.at[s2, 'min_per_zone_year']:.1f}"}
save(rows, "tab_runtime", "Computational cost per control area on one CPU thread (DE2, AMD Ryzen 5 7600X). Per-forecast "
     f"times are summed over the four feature sets, measured on {int(rt['n_sample'].iloc[0]):,} forecasts and extrapolated "
     f"to the {int(rt['n_year'].iloc[0]):,} quarter-hours of 2024. Training: backbone (FA-NN) or meta-training "
     "(FOMAML) for all four feature sets. Test year: training plus all forecasts, without data access. A dash marks steps "
     "that do not exist for a model.", "tab:Runtime")
