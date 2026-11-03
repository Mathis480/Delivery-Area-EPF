"""
Result tables for the paper: python Evaluation/make_tables.py [results_dir]  ->  Contribution/tables/*.tex
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import RUN_DIR, ZONES
from Models.ensemble import daily_loss, dm_test, gw_test, inverse_mae_average

RESULTS_DIR = sys.argv[1] if len(sys.argv) > 1 else RUN_DIR
D = {z: np.load(os.path.join(RESULTS_DIR, f"results_{z}.npz")) for z in ZONES}
STAGE1 = {"LASSO": "lasso", "LASSO+Naive": "lasso_n", "cSVR": "csvr_corr", "FOMAML": "fomaml_scratch", "FA-NN": "fann"}
WITH_NAIVE = {"LASSO (N)": "lasso+n", "cSVR (N)": "csvr_corr+n", "FOMAML (N)": "fomaml_scratch+n",
              "FA-NN (N)": "fann+n"}
_with_naive = {}
STAGE2 = {"FOMAML (after pre-training)": ("fomaml", "fann"), "cSVR without filter": ("csvr", "csvr_corr"),
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
    with open(os.path.join(ROOT, "Contribution", "tables", f"{name}.tex"), "w") as f:
        f.write(tex.replace("\\cline{1-6}\n\\bottomrule", "\\bottomrule").replace("\\cline{1-6}", "\\midrule")
                .replace("\\begin{tabular}", "\\centering\\small\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}"))


save({(metric, lab): {z: f"{err(z, m, p) / err(z, 'naive', p):.4f}" + (stars(p_better(z, m, "naive")) if p == 1 else "")
                      for z in ZONES}
      for metric, p in (("rMAE", 1), ("rRMSE", 2)) for lab, m in {**STAGE1, **WITH_NAIVE}.items()},
     "tab_main", "Forecast accuracy in 2024 relative to the naive benchmark (naive MAE: "
     + ", ".join(f"{z} {err(z, 'naive'):.2f}" for z in ZONES) + " EUR/MWh). Stars: one-sided DM test against the naive "
     r"benchmark, $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$. FOMAML without pre-training. (N): naive forecast as fifth member of the average.", "tab:MainResults")

pairs = [("FA-NN", "fann", "cSVR", "csvr_corr"), ("cSVR", "csvr_corr", "FA-NN", "fann"),
         ("FA-NN", "fann", "FOMAML", "fomaml_scratch"), ("FOMAML", "fomaml_scratch", "FA-NN", "fann"),
         ("FA-NN (N)", "fann+n", "cSVR (N)", "csvr_corr+n"), ("cSVR (N)", "csvr_corr+n", "FA-NN (N)", "fann+n")]
save({f"{la} $\\succ$ {lb}": {(z, name): f"{p_better(z, a, b, test):.3f}" for z in ZONES
                               for name, test in (("DM", dm_test), ("GW", gw_test))} for la, a, lb, b in pairs},
     "tab_dm_gw", "One-sided $p$-values of the Diebold--Mariano (DM) and Giacomini--White (GW) tests on daily MAE in "
     "2024, H1: the first model is more accurate than the second (FOMAML without pre-training). (N): naive forecast as "
     "fifth member of the average.", "tab:DMGW")

save({lab: {z: f"{err(z, m) / err(z, 'naive'):.4f}"
            + ("$^{+}$" if p_better(z, m, b) < 0.05 else "$^{-}$" if p_better(z, b, m) < 0.05 else "") for z in ZONES}
      for lab, (m, b) in STAGE2.items()},
     "tab_stage2", "Model variants and post-hoc modifications in 2024 (rMAE). $^{+}$/$^{-}$: significantly more/less accurate "
     "than the base model (FA-NN, cSVR with filter, FA-NN), one-sided DM test, $p<0.05$. The modifications were "
     "developed after the main evaluation; their accuracy is an optimistic estimate.", "tab:Stage2")
