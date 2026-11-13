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
from Models.ensemble import daily_loss, dm_test, inverse_mae_average

RESULTS_DIR = sys.argv[1] if len(sys.argv) > 1 else RUN_DIR
D = {z: np.load(os.path.join(RESULTS_DIR, f"results_{z}.npz")) for z in ZONES}
STAGE1 = {"LASSO": "lasso", "LASSO+Naive": "lasso_n", "cSVR": "csvr_corr", "FOMAML": "fomaml", "FA-NN": "fann"}
WITH_NAIVE = {"LASSO (N)": "lasso+n", "cSVR (N)": "csvr_corr+n", "FOMAML (N)": "fomaml+n",
              "FA-NN (N)": "fann+n"}
_with_naive = {}


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


def save(table, name, caption, label, midrule_before=None, colsep=4):
    df = pd.DataFrame(table).T
    tex = df.to_latex(escape=False, caption=caption, label=label, position="htbp",
                      column_format="l" * df.index.nlevels + "c" * df.shape[1])
    if midrule_before is not None:
        tex = tex.replace(f"\n{midrule_before} &", f"\n\\midrule\n{midrule_before} &")
    os.makedirs(os.path.join(ROOT, "Contribution", "tables"), exist_ok=True)
    with open(os.path.join(ROOT, "Contribution", "tables", f"{name}.tex"), "w") as f:
        tex = re.sub(r"\\multicolumn\{(\d+)\}\{r\}", r"\\multicolumn{\1}{c}", tex)  # center group headers
        f.write(tex.replace("\\cline{1-6}\n\\bottomrule", "\\bottomrule").replace("\\cline{1-6}", "\\midrule")
                .replace("\\begin{tabular}", f"\\centering\\small\\setlength{{\\tabcolsep}}{{{colsep}pt}}\n\\begin{{tabular}}"))


save({lab: {z: f"{err(z, m) / err(z, 'naive'):.4f}{stars(p_better(z, m, 'naive'))}" for z in ZONES}
      for lab, m in {**STAGE1, **WITH_NAIVE}.items()},
     "tab_main", "Forecast accuracy (rMAE) in 2024. Naive MAE: "
     + ", ".join(f"{z} {err(z, 'naive'):.2f}" for z in ZONES) + r" EUR/MWh. Stars: one-sided DM test vs.\ naive "
     r"($^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$). (N): naive included in average. FOMAML is trained without supervised pre-training.", "tab:MainResults",
     midrule_before="LASSO (N)")

# Individual feature sets (members of the inverse-MAE average)
SETS = {"LASSO": "lasso", "cSVR": "csvr_corr", "FOMAML": "fomaml", "FA-NN": "fann"}
save({(lab, f"S{i}"): {z: f"{err(z, f'{m}_s{i}') / err(z, 'naive'):.4f}{stars(p_better(z, f'{m}_s{i}', 'naive'))}"
                       for z in ZONES} for lab, m in SETS.items() for i in range(1, 5)},
     "tab_sets", r"Forecast accuracy (rMAE) of the individual feature sets in 2024, before forecast averaging. Stars: "
     r"one-sided DM test vs.\ naive ($^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$). FOMAML is trained without supervised pre-training.",
     "tab:SetResults")

# Runtime (Evaluation/runtime_benchmark.py): DE2, one thread, sample extrapolated to the full test year
rt = pd.read_csv(os.path.join(ROOT, "Results", "runtime_benchmark", "runtime_DE2.csv")).set_index("component")
RUNTIME = {"LASSO (LARS)": (None, "lasso"), "cSVR": (None, "csvr_corr"), "FA-NN": ("fann_stage1", "fann_stage2"),
           "FOMAML": ("fomaml_stage1", "fomaml_stage2")}
rows = {"Data access (all models)": {"Per forecast (ms)": f"{rt.at['data', 'ms_per_forecast']:.1f}",
                                     "Test year (min, sum)": f"{rt.at['data', 'min_per_zone_year']:.1f}"}}
for lab, (s1, s2) in RUNTIME.items():
    once = rt.at[s1, "min_per_zone_year"] if s1 else 0.0
    rows[lab] = {"Per forecast (ms)": f"{rt.at[s2, 'ms_per_forecast']:.1f}",
                 "Test year (min, sum)": f"{once + rt.at[s2, 'min_per_zone_year']:.1f}"}
save(rows, "tab_runtime", "Single-thread runtime per control area (DE2, summed over S1--S4, extrapolated from "
     f"{int(rt['n_sample'].iloc[0]):,} to {int(rt['n_year'].iloc[0]):,} forecasts). Model rows exclude data access. The test-year sums of FA-NN and FOMAML include the one-time Stage-1 training.",
     "tab:Runtime", colsep=6)

