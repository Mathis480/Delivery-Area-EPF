"""
Result tables for the paper (LaTeX to Contribution/tables/, CSV to <results_dir>/tables/).

Usage: python Evaluation/make_tables.py [results_dir]   (default: RUN_DIR from config)

* tab_main:      rMAE and rRMSE per control area, stage 1 (models as published or proposed); DM vs. naive.
* tab_dm_gw:     DM and GW p-values of the FA-NN against the cSVR and FOMAML without pre-training.
* tab_stage2:    ablation and post-hoc modifications relative to their base model.
* tab_s4:        post-hoc contribution of S4: averaging with vs. without S4.
* posthoc.csv:   LASSO-naive combinations and robustness without quarter-hours lacking target trades.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import MASTER_PARQUET, RUN_DIR, ZONES
from Models.ensemble import daily_loss, dm_test, gw_test, inverse_mae_average

RESULTS_DIR = sys.argv[1] if len(sys.argv) > 1 else RUN_DIR
TEX_DIR = os.path.join(ROOT, "Contribution", "tables")
CSV_DIR = os.path.join(RESULTS_DIR, "tables")
os.makedirs(TEX_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

STAGE1 = {"LASSO": "lasso", "LASSO+N": "lasso_n", "cSVR": "csvr_corr",
          "FOMAML (no pre-training)": "fomaml_scratch", "FA-NN": "fann"}
STAGE2 = {  # label: (model, base model)
    "FOMAML (after pre-training)": ("fomaml", "fann"),
    "cSVR without correlation filter": ("csvr", "csvr_corr"),
    "FA-NN with LASSO bypass": ("fann_bypass", "fann"),
    "FOMAML (after pre-training) with bypass": ("fomaml_bypass", "fomaml"),
    "FOMAML (no pre-training) with bypass": ("fomaml_scratch_bypass", "fomaml_scratch"),
}
SETS = ["s1", "s2", "s3"]


def load() -> dict:
    return {z: np.load(os.path.join(RESULTS_DIR, f"results_{z}.npz")) for z in ZONES}


def mae(p, y): return float(np.nanmean(np.abs(p - y)))
def rmse(p, y): return float(np.sqrt(np.nanmean((p - y) ** 2)))
def pred(d, m): return d["naive"] if m == "naive" else d[f"pred_{m}"]


def p_better(d, a: str, b: str, test=dm_test) -> float:
    """One-sided p-value for H1: model a is more accurate than model b (daily MAE)."""
    y = d["y_true"]
    return test(daily_loss(pred(d, b), y), daily_loss(pred(d, a), y))[1]


def stars(p: float) -> str:
    return "^{***}" if p < 0.01 else "^{**}" if p < 0.05 else "^{*}" if p < 0.10 else ""


def write(name: str, tex: str, df: pd.DataFrame) -> None:
    with open(os.path.join(TEX_DIR, f"{name}.tex"), "w") as f:
        f.write(tex)
    df.to_csv(os.path.join(CSV_DIR, f"{name}.csv"))


def tab_main(D: dict) -> None:
    rows, lines = [], {"rMAE": [], "rRMSE": []}
    for metric, fn in (("rMAE", mae), ("rRMSE", rmse)):
        rel = {lab: {z: fn(pred(D[z], m), D[z]["y_true"]) / fn(D[z]["naive"], D[z]["y_true"]) for z in ZONES}
               for lab, m in STAGE1.items()}
        best = {z: min(rel[lab][z] for lab in rel) for z in ZONES}
        naive_abs = [fn(D[z]["naive"], D[z]["y_true"]) for z in ZONES]
        lines[metric].append(f"    Naive (EUR/MWh) & " + " & ".join(f"{v:.2f}" for v in naive_abs) + r" \\")
        for lab, m in STAGE1.items():
            cells = []
            for z in ZONES:
                v = rel[lab][z]
                s = stars(p_better(D[z], m, "naive")) if metric == "rMAE" else ""
                txt = f"{v:.4f}{'$' + s + '$' if s else ''}"
                cells.append(rf"\textbf{{{txt}}}" if np.isclose(v, best[z]) else txt)
                rows.append({"metric": metric, "model": lab, "zone": z, "value": v})
            lines[metric].append(f"    {lab} & " + " & ".join(cells) + r" \\")
    tex = r"""\begin{table}[htbp]
  \centering
  \caption{Forecast accuracy in 2024 relative to the naive benchmark (stage 1: models as published or proposed). The first row of each panel gives the absolute error of the naive benchmark in EUR/MWh; best value per control area in bold.}
  \label{tab:MainResults}
  \footnotesize
  \begin{tabular}{@{}lcccc@{}}
    \toprule
     & DE1 & DE2 & DE3 & DE4 \\
    \midrule
    \multicolumn{5}{@{}l}{\textit{Panel A: rMAE}} \\
""" + "\n".join(lines["rMAE"]) + r"""
    \midrule
    \multicolumn{5}{@{}l}{\textit{Panel B: rRMSE}} \\
""" + "\n".join(lines["rRMSE"]) + r"""
    \bottomrule
  \end{tabular}

  \smallskip
  \parbox{\linewidth}{\scriptsize Stars: one-sided Diebold--Mariano test on daily MAE against the naive benchmark (H1: model more accurate), $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.}
\end{table}
"""
    write("tab_main", tex, pd.DataFrame(rows).pivot_table(index=["metric", "model"], columns="zone", values="value"))


def tab_dm_gw(D: dict) -> None:
    pairs = [("FA-NN", "fann", "cSVR", "csvr_corr"), ("FA-NN", "fann", "FOMAML (no pre-training)", "fomaml_scratch")]
    rows, lines = [], []
    for la, a, lb, b in pairs:
        for (x, lx), (y_, ly) in (((a, la), (b, lb)), ((b, lb), (a, la))):
            cells = []
            for z in ZONES:
                dm, gw = p_better(D[z], x, y_), p_better(D[z], x, y_, gw_test)
                cells += [f"{dm:.3f}", f"{gw:.3f}"]
                rows.append({"H1": f"{lx} better than {ly}", "zone": z, "DM": dm, "GW": gw})
            lines.append(f"    {lx} $\\succ$ {ly} & " + " & ".join(cells) + r" \\")
        lines.append(r"    \midrule")
    tex = r"""\begin{table}[htbp]
  \centering
  \caption{Pairwise tests in 2024: one-sided $p$-values of the Diebold--Mariano (DM) and Giacomini--White (GW) tests on daily MAE for H1: the first model is more accurate than the second.}
  \label{tab:DMGW}
  \footnotesize
  \setlength{\tabcolsep}{4pt}
  \begin{tabular}{@{}>{\raggedright\arraybackslash}p{3.3cm}cccccccc@{}}
    \toprule
     & \multicolumn{2}{c}{DE1} & \multicolumn{2}{c}{DE2} & \multicolumn{2}{c}{DE3} & \multicolumn{2}{c}{DE4} \\
    \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(l){8-9}
    H1 & DM & GW & DM & GW & DM & GW & DM & GW \\
    \midrule
""" + "\n".join(lines[:-1]) + r"""
    \bottomrule
  \end{tabular}
\end{table}
"""
    write("tab_dm_gw", tex, pd.DataFrame(rows).pivot_table(index="H1", columns="zone", values=["DM", "GW"]))


def tab_stage2(D: dict) -> None:
    rows, lines = [], []
    for lab, (m, base) in STAGE2.items():
        cells = []
        for z in ZONES:
            y = D[z]["y_true"]
            r, rb = mae(pred(D[z], m), y) / mae(D[z]["naive"], y), mae(pred(D[z], base), y) / mae(D[z]["naive"], y)
            pb, pw = p_better(D[z], m, base), p_better(D[z], base, m)
            mark = r"$^{+}$" if pb < 0.05 else r"$^{-}$" if pw < 0.05 else ""
            cells.append(f"{r:.4f}{mark}")
            rows.append({"model": lab, "base": base, "zone": z, "rMAE": r, "rMAE_base": rb, "p_better": pb, "p_worse": pw})
        lines.append(f"    {lab} & " + " & ".join(cells) + r" \\")
    tex = r"""\begin{table}[htbp]
  \centering
  \caption{Ablation and post-hoc modifications in 2024 (rMAE). The modifications were developed after the main evaluation; their accuracy is an optimistic estimate.}
  \label{tab:Stage2}
  \footnotesize
  \begin{tabular}{@{}>{\raggedright\arraybackslash}p{5.2cm}cccc@{}}
    \toprule
     & DE1 & DE2 & DE3 & DE4 \\
    \midrule
""" + "\n".join(lines) + r"""
    \bottomrule
  \end{tabular}

  \smallskip
  \parbox{\linewidth}{\scriptsize $^{+}$/$^{-}$: significantly more/less accurate than the respective base model (FOMAML after pre-training: FA-NN; cSVR without filter: cSVR; bypass variants: same model without bypass), one-sided DM test on daily MAE, $p<0.05$.}
\end{table}
"""
    write("tab_stage2", tex, pd.DataFrame(rows).set_index(["model", "zone"]))


def tab_s4(D: dict) -> None:
    rows, lines = [], []
    for lab, m in {"LASSO": "lasso", "cSVR": "csvr_corr", "FOMAML (no pre-training)": "fomaml_scratch", "FA-NN": "fann"}.items():
        cells = []
        for z in ZONES:
            d, y = D[z], D[z]["y_true"]
            p4 = d[f"pred_{m}"]
            p3, _ = inverse_mae_average({s: d[f"pred_{m}_{s}"] for s in SETS}, y)
            delta = mae(p3, y) - mae(p4, y)
            l3, l4 = daily_loss(p3, y), daily_loss(p4, y)
            dm_help = dm_test(l3, l4)[1]
            cells.append(f"{delta:+.3f}{'$' + stars(dm_help) + '$' if stars(dm_help) else ''}")
            rows.append({"model": lab, "zone": z, "delta_mae": delta, "p_dm_s4_helps": dm_help})
        lines.append(f"    {lab} & " + " & ".join(cells) + r" \\")
    tex = r"""\begin{table}[htbp]
  \centering
  \caption{Post-hoc contribution of S4 in 2024: MAE reduction (EUR/MWh) from adding S4 to the averaged S1--S3 forecast; negative values indicate that S4 worsens the average.}
  \label{tab:S4}
  \footnotesize
  \begin{tabular}{@{}>{\raggedright\arraybackslash}p{4.2cm}cccc@{}}
    \toprule
     & DE1 & DE2 & DE3 & DE4 \\
    \midrule
""" + "\n".join(lines) + r"""
    \bottomrule
  \end{tabular}

  \smallskip
  \parbox{\linewidth}{\scriptsize Stars: one-sided DM test, H1: the average including S4 is more accurate, $^{***}p<0.01$, $^{**}p<0.05$, $^{*}p<0.10$.}
\end{table}
"""
    write("tab_s4", tex, pd.DataFrame(rows).set_index(["model", "zone"]))


def posthoc(D: dict) -> None:
    rows = []
    for z in ZONES:
        d, y = D[z], D[z]["y_true"]
        for w in (0.5, 0.75, 0.9):
            p = w * d["naive"] + (1 - w) * d["pred_lasso"]
            rows.append({"analysis": f"LASSO with naive weight {w}", "zone": z,
                         "rMAE": mae(p, y) / mae(d["naive"], y),
                         "p_vs_naive": dm_test(daily_loss(d["naive"], y), daily_loss(p, y))[1]})
    master = pd.read_parquet(MASTER_PARQUET, columns=["Date"] + [f"{z}_VWAP_total_trades_0to30" for z in ZONES])
    master["day"] = pd.to_datetime(master["Date"], utc=True).dt.strftime("%Y-%m-%d")
    master["qh"] = pd.to_datetime(master["Date"], utc=True).dt.hour * 4 + pd.to_datetime(master["Date"], utc=True).dt.minute // 15
    for z in ZONES:
        d = D[z]; days = list(d["dates"].astype(str))
        sel = master[master["day"].isin(days) & (master[f"{z}_VWAP_total_trades_0to30"] == 0)]
        y = d["y_true"].copy(); y[[days.index(x) for x in sel["day"]], sel["qh"].to_numpy()] = np.nan
        for m in ("csvr_corr", "fann"):
            rows.append({"analysis": f"{m} without {len(sel)} QH lacking target trades", "zone": z,
                         "rMAE": mae(d[f"pred_{m}"], y) / mae(d["naive"], y), "p_vs_naive": np.nan})
    pd.DataFrame(rows).to_csv(os.path.join(CSV_DIR, "posthoc.csv"), index=False)


if __name__ == "__main__":
    data = load()
    for fn in (tab_main, tab_dm_gw, tab_stage2, tab_s4, posthoc):
        fn(data)
    print(f"Tables written to {TEX_DIR} and {CSV_DIR}")
