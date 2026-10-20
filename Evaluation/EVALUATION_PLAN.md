# Evaluation Plan (Results Section)

Guiding principle: lean paper. Three tables and three to four figures in the main text; everything else in an appendix or one sentence.

## Main text

| ID | Content | Status |
| :--- | :--- | :--- |
| T1 | MAE, rMAE, RMSE per control area: Naive, LASSO, LASSO + Naive (Marcjasz et al., 2020), cSVR (main, no filter), cSVR (filter, replication of Puć & Janczura), FA-NN | table script to write |
| F1 | One-sided daily-vector DM chessboard (HAC, h = 7), 4 panels (DE1–DE4) | draft: `plots/drafts/draft_F1_dm_chessboard.png` |
| T2 | Runtime: calibration time per zone (LASSO, cSVR, FA-NN Stage 1 once + Stage 2 per forecast) vs. NN with daily full retraining | measurement missing |
| T3 | Ablation: FA-NN vs. FOMAML (warm start), FOMAML (scratch), backbone without adaptation (G = 0) | data available after rerun; G = 0 from `Results/sensitivity_inner_steps_DE2.npz` |
| F0 | Motivation: zonal decoupling, abs(zonal − national VWAP) for 90–105 min vs. 0–30 min | draft: `draft_F0_zonal_decoupling.png` (candidate for Data section) |
| F2 | rMAE by delivery hour, DE2 (where do the gains occur?) | draft: `draft_F2_rmae_by_hour_DE2.png` |
| F3 | Cumulative daily MAE advantage of FA-NN vs. Naive and cSVR (stability over the year) | draft: `draft_F3_cumulative_advantage.png` |

## Appendix / one sentence

- "+ Naive" ensemble variants of all architectures.
- Feature-set contributions: S4 helps LASSO, not the nonlinear models; inverse-MAE weights ≈ 0.25 (nearly equal weighting).
- Inner-step sensitivity (DE2), explicitly labelled as post-hoc analysis.
- Support-set age distribution (median ≈ 145 days): adaptation selects similar regimes, not recent days.
- GW test (conditional predictive ability) for FA-NN vs. cSVR.

## Dropped (keep the paper lean)

- SAGE / feature importance.
- Section "Proposed Model for parallel Zone Forecasting" → outlook (multi-zone model).
- OptLASSO bypass → at most one paragraph as extension, otherwise omit.

## Additional tests / to-dos

1. Seed robustness: FA-NN with 5 seeds, mean ± std per zone (gain vs. cSVR is only a few tenths of EUR/MWh).
2. Runtime: log wall-clock per model and zone in `run_experiment.py` (already partly printed) plus a reference run of an NN with daily retraining on a subset (e.g. DE2, one month), extrapolated; report hardware.
3. Table script `Evaluation/make_tables.py` (T1, T3) writing LaTeX tables to `Contribution/tables/`.
4. Results section: rewrite in English; replace German placeholder text.
5. Draft figure fixes before inclusion: colour legend for F1, local time (CET) on F2, move "Naive" label in F2.
