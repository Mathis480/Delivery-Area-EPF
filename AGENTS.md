# Agent Instructions & Workspace Rules

> Keep text minimal, stick to the point, and avoid lengthy explanations.

## 1. Language & Literature
* Chat: German or English, matching the user.
* Code & Docs: Strictly English.
* Literature ground truth: Consult [`old/LITERATURE_KNOWLEDGE_BASE.md`](old/LITERATURE_KNOWLEDGE_BASE.md) before citing formulas or papers.
  * Marcjasz et al. (2018): Inverse-MAE window weighting ($w \propto 1/\text{MAE}$, Eq. 5).
  * Marcjasz et al. (2020): 50:50 arithmetic average with naive baseline ($\text{ens} = 0.5\hat{y} + 0.5y_{\text{naive}}$, Eq. 15).
  * Puć & Janczura (2024): Linear inverse-MAE rolling weighting ($p=1.0$, Eq. 25).

## 2. Formula Rendering (Chat Readability)
* No raw LaTeX math tags (`$...$` or `$$...$$`) in chat responses (IDE markdown renderer does not parse them).
* Render non-trivial equations as PNG via matplotlib (`dpi=250`, `facecolor='white'`, `pad_inches=0.15`, `bbox_inches='tight'`) in scratch/artifacts and embed via markdown image syntax.
* Short expressions can use standard fenced ````math ```` blocks.

## 3. Core Methodological Rules
* Grid: Continuous 15-minute intervals in pure UTC (`+00:00`). Never drop rows.
* Regional targets: `DE1_VWAP_0to30`, `DE2_VWAP_0to30`, `DE3_VWAP_0to30`, `DE4_VWAP_0to30`.
* Primary reference area: Prioritize DE2 (Amprion) for detailed model comparisons.
* Naive benchmark: `{zone}_VWAP_90to105` (last observed window at the 90-minute pre-delivery cutoff).
* National `VWAP_0to30`: Internal imputation fallback only. Never use as target, feature, or benchmark.
* Publication delay: Nominal 60 min gate closure + 30 min reporting buffer = effective cutoff 90 min before delivery.
* Neighbor contracts: Window ending $A$ minutes before delivery is safe if $A \ge 90 + 15 \cdot k$ for $k \in [-4, +2]$.
* Ex-post features: Realized fundamentals (generation, load, balancing) must be lagged by at least 8 quarter-hours (2 hours).
* No backward fill: `bfill` is prohibited.
* Feature catalog (`features.csv`):
  * `yes` (338): Direct features for target contract $k=0$.
  * `only past products` (52): Safe neighbor windows for $k \in [-4, -1]$. Never use directly for $k=0$.
  * `no` (520): Inactive variables.

## 4. TSO Delivery Area Mapping
Map German TSOs 1:1 to official EPEX delivery areas:
* `DE1_`: TransnetBW (`DE-ENBW`)
* `DE2_`: Amprion (`DE-AMP`)
* `DE3_`: TenneT (`DE-TPS`)
* `DE4_`: 50Hertz (`DE-50HZ`)
* `DE_`: National aggregate

Never use raw company names in code, features, or model configs.

## 5. Missing Data Imputation
* Regional prices: Own window first; fallback to national `VWAP_{w}`.
* National prices: Window `345to360` falls back to Intraday Auction; subsequent windows forward-fill horizontally; window `0to30` falls back to `VWAP_0to15` / `VWAP_15to30`.
* Volumes, trade counts, net flows: Fill missing values with `0.0`.

## 6. Backtesting Protocol
* Rolling window: Fixed 822 days (~2.25 years).
* Validation window: 30 days prior to test date ($T-30$ to $T-1$).
* Test horizon: Calendar year 2024 (35,132 quarter-hours, zero NaNs).
* Canonical 2024 Naive MAE baselines:
  * DE1: `26.2815 EUR/MWh`
  * DE2: `26.6818 EUR/MWh` (RMSE: `133.8644`, Median: `13.9600`)
  * DE3: `27.2725 EUR/MWh`
  * DE4: `27.2536 EUR/MWh`
* Competition policy: Strict 1:1 evaluation (MAML-NN vs cSVR). Do not create hybrid cross-architecture ensembles.
