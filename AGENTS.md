# Agent Instructions & Workspace Rules

> **Top Rule:** No human likes to read a lot of text. Especially not in the documentation. Therefore, avoid lengthy explanations and stick to the point.

## 1. Language Policy
* **User Chat:** Match user's language (German/English).
* **Code & Artifacts:** Strictly English (code, documentation, comments, commit messages).

## 1.1 Literature Ground Truth & Knowledge Base
* Before citing papers, formulas, or ensemble schemes, **always consult the immutable knowledge base:** [`old/LITERATURE_KNOWLEDGE_BASE.md`](old/LITERATURE_KNOWLEDGE_BASE.md).
* **Marcjasz et al. (2018, *Energies* / `old/energies-11-02364.pdf`):** Origin of inverse-MAE window weighting ($w \propto 1/\text{MAE}$, Eq. 5).
* **Marcjasz et al. (2020, *Energies* / `old/energies-13-01667-2.pdf`):** Simple 50:50 arithmetic average with Naive ($\text{ens} = 0.5\hat{y} + 0.5y_{\text{naive}}$, Eq. 15).
* **Puć & Janczura (2024, IJF / `old/2411.16237v1.pdf`):** linear inverse-MAE rolling weighting ($p=1.0$, Eq. 25 & `intel_avg_generator.py`).

## 1.2 Formula Rendering Policy (Chat Readability Standard)
* **No Raw LaTeX in Chat:** Never output unrendered LaTeX math tags (`$...$` or `$$...$$`) in chat responses, as the IDE markdown renderer does not parse them into math.
* **Standard Representation:**
  1. **Rendered Formula Images (Primary):** Render non-trivial equations on-the-fly as crisp PNG images via Python (`matplotlib`, `dpi=250`, `facecolor='white'`) into the scratch/artifacts folder and embed them via `![caption](file:///path/to/image.png)`.
  2. **Margin & Padding Rule:** Always set balanced margins (`pad_inches=0.15`, `bbox_inches='tight'`) so that indices, superscripts, and fractions are never cropped or touching the borders.
  3. **GitHub Math Blocks (Secondary / Fallback):** For short expressions, use standard fenced ````math ```` blocks.

## 2. Core Methodological Rules

### 2.1 Time Series Grid & UTC Base
* All data must use strictly **UTC (`+00:00` / `Z`)**.
* Master index `Date` represents delivery start time ($t_{\text{delivery}}$).
* Continuous 15-minute grid must remain complete and unbroken across the entire dataset. Never drop rows.

### 2.2 Targets & Benchmark
* **Regional Forecasting Targets:** `DE1_VWAP_0to30`, `DE2_VWAP_0to30`, `DE3_VWAP_0to30`, `DE4_VWAP_0to30`.
* **Primary Reference Delivery Area:** Always prioritize **DE2 (Amprion)** for detailed cross-model comparisons, deep-dives, and reporting, as it represents a liquid and structurally complete electricity delivery area in Germany. Avoid relying solely on DE1 due to regional structural idiosyncrasies.
* **Naive Benchmark:** `{zone}_VWAP_90to105` (last fully observed and reported window at the effective 90-minute pre-delivery cutoff).
* **National `VWAP_0to30` Rule:** Strictly an internal technical imputation fallback for zero-trade regional intervals. Never use it as a target, feature, or benchmark.

### 2.3 Look-Ahead Bias, Reporting Delay & Feature Shifting
* **Prediction Cutoff & Reporting Delay:**
  * Nominal trading cutoff is **60 minutes before delivery** ($t_{\text{delivery}} - 60\text{ min}$).
  * An operational publication and processing delay of **30 minutes** (2 QH) is enforced to ensure all transaction indices are strictly ex-ante (rounding up Puć et al.'s 20-minute buffer to the 15-minute grid granularity).
  * Consequently, the **effective information availability cutoff** is **90 minutes before delivery** ($t \le t_{\text{delivery}} - 90\text{ min}$).
  * For target contract $T$ ($k = 0$), the last valid, non-leaking window is `{zone}_VWAP_90to105`.
* **Neighboring Contract Temporal Availability Rule:**
  * For neighbor contract $k \in [-4, +2]$ delivered at $t_{\text{delivery}}(T) + k \cdot 15\text{ min}$, a VWAP window ending $A$ minutes before its delivery is available without look-ahead bias if and only if:
    $$A \ge 90 + 15 \cdot k$$
  * Safe windows: $k=-4 \to \text{30to45}$; $k=-3 \to \text{45to60}$; $k=-2 \to \text{60to75}$; $k=-1 \to \text{75to90}$; $k=0 \to \text{90to105}$; $k=+1 \to \text{105to120}$; $k=+2 \to \text{120to135}$.
* **Ex-Post Features (Fundamentals & Reserves):**
  * All realized/ex-post features (actual generation, demand, cross-border flows, balancing activations flagged with `shift_needed_2h = "x"` in `features.csv`) must be lagged by at least **8 quarter-hours (120 minutes / 2 hours)** prior to delivery.
* **No Backward Fill:** `bfill` is strictly prohibited.

### 2.4 Feature Catalogue & `keep` Policy (910 Variables in `features.csv`)
* `yes` (338): Direct features for target contract $k=0$ ($t \le t_{\text{delivery}} - 90\text{ min}$).
* `only past products` (52): Safe neighbor windows (`30to45`..`75to90`) shifted for $k \in [-4, -1]$. Strictly forbidden as direct features for $k=0$.
* `no` (520): Excluded / inactive variables.

## 3. TSO Delivery Area Mapping
Always map German TSOs 1:1 to official EPEX delivery area prefixes:
* `DE1_`: TransnetBW (`DE-ENBW`)
* `DE2_`: Amprion (`DE-AMP`)
* `DE3_`: TenneT (`DE-TPS`)
* `DE4_`: 50Hertz (`DE-50HZ`)
* `DE_`: National aggregate

Never use raw company names in code, features, or model configs.

## 4. Missing Data & Imputation Protocol
* **Regional Price Windows (`DEx_VWAP_{w}`):**
  1. Own observed regional VWAP in window $w$.
  2. Fallback: National aggregate in the same window (`VWAP_{w}`).
* **National Price Windows (`VWAP_{w}`):**
  1. Window `345to360`: Fallback to `Intraday Auction Price` (or `Dayahead_Auction_hourly`).
  2. Subsequent windows: Horizontal forward-fill from preceding window within the same contract maturity.
  3. Window `0to30`: Fallback to `VWAP_0to15` / `VWAP_15to30`.
* **Volumes, Trade Counts, Net Flows:** Fill missing values strictly with `0.0`.

## 5. Experimental Setup & Backtesting Protocol
* **Historical Lookback:** Fixed rolling window of **822 days** (~2.25 years, covering full Q4 2021 through 2023 history prior to 2024 test start).
* **Validation Window:** Last **30 days** prior to test date ($T-30$ to $T-1$) for hyperparameter tuning and model adaptation.
* **Test Horizon:** Calendar year **2024** (`2024-01-01` to `2024-12-31`), evaluated out-of-sample across all 96 QH/day (total 35,132 intervals up to 2024-12-31 22:45 UTC).

### 5.1 Canonical 2024 Naive Benchmark Baselines (Ground Truth)
Use these exact MAE baselines to validate future test and model runs for calendar year 2024 ($| \text{VWAP\_0to30} - \text{VWAP\_90to105} |$):
* **DE1:** `26.2815 EUR/MWh`
* **DE2:** `26.6818 EUR/MWh` (RMSE: `133.8644 EUR/MWh`, Median: `13.9600 EUR/MWh`)
* **DE3:** `27.2725 EUR/MWh`
* **DE4:** `27.2536 EUR/MWh`
* **Total 2024 Observations:** $N = 35,132$ (zero NaNs across entire year).

### 5.2 Model Competition Policy: Strict MAML-NN vs cSVR (No Hybrids)
* **Strict Head-to-Head Evaluation:** The research evaluation is strictly designed as a 1:1 contest (**MAML-NN versus cSVR**).
* **No Cross-Architecture Ensembles:** Ensembles combining MAML-NN and cSVR predictions together are **strictly prohibited / not wanted**. Never report or construct hybrid MAML+cSVR models.
* Comparisons must strictly evaluate:
  1. **Individual Feature Sets (S1, S2, S3, S4):** MAML-NN vs cSVR (and LASSO).
  2. **Intra-Architecture Ensembles:** Pure MAML-NN Ensemble (S1..S4) vs Pure cSVR Ensemble (S1..S4).


