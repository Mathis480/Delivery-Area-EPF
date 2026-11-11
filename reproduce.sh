#!/usr/bin/env bash
# Reproduce the paper results in order. Usage: bash reproduce.sh [stage ...]
# Stages: data backtest seeds retrain runtime paper (default: all, in this order).
# Run from the repository root inside the conda environment (see README.md).
set -euo pipefail
cd "$(dirname "$0")"

stage_data() {      # ~2-3 h: raw EPEX trades -> zonal VWAPs, then master dataset and consistency tests
    python "Data/EPEX Intraday Continuous/calculate_ID_DEx_UTC.py"
    jupyter execute create_master_dataset.ipynb
    python -m unittest discover -s tests
}
stage_backtest() {  # ~3.7 h: all models, four zones in parallel
    python run_experiment.py
}
stage_seeds() {     # ~3 h: four further seeds for FA-NN and FOMAML
    for s in 1 2 3 4; do
        python run_experiment.py --models fann --seed "$s"
        python run_experiment.py --models fomaml,fomaml_pre --seed "$s"
    done
}
stage_retrain() {   # ~1.75 h: per-forecast retraining reference on 2023 (Sec. 5.5)
    python Evaluation/backbone_epochs.py --zones DE2
    python Evaluation/retrain_baseline.py --zones DE2 --every 13
    python Evaluation/retrain_baseline.py --summary
    python Evaluation/retrain_baseline.py --timing
}
stage_runtime() {   # ~10 min: single-thread runtime benchmark (Table 3)
    python Evaluation/runtime_benchmark.py
}
stage_paper() {     # ~5 min: tables, figures and numbers quoted in the text
    python Evaluation/make_tables.py
    python Evaluation/data_figures.py
    python Evaluation/draft_figures.py
    python Evaluation/text_numbers.py
}

stages=("$@")
[ ${#stages[@]} -eq 0 ] && stages=(data backtest seeds retrain runtime paper)
for st in "${stages[@]}"; do
    echo "=== $st ($(date +%H:%M))"
    "stage_$st"
done
