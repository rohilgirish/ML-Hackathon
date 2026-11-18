#!/usr/bin/env bash
# Reproduce every result in the report from the raw data (≈30-40 min on a laptop).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
$PY -W ignore scripts/eda.py               # dataset inventory, EDA plots, EDA folium maps
$PY -W ignore scripts/train_models.py      # 8 models: GroupKFold CV + student-test comparison
$PY -W ignore scripts/bayes_opt.py 40      # Optuna TPE on LightGBM, exports artifacts/model.joblib
$PY -W ignore scripts/hmm_eval.py          # HMM decoder tuning + classifier vs classifier+HMM
$PY -W ignore scripts/make_figures.py      # comparison figures
$PY -W ignore scripts/make_maps.py         # folium maps of predicted routes (student-test)
