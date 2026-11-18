# Work split — 4 teammates

Each member **owns** one slice of the system. The owner must be able to explain it in the viva and to rerun it. The rules say *every* member may be asked about *any* part, so everyone should also read the "Everyone" section at the bottom.

Fill in names and SRNs:

| Member | Name / SRN | Area |
|---|---|---|
| **M1** | _____________ | Data, EDA & feature engineering |
| **M2** | _____________ | Model training & comparison (8 models) |
| **M3** | _____________ | Bayesian optimisation + HMM / Viterbi |
| **M4** | _____________ | API, UI, deployment & report |

---

## M1: Data understanding, preprocessing & features

**Files:** `pathsnap/data_io.py`, `pathsnap/geo.py`, `pathsnap/network.py`, `pathsnap/labeling.py`, `pathsnap/features.py`, `pathsnap/dataset.py`, `scripts/eda.py`
**Outputs:** `results/eda_inventory.csv`, `results/eda_summary.json`, `results/figures/eda_*.png`, `results/maps/eda_*.html`
**Report sections:** 2 (Dataset understanding and EDA)

Be ready to explain:
- What `.track` / `.nodes` / `.arcs` / `.route` contain, and how they relate. A row index is the node or arc id.
- Cleaning steps (invalid coordinates, sorting, duplicate timestamps).
- Why a local equirectangular projection instead of raw degrees.
- **Per-fix labels:** the monotone dynamic-programming alignment of GPS fixes to the route, plus the 100 m "reliable label" cut and why trips 54 and 33 are mostly unreliable.
- Candidate generation: 50 m radius, at most 12 candidates, fallback to the 3 nearest; candidate miss rate.
- Each of the 24 features and why it helps. Example: the heading features separate the two directions of a two-way road; this is why the nearest arc is right only about 55% of the time.
- EDA findings: time gaps (up to 42 h), network components, trips with disconnected routes.

To rerun: `.venv/bin/python -W ignore scripts/eda.py`

## M2: Model experimentation & comparison

**Files:** `pathsnap/models.py`, `pathsnap/metrics.py`, `scripts/train_models.py`, `scripts/make_figures.py`
**Outputs:** `results/model_comparison.csv`, `results/cv_folds.csv`, `results/histories/`, `results/student_test_scores/`, `results/figures/cmp_*.png`
**Report sections:** 3.1, 3.2 (all 8 model subsections), 3.3

Be ready to explain:
- The task framing: binary classification per (fix, candidate) pair, then Top-1 is the argmax per fix.
- GroupKFold by trip: why random splits would leak.
- Stride-3 subsampling of training fixes.
- The metric definitions: candidate-level macro-F1, Top-1, log loss, route continuity, inference time.
- For **each** of the 8 models: why it was tried, its hyperparameters, its loss function, and its over/underfitting evidence. Examples: KNN train F1 = 1.0 (memorisation), Naive Bayes underfits, boosting validation loss bottoms out around round 130.
- Why LightGBM was selected: best test Top-1 and log loss, tied on CV, good probabilities for the HMM.

To rerun: `.venv/bin/python -W ignore scripts/train_models.py`

## M3: Bayesian optimisation & HMM/Viterbi

**Files:** `scripts/bayes_opt.py`, `pathsnap/hmm.py`, `scripts/hmm_eval.py`, `artifacts/model_meta.json`, `artifacts/hmm_config.json`
**Outputs:** `results/optuna.db`, `results/optuna_trials.csv`, `results/optuna_summary.json`, `results/tuned_vs_untuned.csv`, `results/hmm_*.csv`, `results/figures/bayes_opt.png`, `results/figures/hmm_*.png`
**Report sections:** 4, 5

Be ready to explain:
- Optuna's TPE (how it differs from grid and random search), the median pruner, and the search space with its log-uniform ranges.
- The objective (CV macro-F1), the 40 trials, and the baseline run as trial 0.
- **The honest result:** tuning improved CV only within noise, and student-test Top-1 dropped. We still deploy the tuned model because selecting on student-test would turn it into a tuning set.
- HMM hidden states: the top-8 candidate arcs per fix, each with its offset along the arc.
- Emission = α·log p(classifier).
- Transition = −|d_route − d_gc| / (β·√Δt), where d_route comes from a bounded Dijkstra search.
- The Viterbi recurrence and backtracking.
- Gaps over 120 s and disconnected candidates cause a chain break and restart.
- Filling the route between matched arcs with shortest paths.
- Decoder grid search on **out-of-fold** training probabilities (α = 1, β = 2).
- Results: Top-1 0.893 → 0.921 and route F1 0.866 → 0.985. Failure cases: trip 00000007, and the detour loop when the vehicle is stopped.

To rerun: `.venv/bin/python -W ignore scripts/bayes_opt.py 40 && .venv/bin/python -W ignore scripts/hmm_eval.py`

## M4: API, UI, deployment, maps & report

**Files:** `api/main.py`, `ui/` (`index.html`, `app.js`, `style.css`), `pathsnap/pipeline.py`, `scripts/make_maps.py`, `report/build_report.py`, `report/take_screenshots.sh`, `README.md`, `run_pipeline.sh`
**Outputs:** `results/maps/student_test_*.html`, `report/screenshots/`, `report/PathSnap_Report.pdf`
**Report sections:** 6, 7, cover/conclusions; final assembly of the PDF

Be ready to explain:
- The endpoints `/health`, `/trips` and `/predict`.
- **Leakage safeguards:**
  - `predict_trip` loads trips with `with_route=False`.
  - Scoring happens afterwards in `evaluate_trip`, on the server.
  - Only aggregate metrics go to the UI.
  - Nothing is hardcoded.
- How the UI talks to the API: CORS, the `?api=` parameter, and the Leaflet layers.
- The Folium map layers, legend and layer control.
- The two-laptop setup: IP address, port 8000, firewall.
- How the report is generated from `results/` (no hand-typed numbers).

To run on demo day:
```bash
# Laptop 1 (API)
PATHSNAP_EVAL_DIR="/path/to/evaluator_test" .venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000
# Laptop 2 (UI)
cd ui && python3 -m http.server 5500   # open http://localhost:5500/?api=http://<LAPTOP1_IP>:8000
```

---

## Demo-day checklist (owner in brackets)

- [ ] Both laptops are on the same network and Laptop 1's IP is known (M4)
- [ ] The evaluator trip folder is copied to Laptop 1 and `GET /trips` lists 10 trips (M4)
- [ ] Run all 10 trips once before the evaluator arrives (M4, M3)
- [ ] Screenshot of Laptop 1's terminal logs, pasted into report section 7, PDF rebuilt (M4)
- [ ] Team names and SRNs filled in here and in `TEAM` in `report/build_report.py` (M4)
- [ ] Each member does a 2-minute dry run explaining their area to the others (all)

## Everyone should know (viva can ask anyone)

- The one-line pipeline: clean + project → 50 m candidates → 24 features → LightGBM probability → HMM/Viterbi over road distances → connected route.
- The headline numbers in the table on the report's cover page.
- That evaluator data was never used for training or tuning, and that ground truth stays on the server.
- Why the HMM helps: it enforces road connectivity and smooths the matched sequence over time.
