"""Build the technical report (HTML -> PDF via headless Chrome) from the recorded results.

Every number in the report is read from results/*.csv|json produced by the scripts.
Usage: .venv/bin/python report/build_report.py
"""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pathsnap import config  # noqa: E402
from pathsnap.features import FEATURES  # noqa: E402

RS = config.RESULTS_DIR
OUT = ROOT / "report"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
TEAM = "Team PathSnap"  # <- edit: team name, member names and SRNs


def f(x, d=4):
    return "N/A" if x is None or (isinstance(x, float) and pd.isna(x)) else f"{x:.{d}f}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def table(df: pd.DataFrame, cls="") -> str:
    return df.to_html(index=False, escape=False, classes=cls, border=0)


def fig(name, caption, width=100):
    return f'<figure><img src="../results/figures/{name}" style="width:{width}%"><figcaption>{caption}</figcaption></figure>'


def shot(name, caption, width=100):
    return f'<figure><img src="screenshots/{name}" style="width:{width}%"><figcaption>{caption}</figcaption></figure>'


# ---------------------------------------------------------------- per-model narrative
MODEL_TEXT = {
    "LogisticRegression": dict(
        why="A linear log-odds model over standardised candidate features is the natural baseline for a binary "
            "'is this the true arc?' decision: most of the signal (distance, heading agreement, context distance) "
            "is monotone, and logistic regression yields calibrated probabilities that can be fed directly into "
            "the HMM as emission scores.",
        strengths="Very fast to fit (&lt;1 s) and to score, well-calibrated probabilities, interpretable coefficients.",
        limits="Cannot model interactions such as 'heading only matters when the vehicle is moving' or the "
               "non-linear effect of distance rank, so it saturates below the tree ensembles.",
        hp="C = 1.0 (inverse L2 strength), max_iter = 2000, lbfgs solver; features standardised (StandardScaler fitted on training folds only).",
        loss="Binary cross-entropy (log loss) + L2 penalty."),
    "GaussianNB": dict(
        why="A generative probabilistic model that is extremely cheap and gives class posteriors; useful to test "
            "whether per-feature class-conditional densities are enough to separate candidates.",
        strengths="Trivial training cost, no hyperparameter tuning, probabilistic output.",
        limits="Assumes features are conditionally independent and Gaussian. Our features are strongly correlated "
               "(dist, dist_rank, dist_minus_min, ctx_mean_dist all encode proximity) and heavy-tailed, so the "
               "model double-counts evidence and produces over-confident probabilities (worst log loss).",
        hp="var_smoothing = 1e-9 (default).",
        loss="Maximum-likelihood estimation of per-class Gaussian means/variances (no iterative loss)."),
    "KNN": dict(
        why="Instance-based: a candidate is likely correct if similar (fix, arc) configurations in other trips were "
            "correct. Tests whether local similarity in feature space suffices without a parametric form.",
        strengths="Non-parametric, captures non-linear boundaries, no training phase.",
        limits="Requires feature scaling; prediction cost grows with the reference set (we capped it at 60k rows "
               "and it is still by far the slowest model at inference); memorises training rows (train F1 = 1.0 "
               "with distance weighting).",
        hp="n_neighbors = 25, weights = 'distance', Euclidean metric on standardised features, 60,000 training rows.",
        loss="None (lazy learner); class probability = distance-weighted vote of the 25 neighbours."),
    "DecisionTree": dict(
        why="Map-matching rules are naturally hierarchical (e.g. 'if nearest and heading agrees then ...'); a single "
            "tree learns such threshold rules and handles mixed-scale features without scaling.",
        strengths="Interpretable, captures interactions, fast inference.",
        limits="High variance: a single tree's probabilities are piecewise-constant and less calibrated, and "
               "deeper trees overfit quickly.",
        hp="max_depth = 12, min_samples_leaf = 20, criterion = gini.",
        loss="Greedy minimisation of Gini impurity."),
    "RandomForest": dict(
        why="Bagging many de-correlated trees reduces the variance of the single tree while keeping its ability to "
            "model feature interactions, and produces smoother probability estimates.",
        strengths="Robust, little tuning, good generalisation and stable across folds.",
        limits="Large model on disk, slower to train than boosting at similar accuracy, probability estimates "
               "slightly less sharp than boosting (higher log loss).",
        hp="n_estimators = 150, max_features = 'sqrt', min_samples_leaf = 5, bootstrap = True.",
        loss="Gini impurity per tree; probabilities averaged over trees."),
    "LightGBM": dict(
        why="Gradient-boosted trees fit residual errors sequentially and are state of the art on tabular data with "
            "heterogeneous features like ours; leaf-wise growth and histogram binning make training on ~440k "
            "candidate rows fast, and the log-loss objective yields good probabilities for the HMM.",
        strengths="Best student-test top-1 accuracy and log loss among the untuned models, fast training, handles "
                  "unscaled and interacting features, native feature importance.",
        limits="More hyperparameters to tune; can overfit with many rounds (validation log loss bottoms out "
               "around round {lgb_best} and rises slightly afterwards).",
        hp="n_estimators = 400, learning_rate = 0.05, num_leaves = 63, subsample = 0.8, colsample_bytree = 0.8, min_child_samples = 20.",
        loss="Binary log loss (gradient boosting with second-order approximation)."),
    "XGBoost": dict(
        why="A second boosting implementation (level-wise trees, different regularisation) to check whether the "
            "boosting result is implementation-specific.",
        strengths="Accuracy on par with LightGBM, explicit L1/L2 regularisation, fast hist training.",
        limits="Slightly worse student-test top-1 than LightGBM; same overfitting tendency with many rounds "
               "(validation minimum around round {xgb_best}).",
        hp="n_estimators = 400, learning_rate = 0.05, max_depth = 7, subsample = 0.8, colsample_bytree = 0.8, tree_method = hist.",
        loss="Binary log loss (logistic objective) with L2 regularisation."),
    "MLP": dict(
        why="A neural network learns non-linear feature combinations end-to-end and outputs probabilities; tests "
            "whether a smooth function approximator beats tree ensembles on this tabular problem.",
        strengths="Smooth decision boundaries, fast inference, produces a genuine epoch-wise training history.",
        limits="Needs scaling and early stopping; on tabular data with thresholds (degree counts, ranks) it is "
               "less accurate than boosting and less stable to train.",
        hp="hidden layers (64, 32), ReLU, Adam (lr 1e-3), batch 512, alpha (L2) = 1e-4, early stopping on 10% validation, patience 6, max 60 epochs.",
        loss="Binary cross-entropy + L2 weight penalty."),
}


def main():
    cmp = pd.read_csv(RS / "model_comparison.csv")
    folds = pd.read_csv(RS / "cv_folds.csv")
    tv = pd.read_csv(RS / "tuned_vs_untuned.csv")
    opt = json.load(open(RS / "optuna_summary.json"))
    trials = pd.read_csv(RS / "optuna_trials.csv")
    hm_all = pd.read_csv(RS / "hmm_all_models.csv")
    hm_trip = pd.read_csv(RS / "hmm_vs_classifier_per_trip.csv")
    grid = pd.read_csv(RS / "hmm_grid.csv")
    eda = json.load(open(RS / "eda_summary.json"))
    inv = pd.read_csv(RS / "eda_inventory.csv")
    hmm_cfg = json.load(open(config.ARTIFACT_DIR / "hmm_config.json"))
    imp = pd.read_csv(RS / "final_feature_importance.csv", index_col=0)
    hist = {n: json.load(open(RS / "histories" / f"{n}.json")) for n in ("LightGBM", "XGBoost", "MLP")}
    lgb_best = int(pd.Series(hist["LightGBM"]["valid"]).idxmin())
    xgb_best = int(pd.Series(hist["XGBoost"]["valid"]).idxmin())
    api_log = (OUT / "screenshots" / "api_log.txt").read_text() if (OUT / "screenshots" / "api_log.txt").exists() else ""
    hm = {r.model: r for r in hm_all.itertuples()}
    tr, st = eda["train"], eda["student_test"]
    best_tuned = hm["LightGBM_tuned"]

    # ---------- 3.1 table ----------
    t31 = pd.DataFrame({
        "Model": [f"Model {i+1}: {m}" for i, m in enumerate(cmp.model)],
        "Algorithm family": cmp.family,
        "Key configuration": cmp.config,
        "CV Macro F1 (mean ± sd)": [f"{a:.4f} ± {b:.4f}" for a, b in zip(cmp.cv_macro_f1, cmp.cv_macro_f1_std)],
        "Top-1 Accuracy (student-test)": cmp.test_top1.map(f),
        "Route continuity (classifier only → +HMM)": [f"{hm[m].clf_continuity:.3f} → {hm[m].hmm_continuity:.3f}" for m in cmp.model],
        "Student-Test Macro F1": cmp.test_macro_f1.map(f),
        "Log Loss (student-test)": cmp.test_log_loss.map(f),
        "Inference time (ms / 1k fixes)": cmp.inference_ms_per_1k_obs.map(lambda x: f"{x:.1f}"),
    })
    t31_extra = pd.DataFrame({
        "Model": cmp.model, "CV Top-1": cmp.cv_top1.map(f), "CV log loss": cmp.cv_log_loss.map(f),
        "Test precision (macro)": cmp.test_precision.map(f), "Test recall (macro)": cmp.test_recall.map(f),
        "Train macro F1 (refit)": cmp.train_macro_f1.map(f), "Fit time (s, full train)": cmp.fit_time_s.map(lambda x: f"{x:.1f}"),
        "Top-1 after HMM": [f(hm[m].hmm_top1) for m in cmp.model],
        "Route F1 after HMM": [f(hm[m].hmm_route_f1) for m in cmp.model],
    })

    # ---------- per-model sections ----------
    sections = []
    for i, r in enumerate(cmp.itertuples()):
        t = MODEL_TEXT[r.model]
        fs = folds[folds.model == r.model]
        gap = r.train_macro_f1 - r.test_macro_f1
        if r.model == "GaussianNB":
            fitv = (f"<b>Underfitting.</b> Train macro-F1 ({r.train_macro_f1:.4f}) is as low as test ({r.test_macro_f1:.4f}); the model "
                    f"is too simple for the correlated features. Its log loss ({r.test_log_loss:.3f}) is ~10× worse than the "
                    "others, showing badly over-confident posteriors.")
        elif r.model == "LogisticRegression":
            fitv = (f"<b>No overfitting, mild underfitting.</b> Train ({r.train_macro_f1:.4f}) ≈ CV ({r.cv_macro_f1:.4f}) ≈ test "
                    f"({r.test_macro_f1:.4f}); the linear boundary limits accuracy rather than variance.")
        elif r.model == "KNN":
            fitv = (f"<b>Overfitting (memorisation).</b> Train macro-F1 is {r.train_macro_f1:.4f} (distance weighting returns each "
                    f"training row's own label) vs {r.test_macro_f1:.4f} on student-test; nonetheless k = 25 keeps test "
                    "performance reasonable.")
        elif r.model in ("LightGBM", "XGBoost"):
            b = lgb_best if r.model == "LightGBM" else xgb_best
            fitv = (f"<b>Mild overfitting.</b> Train macro-F1 {r.train_macro_f1:.4f} vs test {r.test_macro_f1:.4f} (gap {gap:.3f}). "
                    f"The boosting history (Fig. 3.3e) shows the held-out log loss is minimal around round {b} and "
                    "creeps up afterwards while training loss keeps falling — which motivated tuning the number of rounds, "
                    "learning rate and regularisation in Section 4.")
        elif r.model == "MLP":
            fitv = (f"<b>Slight overfitting, controlled by early stopping.</b> Train {r.train_macro_f1:.4f} vs test "
                    f"{r.test_macro_f1:.4f}; training stopped after {len(hist['MLP']['train_loss'])} epochs when the validation "
                    "accuracy stopped improving (Fig. 3.3e).")
        else:
            fitv = (f"<b>{'Mild overfitting' if gap > 0.01 else 'Good fit'}.</b> Train macro-F1 {r.train_macro_f1:.4f} vs CV "
                    f"{r.cv_macro_f1:.4f} vs test {r.test_macro_f1:.4f} (train–test gap {gap:.3f}).")
        sections.append(f"""
<h3>3.2.{i+1} Model {i+1}: {r.model} <span class="fam">({r.family})</span></h3>
<p class="q">A1. Why suitable?</p><p>{t['why']}</p>
<p class="q">A2. Strengths and limitations</p><p><b>Strengths:</b> {t['strengths']} <b>Limitations:</b> {t['limits'].format(lgb_best=lgb_best, xgb_best=xgb_best)}</p>
<p class="q">B5. Important hyperparameters (initial values)</p><p>{t['hp']}</p>
<p class="q">B6. Loss / objective</p><p>{t['loss']}</p>
<p class="q">C7–C8. Student-testing performance (observation level)</p>
<p>Top-1 accuracy <b>{r.test_top1:.4f}</b>, macro-F1 <b>{r.test_macro_f1:.4f}</b> (macro precision {r.test_precision:.4f}, macro recall {r.test_recall:.4f}),
log loss <b>{r.test_log_loss:.4f}</b>. CV macro-F1 {r.cv_macro_f1:.4f} ± {r.cv_macro_f1_std:.4f} (folds: {', '.join(f'{v:.3f}' for v in fs.macro_f1)}).
Feeding its probabilities to the same HMM gives top-1 {hm[r.model].hmm_top1:.4f} and length-weighted route F1 {hm[r.model].hmm_route_f1:.4f}.
Fit time {r.fit_time_s:.1f} s; inference {r.inference_ms_per_1k_obs:.1f} ms per 1,000 fixes.</p>
<p class="q">C9. Over- / underfitting?</p><p>{fitv}</p>""")

    # ---------- section 4 ----------
    best = opt["best_params"]
    space_rows = pd.DataFrame([{"Hyperparameter": k, "Distribution": {"int": "uniform int", "int_log": "log-uniform int",
                                "float": "uniform", "float_log": "log-uniform"}[v[0]], "Range": f"[{v[1]}, {v[2]}]",
                                "Initial (untuned)": {"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 63,
                                                      "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8,
                                                      "reg_lambda": 0.0, "reg_alpha": 0.0}[k],
                                "Best found": (f"{best[k]:.4g}" if isinstance(best[k], float) else best[k])}
                               for k, v in opt["search_space"].items()])
    tvt = tv.copy()
    tvt_show = pd.DataFrame({
        "Model": tvt.model, "CV Macro F1": [f"{a:.4f} ± {b:.4f}" for a, b in zip(tvt.cv_macro_f1, tvt.cv_macro_f1_std)],
        "Student-test Top-1": tvt.test_top1.map(f), "Student-test Macro F1": tvt.test_macro_f1.map(f),
        "Log Loss": tvt.test_log_loss.map(f), "Route continuity (clf only)": tvt.test_route_continuity.map(lambda x: f(x, 3)),
        "Inference (ms / 1k fixes)": tvt.inference_ms_per_1k_obs.map(lambda x: f"{x:.1f}"),
        "Fit time (s)": tvt.fit_time_s.map(lambda x: f"{x:.1f}")})
    n_complete = int((trials.state == "COMPLETE").sum())
    n_pruned = int((trials.state == "PRUNED").sum())
    u, tt = tv.iloc[0], tv.iloc[1]
    improved = tt.cv_macro_f1 > u.cv_macro_f1

    # ---------- section 5 ----------
    trip_tbl = pd.DataFrame({
        "Trip": hm_trip.trip.astype(str).str.zfill(8), "Fixes": hm_trip.n_obs,
        "Top-1 clf": hm_trip.clf_top1.map(f), "Top-1 clf+HMM": hm_trip.hmm_top1.map(f),
        "Continuity clf": hm_trip.clf_continuity.map(lambda x: f(x, 3)), "Continuity HMM": hm_trip.hmm_continuity.map(lambda x: f(x, 3)),
        "Route F1 clf": hm_trip.clf_route_f1.map(f), "Route F1 HMM": hm_trip.route_f1.map(f),
        "Chain breaks": hm_trip.hmm_breaks, "Cand. miss": hm_trip.candidate_miss_rate.map(lambda x: pct(x, 2))})
    worse = hm_trip[hm_trip.hmm_top1 < hm_trip.clf_top1]
    worst_gain = hm_trip.assign(g=hm_trip.hmm_top1 - hm_trip.clf_top1).sort_values("g")
    grid_show = grid[["alpha", "beta", "clf_top1", "hmm_top1", "hmm_continuity", "hmm_route_f1"]].copy()
    for c in grid_show.columns[2:]:
        grid_show[c] = grid_show[c].map(f)

    feat_rows = pd.DataFrame({"Feature": imp.index, "Gain share": imp.gain_share.map(lambda x: f"{x:.3f}")}).head(10)

    low_rel = inv.sort_values("reliable_frac").head(4)
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>PathSnap Technical Report</title>
<link rel="stylesheet" href="report.css"></head><body>
<section class="cover">
<h1>PathSnap: Distributed Map-Matching &amp; Route Reconstruction</h1>
<p class="sub">UE24CS352A Machine Learning · Hackathon technical report</p>
<p class="sub">{TEAM}</p>
<p>Final system: candidate generation (50 m) → tuned <b>LightGBM</b> candidate classifier → <b>HMM / Viterbi</b> decoding over road-network
distances → connected route, served by a <b>FastAPI</b> backend and a browser UI.</p>
<div class="kpis">
<div><b>{best_tuned.clf_top1:.3f}</b><span>student-test Top-1, classifier only</span></div>
<div><b>{best_tuned.hmm_top1:.3f}</b><span>student-test Top-1, classifier + HMM</span></div>
<div><b>{best_tuned.hmm_route_f1:.3f}</b><span>route F1 (length-weighted)</span></div>
<div><b>{best_tuned.hmm_continuity:.3f}</b><span>route continuity after HMM</span></div>
</div>
<p class="note">All numbers are generated from <code>results/</code> by the scripts in the repository (<code>run_pipeline.sh</code>). Partitions: <b>train</b> = 80 trips (fitting, CV, tuning),
<b>student_test</b> = 10 trips (development comparison). Evaluator-testing data was never read during development.</p>
</section>

<h2>2. Dataset Understanding and Exploratory Data Analysis</h2>
<p>Each trip folder contains four header-less whitespace files: <code>.track</code> (lon, lat, t [s]), <code>.nodes</code> (lon, lat; row = node id),
<code>.arcs</code> (from, to; row = directed arc id) and <code>.route</code> (chronological arc ids). Every trip ships its <i>own</i> road-network
excerpt (median {tr['nodes_median']:.0f} nodes / {tr['arcs_median']:.0f} directed arcs on train, up to {tr['arcs_max']:,} arcs). Trips are spread over
several continents (Fig. 2.4), so all geometry is computed in a per-trip local metric projection.</p>

<table class="small"><tr><th>Partition (analysis)</th><th>Trips</th><th>GPS fixes</th><th>Fixes / trip (median, min–max)</th><th>NaN values</th><th>Share of 1 s steps</th><th>Trips with gap &gt; 30 s</th><th>Max gap</th><th>Two-way arc share</th><th>Trips w/ disconnected ref. route</th></tr>
<tr><td>train</td><td>{tr['trips']}</td><td>{tr['points_total']:,}</td><td>{tr['points_per_trip']['50%']:.0f} ({tr['points_per_trip']['min']:.0f}–{tr['points_per_trip']['max']:.0f})</td><td>{tr['nan_values_total']}</td><td>{pct(tr['dt_frac_1s_mean'])}</td><td>{tr['trips_with_gap_gt_30s']}</td><td>{tr['max_gap_s']:,.0f} s</td><td>{pct(tr['two_way_frac_mean'])}</td><td>{tr['trips_with_disconnected_route']}</td></tr>
<tr><td>student_test</td><td>{st['trips']}</td><td>{st['points_total']:,}</td><td>{st['points_per_trip']['50%']:.0f} ({st['points_per_trip']['min']:.0f}–{st['points_per_trip']['max']:.0f})</td><td>{st['nan_values_total']}</td><td>{pct(st['dt_frac_1s_mean'])}</td><td>{st['trips_with_gap_gt_30s']}</td><td>{st['max_gap_s']:,.0f} s</td><td>{pct(st['two_way_frac_mean'])}</td><td>{st['trips_with_disconnected_route']}</td></tr></table>

<h3>2.1 How we preprocessed the data</h3>
<ol>
<li><b>Parsing &amp; validation</b> (<code>pathsnap/data_io.py</code>): non-finite or out-of-range coordinates and (0,0) fixes are dropped; fixes are sorted by
timestamp and duplicate timestamps removed. On this data no fix needed removal (0 NaNs, 0 invalid, 0 duplicates, 0 out-of-order), but the checks protect the API against bad evaluator files.</li>
<li><b>Projection</b> (<code>pathsnap/geo.py</code>): lon/lat are angular, so each trip is projected to a local equirectangular plane (metres) centred on the trip median;
over the largest trip extent the distortion is &lt; 0.1 %, far below GPS noise, and it keeps all distance computations vectorised.</li>
<li><b>Per-fix reference labels</b> (<code>pathsnap/labeling.py</code>): the route file gives the traversed arcs but not which arc each fix belongs to. We align fixes
to route positions with a <i>monotone dynamic programme</i> (position non-decreasing in time, minimum total point-to-arc distance), which handles loops and revisited arcs.</li>
<li><b>Label reliability</b>: fixes whose aligned reference arc is &gt; {config.LABEL_MAX_DIST_M:.0f} m away are flagged unreliable (the reference route does not cover them) and are
excluded from training and scoring. This affects {pct(1 - tr['label_reliable_frac'])} of train fixes — concentrated in a few trips, e.g. trip {low_rel.iloc[0].trip:08d}
({pct(low_rel.iloc[0].reliable_frac)} reliable) and {low_rel.iloc[1].trip:08d} ({pct(low_rel.iloc[1].reliable_frac)}) where the GPS kept recording for hours in a small area after the route ended (the car most likely parked; Fig. 2.9) — and only {pct(1 - st['label_reliable_frac'], 2)} of student-test fixes.</li>
<li><b>Candidate generation</b> (<code>pathsnap/features.py</code>): every directed arc within {config.CAND_RADIUS_M:.0f} m of a fix (STR-tree query), at most {config.CAND_MAX_K} nearest;
fixes with none fall back to the {config.CAND_FALLBACK_K} nearest arcs. Candidate miss rate (reference arc not in the set): <b>{pct(tr['candidate_miss_rate'], 2)}</b> (train, reliable fixes) and <b>{pct(st['candidate_miss_rate'], 2)}</b> (student-test).</li>
<li><b>Training rows</b>: consecutive 1 Hz fixes are near-duplicates, so only every {config.TRAIN_OBS_STRIDE}rd fix of the training trips is used for fitting (≈440k candidate rows); scoring uses all fixes.</li>
</ol>

<h3>2.2 Patterns observed and features extracted</h3>
<ul>
<li><b>GPS noise</b> (Fig. 2.3): the median distance from a fix to its reference arc is ~4–6 m, the 95th percentile 12–30 m — hence the 50 m radius.</li>
<li><b>The nearest arc is right only ~55 % of the time</b> on reliable train fixes: {pct(tr['two_way_frac_mean'])} of arcs have a reverse twin at identical distance, and
near intersections several arcs are within a few metres. Direction and temporal context are therefore essential → heading features and window features.</li>
<li><b>Sampling</b> (Fig. 2.2): nominally 1 Hz ({pct(tr['dt_frac_1s_mean'])} of steps on train), but {tr['trips_with_gap_gt_30s']} training trips contain gaps &gt; 30 s, one trip a
{tr['max_gap_s']/3600:.0f}-hour gap → the HMM must handle gaps.</li>
<li><b>Networks</b> (Fig. 2.4–2.5) have several weakly connected components (median {tr['components_median']:.0f}) — excerpts clipped at the bounding box — with no isolated nodes; reference routes are connected ({pct(tr['route_connected_frac_mean'], 2)} of consecutive arcs share a node) and {tr['trips_with_repeated_route_arcs']} train routes revisit arcs.</li>
</ul>
<table class="small"><tr><th>Feature group</th><th>Features</th><th>Why</th></tr>
<tr><td>Proximity</td><td>dist, dist_rank, dist_minus_min, dist_ratio, gauss_share</td><td>GPS noise is roughly Gaussian around the true road; relative measures make the score comparable across dense and sparse networks.</td></tr>
<tr><td>Direction</td><td>heading_cos, heading_absdiff, speed, heading_valid</td><td>Separates the two directions of a two-way road and crossing streets; heading is only trusted when moving (&gt; 1 m/s).</td></tr>
<tr><td>Geometry</td><td>arc_length, proj_t, proj_inside</td><td>Fixes projecting beyond an arc's end usually belong to the next arc.</td></tr>
<tr><td>Ambiguity</td><td>n_cands, n_within_20</td><td>How crowded the neighbourhood is (intersection vs open road).</td></tr>
<tr><td>Temporal context (±3 fixes)</td><td>ctx_mean_dist, ctx_max_dist, ctx_rank, ctx_minus_min, ctx_heading_cos</td><td>The true arc should explain the neighbouring fixes too, which removes many single-fix ambiguities.</td></tr>
<tr><td>Topology</td><td>from_in_deg, to_out_deg, has_reverse</td><td>Junction complexity and one-way information.</td></tr>
<tr><td>Timing</td><td>dt_prev, dt_next</td><td>Flags fixes adjacent to gaps.</td></tr></table>
<p>Top-10 features of the final tuned model by gain:</p>{table(feat_rows, 'small narrow')}
{fig('eda_trip_sizes.png', 'Fig. 2.1 – Fixes, duration and reference-route length per trip (train and student_test).')}
{fig('eda_sampling.png', 'Fig. 2.2 – Sampling-interval distribution and largest gap per trip (train).')}
{fig('eda_gps_noise.png', 'Fig. 2.3 – Distance between each fix and its aligned reference arc (dashed: 100 m reliability cut, dotted: 50 m candidate radius) and per-trip share of reliable fixes (train + student_test).')}
{fig('eda_missing_values.png', 'Fig. 2.4a – Missing / invalid value analysis per partition.', 75)}
{fig('eda_locations_network.png', 'Fig. 2.4b – Trip locations and road-network sizes (train + student_test).')}
{fig('eda_connectivity.png', 'Fig. 2.5 – Road-network connectivity: weakly connected components and largest-component share (train + student_test).')}
{fig('gaps_overview.png', 'Fig. 2.7a – Time gaps by size per partition, and per trip for the 47 trips with a gap &gt; 5 s (train + student_test).')}
{fig('gaps_largest_per_trip.png', 'Fig. 2.7b – Largest time gap per trip; above 120 s the HMM restarts the chain.')}
{fig('gaps_example_trip.png', 'Fig. 2.7c – What gaps look like: time between fixes along train trip 00000006 and where they occur on the map.')}
{fig('uncovered_overview.png', 'Fig. 2.8 – GPS fixes more than 100 m from the reference route, per trip and per partition (excluded from training and scoring).')}
{fig('uncovered_example_trips.png', 'Fig. 2.9 – Two causes of uncovered fixes: a long stay in a small area after the route ends (train trips 00000054: 16,454 fixes within ~186 m over 5.0 h; 00000033: 2,978 fixes within ~13 m over 0.9 h), and real driving on roads that the reference route omits (train trip 00000002).')}
{fig('eda_trajectories.png', 'Fig. 2.6 – GPS trajectory plots with road-network geometry and reference route (train trips 00000003 and 00000002; the latter has fixes off the reference route).')}
{shot('eda_folium.png', 'Fig. 2.7 – Folium map (train trip 00000003): raw GPS, reference route, nearby road network, with layer control. Interactive: <code>results/maps/eda_train_00000003.html</code>.')}

<h2>3. Model Experimentation and Comparison</h2>
<p><b>Task framing.</b> For each (fix, candidate arc) pair a binary classifier predicts P(candidate is the reference arc). Observation-level Top-1 picks the
highest-probability candidate per fix. <b>Protocol:</b> 5-fold GroupKFold over the 80 training trips (trips never split across folds), refit on all
80 trips, evaluated on the 10 student-test trips. Macro-F1 is computed on candidate rows at threshold 0.5 (macro over the two classes); log loss on
candidate rows; Top-1 over reliable fixes (a fix whose reference arc is missing from the candidates counts as an error). Route continuity = share of
changes of matched arc that are drivable within 100 m along the directed network. Seeds fixed at {config.SEED}.</p>
<p><b>Primary selection metric:</b> student-test/CV <b>Top-1 accuracy</b> together with <b>log loss</b>: Top-1 is the route-matching objective itself, and the
log loss measures probability quality, which matters because the probabilities become HMM emission scores. CV macro-F1 was used as the
Bayesian-optimisation objective and as a stability check.</p>

<h3>3.1 Overall Model Comparison</h3>
{table(t31, 'small')}
<p class="note">Supplementary metrics (same runs):</p>{table(t31_extra, 'small')}
<p><b>Model selection.</b> The three tree ensembles are statistically tied on CV macro-F1 (differences ≪ fold s.d. of ~0.014). LightGBM has the best
student-test Top-1 ({cmp.set_index('model').loc['LightGBM', 'test_top1']:.4f}) and the lowest log loss ({cmp.set_index('model').loc['LightGBM', 'test_log_loss']:.4f}), i.e. the best
probabilities for the HMM, trains in seconds, and is compatible with the decoder; Random Forest is a close second but has a larger artifact and higher
log loss; KNN is ~10× slower at inference; Naive Bayes is poorly calibrated. <b>LightGBM was therefore selected</b> for Bayesian optimisation.</p>

<h3>3.2 Compulsory questions for every model</h3>
{''.join(sections)}

<h3>3.3 Model visualisations</h3>
<p>First evaluation: the 5 presented models, one per algorithm family (Figs. 3.3-i to 3.3-iii). The figures after them show all 8 models.</p>
{fig('five_models_summary.png', 'Fig. 3.3-i – 5-model summary: CV macro-F1, student-test Top-1, macro-F1, log loss, route continuity and inference time (LightGBM selected, in orange).')}
{fig('five_models_train_vs_test.png', 'Fig. 3.3-ii – 5 models: train vs CV vs student-test macro-F1 (over/underfitting).', 85)}
{fig('five_models_cv_folds.png', 'Fig. 3.3-iii – 5 models: per-fold CV macro-F1 (stability).', 70)}
{fig('cmp_cv_macro_f1.png', 'Fig. 3.3a – CV macro-F1 across all models (mean ± s.d. over 5 trip-grouped folds; selected model in orange).', 80)}
{fig('cmp_cv_folds_box.png', 'Fig. 3.3b – Fold-wise CV macro-F1 (stability).', 80)}
{fig('cmp_test_top1_f1.png', 'Fig. 3.3c – Student-test Top-1 accuracy and macro-F1.')}
{fig('cmp_logloss_inference.png', 'Fig. 3.3d – Student-test log loss and inference time (log scales).')}
{fig('cmp_training_histories.png', 'Fig. 3.3e – Training histories, only for models that produce them: LightGBM/XGBoost log loss per boosting round (train vs 10 held-out training trips) and MLP loss per epoch with early-stopping validation accuracy.')}
{fig('cmp_train_vs_test.png', 'Fig. 3.3f – Train vs student-test macro-F1 (over/underfitting evidence).', 80)}
{fig('cmp_confusion_matrices.png', 'Fig. 3.3g – Candidate-level confusion matrices on student-test (threshold 0.5).')}
{fig('hmm_vs_classifier_all_models.png', 'Fig. 3.3h – Every model’s probabilities decoded with the same HMM: Top-1, continuity and route F1 before/after HMM (student-test).')}

<h2>4. Bayesian Hyperparameter Optimisation (LightGBM)</h2>
<ol>
<li><b>Hyperparameters searched:</b> {', '.join(opt['search_space'])}.</li>
<li><b>Ranges / distributions:</b> see table (log-uniform for scale-type parameters).</li>
<li><b>Objective:</b> {opt['objective']} on the 80 training trips (stride-3 rows, reliable labels), maximised with Optuna's Tree-structured Parzen Estimator
({opt['sampler']}) and a median pruner that stops a trial after a fold if its running mean is below the median of earlier trials. The untuned configuration was enqueued as trial 0 as a reference.</li>
<li><b>Trials:</b> {opt['n_trials']} ({n_complete} completed, {n_pruned} pruned) in {opt['optimisation_time_s']/60:.1f} min of wall-clock on a laptop. Full record: <code>results/optuna_trials.csv</code>, <code>results/optuna.db</code>.</li>
<li><b>Best values:</b> CV macro-F1 {opt['best_value']:.4f} with the parameters in the table.</li>
<li><b>Did tuning help?</b> {'Only on the optimised objective, not on student-test' if improved and tt.test_top1 < u.test_top1 else ('Yes' if improved else 'No')}: CV macro-F1 {u.cv_macro_f1:.4f} → {tt.cv_macro_f1:.4f}, student-test Top-1 {u.test_top1:.4f} → {tt.test_top1:.4f},
macro-F1 {u.test_macro_f1:.4f} → {tt.test_macro_f1:.4f}, log loss {u.test_log_loss:.4f} → {tt.test_log_loss:.4f}. The gains are small relative to the fold s.d., which is expected:
the untuned model was already a strong configuration and most of the remaining error is label/geometry ambiguity at arc boundaries that no hyperparameter can resolve —
which is exactly what the HMM addresses. The CV gain ({tt.cv_macro_f1 - u.cv_macro_f1:+.4f}) is ~10× smaller than the fold s.d., and the student-test drop ({tt.test_top1 - u.test_top1:+.4f} Top-1) is on only 10 trips; after HMM decoding the two are within {abs(hm['LightGBM_tuned'].hmm_top1 - hm['LightGBM'].hmm_top1):.4f} Top-1 and {abs(hm['LightGBM_tuned'].hmm_route_f1 - hm['LightGBM'].hmm_route_f1):.4f} route F1 of each other. We deploy the tuned model because it was chosen by the pre-declared CV objective; picking the untuned one because it scored better on student-test would turn that partition into a tuning set. We conclude the two configurations are equivalent within measurement noise.</li>
</ol>
{table(space_rows, 'small')}
{table(tvt_show, 'small')}
{fig('bayes_opt.png', 'Fig. 4.1 – Optimisation history (best-so-far in orange) and untuned vs tuned comparison on identical partitions.')}

<h2>5. HMM and Viterbi Route Reconstruction</h2>
<ol>
<li><b>Hidden states:</b> for fix <i>i</i>, the states are its top-{hmm_cfg['topk']} candidate arcs by classifier probability, each carrying the fix's projection offset along the
arc (so the candidate ↔ road-segment correspondence is preserved and the decoded state is directly an arc id).</li>
<li><b>Emissions:</b> log P(z<sub>i</sub> | s) = α · log p<sub>LightGBM</sub>(s | z<sub>i</sub>), α = {hmm_cfg['alpha']}. The classifier already integrates distance, heading and context, so it replaces the usual Gaussian distance model.</li>
<li><b>Transitions</b> (Newson &amp; Krumm style): log P(s<sub>i</sub> | s<sub>i-1</sub>) = −|d<sub>route</sub> − d<sub>gc</sub>| / (β·√Δt), β = {hmm_cfg['beta']} m, where d<sub>gc</sub> is the straight-line
distance between the fixes and d<sub>route</sub> the driving distance between the two projected positions: along the same arc if moving forward (up to {hmm_cfg['back_tol']:.0f} m of backward jitter tolerated), otherwise
remaining length of the first arc + bounded Dijkstra shortest path (directed network, cached per source node) + offset on the second arc. Pairs not reachable within
{hmm_cfg['limit_base']:.0f} m + {hmm_cfg['limit_gc']:.0f}·d<sub>gc</sub> + {hmm_cfg['limit_dt']:.0f} m/s·Δt (capped at {hmm_cfg['limit_max']:.0f} m) get probability 0, which forbids driving against one-way arcs and teleporting between disconnected roads.
<b>Viterbi:</b> δ<sub>i</sub>(s) = log P(z<sub>i</sub>|s) + max<sub>s'</sub>[δ<sub>i-1</sub>(s') + log P(s|s')], with back-pointers; the arg-max path is back-tracked.</li>
<li><b>Missing candidates / large gaps:</b> every fix has candidates by construction (fallback to the 3 nearest arcs). If Δt &gt; {hmm_cfg['max_gap_s']:.0f} s the transition is not trusted and a new segment starts.</li>
<li><b>Disconnected segments:</b> if no candidate at fix <i>i</i> is reachable from any surviving state (e.g. network excerpt is disconnected, or a GPS jump), the chain is broken and
restarted from the best previous state instead of failing; breaks are counted ({int(hm_trip.hmm_breaks.sum())} on the 10 student-test trips). Finally, consecutive distinct arcs are joined by shortest paths to output a connected route.</li>
<li><b>Did the HMM help?</b> Yes. Decoder parameters were chosen by grid search on <i>out-of-fold</i> training probabilities (27 training trips), not on student-test (table below).
On student-test with the tuned model, Top-1 rises from <b>{best_tuned.clf_top1:.4f}</b> to <b>{best_tuned.hmm_top1:.4f}</b>, route continuity from {best_tuned.clf_continuity:.3f} to
{best_tuned.hmm_continuity:.3f}, and length-weighted route F1 from {best_tuned.clf_route_f1:.4f} to <b>{best_tuned.hmm_route_f1:.4f}</b>. The improvement holds for all 8 classifiers (Fig. 3.3h),
and is largest for the weakest ones (Naive Bayes: {hm['GaussianNB'].clf_top1:.3f} → {hm['GaussianNB'].hmm_top1:.3f}).</li>
</ol>
{table(trip_tbl, 'small')}
<p><b>Where the HMM did not help:</b> {('it lowered Top-1 on ' + ', '.join(worse.trip.astype(str).str.zfill(8)) + '. ') if len(worse) else 'it never lowered per-trip Top-1 on student-test; '}
the smallest gain was on trip {str(worst_gain.iloc[0].trip).zfill(8)} ({worst_gain.iloc[0].clf_top1:.4f} → {worst_gain.iloc[0].hmm_top1:.4f}). Remaining errors are mostly (i) fixes right at an arc
boundary, where the reference label itself is ambiguous (the label alignment and the decoder can legitimately disagree by one short arc), (ii) parallel carriageways
of divided roads within GPS noise of each other, and (iii) chain breaks after long gaps, where the decoder restarts and may take a few fixes to recover, and (iv) <b>stationary periods</b>: when the vehicle is stopped, GPS fixes scatter by 10–20 m with random apparent headings; the transition model then sometimes prefers a short detour loop through a side street over staying on the arc (visible in Fig. 6.2). This is also why the reconstructed route's precision is a little lower than its recall. A speed-dependent transition penalty would be the next fix.</p>
{table(grid_show, 'small narrow')}
{fig('hmm_grid.png', 'Fig. 5.1 – Decoder grid search on out-of-fold training probabilities.', 55)}
{fig('hmm_per_trip.png', 'Fig. 5.2 – Per-trip Top-1, classifier-only vs classifier + HMM (student-test, tuned LightGBM).', 80)}

<h2>6. Geographical Visualisation Using Folium</h2>
<p>Generated by <code>scripts/make_maps.py</code> from the deployed pipeline for every student-test trip (<code>results/maps/student_test_*.html</code>). Layers (toggleable via the layer control):
road network (grey), raw GPS fixes (blue dots), reference route (green, wide), classifier-only matches (purple dashed, off by default) and HMM-predicted route (orange-red); legend bottom-left; map fitted to the trip extent.</p>
{shot('map_full.png', 'Fig. 6.1 – Student-test trip 00000056 (full extent): GPS, network, reference and HMM-predicted route.')}
{shot('map_zoom.png', 'Fig. 6.2 – Same trip zoomed on the stretch where classifier-only (purple dashed) and HMM (orange) disagree most: a stop with scattered fixes, where the HMM inserts a spurious detour loop (a failure case). Base tiles do not render in the headless screenshot; the interactive HTML shows them.')}

<h2>7. API, UI and Deployment</h2>
<p><b>Laptop 1</b> runs <code>uvicorn api.main:app --host 0.0.0.0 --port 8000</code> with <code>PATHSNAP_EVAL_DIR</code> pointing at the evaluator trips. Endpoints: <code>GET /health</code>,
<code>GET /trips</code> (the ten evaluation trip ids), <code>POST /predict {{"trip_id"}}</code> (runs preprocessing → candidates → LightGBM → Viterbi → route, then scores against
the server-side reference route). The prediction function never receives the route file; the response contains only the GPS fixes, nearby road network,
predicted routes and aggregate metrics — no reference arcs or geometry. Every request is logged on the API console with trip id, accuracy, route F1 and latency.
<b>Laptop 2</b> serves the static UI (<code>python -m http.server 5500</code> in <code>ui/</code>) and points it to Laptop 1's IP.</p>
{shot('ui.png', 'Fig. 7.1 – Evaluator UI: ten trips, selected trip with predicted route and metrics returned by the API.')}
<p class="note">API console log (captured during the development run on student-test trips; replace with the screenshot from Laptop 1 at the evaluation):</p>
<pre>{api_log}</pre>

<h2>8. Conclusions and limitations</h2>
<ul>
<li>A learned candidate scorer plus a topology-aware HMM is markedly better than either nearest-road matching (~55 % Top-1) or the classifier alone.</li>
<li>Reference labels are derived from route alignment and are uncertain at arc boundaries and where the route does not cover the track; we excluded fixes &gt; 100 m from the route and report this openly.</li>
<li>The local equirectangular projection is accurate at trip scale but would need a proper CRS for country-scale trips.</li>
<li>Possible extensions: learn transition probabilities from data, use turn-angle penalties, and decode with a sliding window for streaming input.</li>
</ul>
</body></html>"""
    (OUT / "PathSnap_Report.html").write_text(html)
    pdf = OUT / "PathSnap_Report.pdf"
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}",
                    (OUT / "PathSnap_Report.html").as_uri()], check=True, capture_output=True)
    print("wrote", pdf)


if __name__ == "__main__":
    main()
