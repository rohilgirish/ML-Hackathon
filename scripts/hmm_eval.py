"""Step 5: HMM / Viterbi route reconstruction experiments.

1. Tune decoder parameters (alpha, beta) by grid search on out-of-fold classifier
   probabilities for training trips (no student-test or evaluator data involved).
2. Compare classifier-only vs classifier+HMM on the 10 student-testing trips, for the
   tuned model and for every model from the comparison.
Outputs: artifacts/hmm_config.json, results/hmm_grid.csv, results/hmm_vs_classifier.csv,
         results/hmm_all_models.csv
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathsnap import config  # noqa: E402
from pathsnap.data_io import list_trips, load_trip  # noqa: E402
from pathsnap.features import PreparedTrip, build_features  # noqa: E402
from pathsnap.hmm import HMMParams, reconstruct_route, viterbi  # noqa: E402
from pathsnap.pipeline import evaluate_trip  # noqa: E402


def decode_trip(trip_dir: Path, scores: pd.DataFrame, params_list):
    pt = PreparedTrip(load_trip(trip_dir, with_route=False))
    feats = build_features(pt)
    feats = feats.merge(scores[["obs", "arc", "p"]], on=["obs", "arc"], how="left")
    feats["p"] = feats["p"].fillna(1e-6)
    best = feats.loc[feats.groupby("obs")["p"].idxmax()]
    clf_arc = np.full(len(pt.xy), -1)
    clf_arc[best.obs.to_numpy()] = best.arc.to_numpy()
    out = []
    for params in params_list:
        t0 = time.perf_counter()
        hmm_arc, breaks = viterbi(pt.net, pt.xy, pt.t, feats, params)
        route = reconstruct_route(pt.net, hmm_arc)
        dec_t = time.perf_counter() - t0
        res = {"pt": pt, "cands": feats, "clf_arc": clf_arc, "hmm_arc": hmm_arc, "route": route, "breaks": breaks}
        ev = evaluate_trip(res, trip_dir)
        for k in ("ref_route", "ref_arc_per_obs", "reliable"):
            ev.pop(k)
        ev.update(trip=trip_dir.name, decode_s=dec_t, **params.to_dict())
        out.append(ev)
    return out


def run(part_dir, scores, params_list, trips=None):
    trips = trips or list_trips(part_dir)
    res = Parallel(n_jobs=-1)(delayed(decode_trip)(part_dir / t, scores[scores.trip == t], params_list) for t in trips)
    return pd.DataFrame([r for rs in res for r in rs])


def summarise(df, keys):
    w = df.n_reliable
    agg = df.groupby(keys).apply(lambda g: pd.Series({
        "clf_top1": np.average(g.clf_top1, weights=g.n_reliable),
        "hmm_top1": np.average(g.hmm_top1, weights=g.n_reliable),
        "clf_continuity": g.clf_continuity.mean(), "hmm_continuity": g.hmm_continuity.mean(),
        "clf_route_f1": g.clf_route_f1.mean(), "hmm_route_f1": g.route_f1.mean(),
        "candidate_miss_rate": np.average(g.candidate_miss_rate, weights=g.n_reliable),
        "hmm_breaks": g.hmm_breaks.sum(), "decode_s": g.decode_s.sum(),
    }), include_groups=False)
    return agg.reset_index()


def main():
    rs = config.RESULTS_DIR
    # ---- 1. grid search on OOF training probabilities ----
    oof = pd.read_parquet(rs / "train_oof_tuned.parquet")
    tune_trips = sorted(oof.trip.unique())[::3]  # every 3rd training trip (27 trips) to bound cost
    grid = [HMMParams(alpha=a, beta=b) for a, b in itertools.product([0.5, 1.0, 2.0], [2.0, 5.0, 10.0, 20.0])]
    t0 = time.time()
    g = run(config.TRAIN_DIR, oof, grid, tune_trips)
    g.to_csv(rs / "hmm_grid_per_trip.csv", index=False)
    gs = summarise(g, ["alpha", "beta"]).sort_values("hmm_top1", ascending=False)
    gs.to_csv(rs / "hmm_grid.csv", index=False)
    print(gs.to_string(), f"\ngrid time {time.time()-t0:.0f}s")
    best = gs.iloc[0]
    params = HMMParams(alpha=float(best.alpha), beta=float(best.beta))
    (config.ARTIFACT_DIR / "hmm_config.json").write_text(json.dumps(params.to_dict(), indent=2))

    # ---- 2. student-test: classifier-only vs classifier + HMM ----
    sc_dir = rs / "student_test_scores"
    per_trip = run(config.STUDENT_TEST_DIR, pd.read_parquet(sc_dir / "LightGBM_tuned.parquet"), [params])
    per_trip.to_csv(rs / "hmm_vs_classifier_per_trip.csv", index=False)
    print(per_trip[["trip", "n_reliable", "clf_top1", "hmm_top1", "clf_continuity", "hmm_continuity",
                    "clf_route_f1", "route_f1", "hmm_breaks", "candidate_miss_rate"]].to_string())
    summarise(per_trip.assign(model="LightGBM_tuned"), ["model"]).to_csv(rs / "hmm_vs_classifier.csv", index=False)

    rows = []
    for f in sorted(sc_dir.glob("*.parquet")):
        df = run(config.STUDENT_TEST_DIR, pd.read_parquet(f), [params]).assign(model=f.stem)
        rows.append(summarise(df, ["model"]))
    allm = pd.concat(rows, ignore_index=True)
    allm.to_csv(rs / "hmm_all_models.csv", index=False)
    print(allm.to_string())
    # keep the classifier-only continuity in the comparison tables on the same (final) definition
    cont = dict(zip(allm.model, allm.clf_continuity))
    cmp = pd.read_csv(rs / "model_comparison.csv")
    cmp["test_route_continuity"] = cmp.model.map(cont)
    cmp.to_csv(rs / "model_comparison.csv", index=False)
    tv = pd.read_csv(rs / "tuned_vs_untuned.csv")
    tv["test_route_continuity"] = [cont["LightGBM"], cont["LightGBM_tuned"]]
    tv.to_csv(rs / "tuned_vs_untuned.csv", index=False)


if __name__ == "__main__":
    main()
