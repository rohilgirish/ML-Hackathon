"""Step 3: train and compare candidate classifiers.

- 5-fold GroupKFold over the 80 training trips (no trip is split across folds)
- refit on all training trips, evaluate on the 10 student-testing trips
Outputs: results/cv_folds.csv, results/model_comparison.csv, results/student_test_scores/*.parquet,
         figures (confusion matrices, training histories)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathsnap import config  # noqa: E402
from pathsnap.dataset import build_partition  # noqa: E402
from pathsnap.data_io import load_trip  # noqa: E402
from pathsnap.features import FEATURES  # noqa: E402
from pathsnap.geo import LocalProjection  # noqa: E402
from pathsnap.network import RoadNetwork  # noqa: E402
from pathsnap.metrics import candidate_metrics, route_continuity, top1_accuracy, top1_predictions  # noqa: E402
from pathsnap.models import MODEL_ZOO  # noqa: E402


def training_rows(feats: pd.DataFrame) -> pd.DataFrame:
    f = feats[feats.reliable]
    return f[f.obs % config.TRAIN_OBS_STRIDE == 0]


def cap_rows(df, max_rows, seed=config.SEED):
    if max_rows is None or len(df) <= max_rows:
        return df
    keys = df[["trip", "obs"]].drop_duplicates().sample(frac=1, random_state=seed)
    per_obs = len(df) / len(keys)
    keys = keys.iloc[: int(max_rows / per_obs)]
    return df.merge(keys, on=["trip", "obs"])


def continuity_for(df_pred: pd.DataFrame, part_dir: Path) -> float:
    vals = []
    for trip, g in df_pred.groupby("trip"):
        tr = load_trip(part_dir / trip, with_route=False)
        net = RoadNetwork(tr.nodes, tr.arcs, LocalProjection.for_points(tr.track[:, :2]))
        vals.append(route_continuity(g.sort_values("obs")["pred_arc"].tolist(), net))
    return float(np.mean(vals))


def main(only=None):
    np.random.seed(config.SEED)
    tr_f, tr_l = build_partition("train")
    te_f, te_l = build_partition("student_test")
    train = training_rows(tr_f)
    print(f"training rows: {len(train)} ({train[['trip','obs']].drop_duplicates().shape[0]} obs)")
    cv_eval = train  # CV is scored on the same stride-subsampled held-out trips
    gkf = GroupKFold(n_splits=config.CV_FOLDS)
    folds = list(gkf.split(train, train.y, groups=train.trip))

    out_dir = config.RESULTS_DIR / "student_test_scores"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_cv, rows_cmp = [], []
    hist_dir = config.RESULTS_DIR / "histories"
    hist_dir.mkdir(exist_ok=True)

    for name, (family, factory, desc, max_rows) in MODEL_ZOO.items():
        if only and name not in only:
            continue
        print(f"\n=== {name}")
        fold_scores = []
        for k, (tri, vai) in enumerate(folds):
            fit_df = cap_rows(train.iloc[tri], max_rows)
            va = cv_eval.iloc[vai]
            m = factory()
            t0 = time.time()
            m.fit(fit_df[FEATURES].to_numpy(np.float32), fit_df.y.to_numpy())
            fit_t = time.time() - t0
            p = m.predict_proba(va[FEATURES].to_numpy(np.float32))[:, 1]
            cm = candidate_metrics(va.y.to_numpy(), p)
            va = va.assign(p=p)
            cm["top1_acc"] = top1_accuracy(top1_predictions(va), tr_l[tr_l.trip.isin(va.trip.unique()) & (tr_l.obs % config.TRAIN_OBS_STRIDE == 0)])
            cm.update(model=name, fold=k, fit_time_s=fit_t)
            fold_scores.append(cm)
            print(f"  fold {k}: macroF1={cm['macro_f1']:.4f} top1={cm['top1_acc']:.4f} logloss={cm['log_loss']:.4f} ({fit_t:.1f}s)")
        rows_cv.extend(fold_scores)
        fs = pd.DataFrame(fold_scores)

        # refit on all training trips, evaluate on student-testing trips
        fit_df = cap_rows(train, max_rows)
        m = factory()
        t0 = time.time()
        if name in ("LightGBM", "XGBoost"):
            # hold out 10 training trips only to record the boosting history
            trips = np.array(sorted(fit_df.trip.unique()))
            rng = np.random.default_rng(config.SEED)
            hold = set(rng.choice(trips, 10, replace=False))
            a, b = fit_df[~fit_df.trip.isin(hold)], fit_df[fit_df.trip.isin(hold)]
            h = factory()
            if name == "LightGBM":
                h.fit(a[FEATURES], a.y, eval_set=[(a[FEATURES], a.y), (b[FEATURES], b.y)],
                      eval_names=["train", "valid"], eval_metric="binary_logloss")
                hist = {k2: v["binary_logloss"] for k2, v in h.evals_result_.items()}
            else:
                h.fit(a[FEATURES], a.y, eval_set=[(a[FEATURES], a.y), (b[FEATURES], b.y)], verbose=False)
                r = h.evals_result()
                hist = {"train": r["validation_0"]["logloss"], "valid": r["validation_1"]["logloss"]}
            json.dump(hist, open(hist_dir / f"{name}.json", "w"))
        m.fit(fit_df[FEATURES].to_numpy(np.float32), fit_df.y.to_numpy())
        fit_t = time.time() - t0
        if name == "MLP":
            mlp = m[-1]
            json.dump({"train_loss": mlp.loss_curve_, "valid_score": mlp.validation_scores_},
                      open(hist_dir / "MLP.json", "w"))
        # train-set score (for over/underfitting analysis)
        p_tr = m.predict_proba(fit_df[FEATURES].to_numpy(np.float32))[:, 1]
        train_scores = candidate_metrics(fit_df.y.to_numpy(), p_tr)

        t0 = time.time()
        p = m.predict_proba(te_f[FEATURES].to_numpy(np.float32))[:, 1]
        inf_t = time.time() - t0
        te = te_f.assign(p=p)
        te[["trip", "obs", "arc", "p"]].to_parquet(out_dir / f"{name}.parquet")
        rel = te[te.reliable]
        cmt = candidate_metrics(rel.y.to_numpy(), rel.p.to_numpy())
        pred = top1_predictions(te)
        cmt["top1_acc"] = top1_accuracy(pred, te_l)
        cmt["route_continuity"] = continuity_for(pred, config.STUDENT_TEST_DIR)
        cmat = confusion_matrix(rel.y, (rel.p >= 0.5).astype(int))
        np.save(config.RESULTS_DIR / "histories" / f"cm_{name}.npy", cmat)
        rows_cmp.append({
            "model": name, "family": family, "config": desc,
            "cv_macro_f1": fs.macro_f1.mean(), "cv_macro_f1_std": fs.macro_f1.std(),
            "cv_top1": fs.top1_acc.mean(), "cv_log_loss": fs.log_loss.mean(),
            "train_macro_f1": train_scores["macro_f1"], "train_log_loss": train_scores["log_loss"],
            "test_top1": cmt["top1_acc"], "test_macro_f1": cmt["macro_f1"],
            "test_precision": cmt["precision"], "test_recall": cmt["recall"],
            "test_log_loss": cmt["log_loss"], "test_route_continuity": cmt["route_continuity"],
            "fit_time_s": fit_t, "inference_ms_per_1k_obs": 1000 * inf_t / te_l.shape[0] * 1000,
        })
        print(pd.Series(rows_cmp[-1]).to_string())
        pd.DataFrame(rows_cv).to_csv(config.RESULTS_DIR / ("cv_folds.csv" if not only else f"cv_folds_{'_'.join(only)}.csv"), index=False)
        pd.DataFrame(rows_cmp).to_csv(config.RESULTS_DIR / ("model_comparison.csv" if not only else f"model_comparison_{'_'.join(only)}.csv"), index=False)


if __name__ == "__main__":
    main(sys.argv[1:] or None)
