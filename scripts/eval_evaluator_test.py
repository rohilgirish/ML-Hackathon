"""Evaluation of Member 3 models (Random Forest & MLP) on evaluator_test.

Outputs:
- results/evaluator_test_comparison.csv
- results/evaluator_test_scores/*.parquet
"""
import time
from pathlib import Path
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pathsnap import config
from pathsnap.dataset import build_partition
from pathsnap.features import FEATURES
from pathsnap.data_io import load_trip
from pathsnap.geo import LocalProjection
from pathsnap.network import RoadNetwork
from pathsnap.metrics import candidate_metrics, route_continuity, top1_accuracy, top1_predictions
from pathsnap.models import MODEL_ZOO


def continuity_for(df_pred: pd.DataFrame, part_dir: Path) -> float:
    vals = []
    for trip, g in df_pred.groupby("trip"):
        tr = load_trip(part_dir / trip, with_route=False)
        net = RoadNetwork(tr.nodes, tr.arcs, LocalProjection.for_points(tr.track[:, :2]))
        vals.append(route_continuity(g.sort_values("obs")["pred_arc"].tolist(), net))
    return float(np.mean(vals))


def main():
    print("=" * 60)
    print("MEMBER 3 EVALUATION ON: evaluator_test")
    print(f"Target Directory: {config.EVALUATOR_TEST_DIR}")
    print("=" * 60)

    tr_f, tr_l = build_partition("train")
    te_f, te_l = build_partition("evaluator_test")

    train = tr_f[tr_f.reliable & (tr_f.obs % config.TRAIN_OBS_STRIDE == 0)]
    X_tr = train[FEATURES].to_numpy(np.float32)
    y_tr = train.y.to_numpy()

    X_te = te_f[FEATURES].to_numpy(np.float32)

    out_dir = config.RESULTS_DIR / "evaluator_test_scores"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for name, (family, factory, desc, _) in MODEL_ZOO.items():
        print(f"\n[Training] Fitting {name} on {len(train)} training samples...")
        m = factory()
        t0 = time.time()
        m.fit(X_tr, y_tr)
        fit_t = time.time() - t0
        print(f"  Fitted in {fit_t:.2f}s")

        print(f"[Inference] Scoring {len(te_f)} candidate rows on evaluator_test...")
        t0 = time.time()
        p = m.predict_proba(X_te)[:, 1]
        inf_t = time.time() - t0

        te = te_f.assign(p=p)
        te[["trip", "obs", "arc", "p"]].to_parquet(out_dir / f"{name}.parquet")

        rel = te[te.reliable]
        cmt = candidate_metrics(rel.y.to_numpy(), rel.p.to_numpy())
        pred = top1_predictions(te)
        cmt["top1_acc"] = top1_accuracy(pred, te_l)
        cmt["route_continuity"] = continuity_for(pred, config.EVALUATOR_TEST_DIR)
        cmt.update({
            "model": name,
            "family": family,
            "config": desc,
            "fit_time_s": fit_t,
            "inference_ms_per_1k_obs": 1000 * inf_t / te_l.shape[0] * 1000,
        })
        results.append(cmt)
        print(f"  Result -> Top-1: {cmt['top1_acc']:.4f} | Macro F1: {cmt['macro_f1']:.4f} | Log Loss: {cmt['log_loss']:.4f} | Continuity: {cmt['route_continuity']:.4f}")

    df_res = pd.DataFrame(results)
    out_csv = config.RESULTS_DIR / "evaluator_test_comparison.csv"
    df_res.to_csv(out_csv, index=False)
    print("\n" + "=" * 60)
    print("FINAL BENCHMARK TABLE ON EVALUATOR_TEST:")
    print("=" * 60)
    print(df_res[["model", "top1_acc", "macro_f1", "precision", "recall", "log_loss", "route_continuity", "inference_ms_per_1k_obs"]].to_string(index=False))
    print(f"\nSaved results to {out_csv}")


if __name__ == "__main__":
    main()
