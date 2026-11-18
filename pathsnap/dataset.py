"""Build (and cache) the candidate-level feature table for a data partition."""
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from . import config
from .data_io import list_trips, load_route, load_trip
from .features import PreparedTrip, build_features
from .labeling import align_to_route


def label_trip(pt: PreparedTrip, route: np.ndarray) -> pd.DataFrame:
    """Per-observation reference arc. Kept separate from prediction code on purpose."""
    ref, ref_dist, pos = align_to_route(pt.xy, route, pt.net)
    return pd.DataFrame({"obs": np.arange(len(pt.xy)), "ref_arc": ref, "ref_dist": ref_dist,
                         "ref_pos": pos, "reliable": ref_dist <= config.LABEL_MAX_DIST_M})


def build_trip_table(trip_dir) -> tuple[pd.DataFrame, pd.DataFrame]:
    trip = load_trip(trip_dir, with_route=False)
    pt = PreparedTrip(trip)
    feats = build_features(pt)
    lab = label_trip(pt, load_route(trip_dir))
    feats = feats.merge(lab[["obs", "ref_arc", "reliable"]], on="obs", how="left")
    feats["y"] = (feats["arc"] == feats["ref_arc"]).astype(int)
    feats.insert(0, "trip", trip.trip_id)
    lab.insert(0, "trip", trip.trip_id)
    # candidate miss: reference arc absent from the candidate set
    hit = feats.groupby("obs")["y"].max()
    lab["in_cands"] = lab["obs"].map(hit).fillna(0).astype(bool)
    lab["t"] = pt.t
    return feats, lab


def build_partition(part: str, n_jobs: int = -1):
    cache_f = config.CACHE_DIR / f"{part}_features.parquet"
    cache_l = config.CACHE_DIR / f"{part}_labels.parquet"
    if cache_f.exists() and cache_l.exists():
        return pd.read_parquet(cache_f), pd.read_parquet(cache_l)
    if part == "train":
        part_dir = config.TRAIN_DIR
    elif part == "evaluator_test":
        part_dir = config.EVALUATOR_TEST_DIR
    else:
        part_dir = config.STUDENT_TEST_DIR
    trips = list_trips(part_dir)
    t0 = time.time()
    out = Parallel(n_jobs=n_jobs)(delayed(build_trip_table)(part_dir / t) for t in trips)
    feats = pd.concat([o[0] for o in out], ignore_index=True)
    labs = pd.concat([o[1] for o in out], ignore_index=True)
    config.CACHE_DIR.mkdir(exist_ok=True, parents=True)
    feats.to_parquet(cache_f)
    labs.to_parquet(cache_l)
    print(f"[{part}] {len(trips)} trips, {len(labs)} obs, {len(feats)} candidate rows, {time.time()-t0:.1f}s")
    return feats, labs
