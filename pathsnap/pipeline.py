"""End-to-end inference: raw trip files -> candidates -> classifier -> HMM/Viterbi -> route.

`predict_trip` only ever reads .track/.nodes/.arcs. Reference-route scoring lives in
`evaluate_trip`, which is called separately (server side) after prediction.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config
from .data_io import load_route, load_trip
from .dataset import label_trip
from .features import FEATURES, PreparedTrip, build_features
from .hmm import HMMParams, reconstruct_route, viterbi
from .metrics import route_continuity, route_overlap


class PathSnapMatcher:
    def __init__(self, artifact_dir: Path = config.ARTIFACT_DIR):
        artifact_dir = Path(artifact_dir)
        self.model = joblib.load(artifact_dir / "model.joblib")
        meta = json.loads((artifact_dir / "model_meta.json").read_text())
        self.features = meta["features"]
        assert self.features == FEATURES, "feature list mismatch between code and artifact"
        self.meta = meta
        self.hmm = HMMParams(**json.loads((artifact_dir / "hmm_config.json").read_text()))

    def score(self, feats: pd.DataFrame) -> np.ndarray:
        if feats.empty:
            return np.zeros(0)
        return self.model.predict_proba(feats[self.features].to_numpy(np.float32))[:, 1]

    def predict_trip(self, trip_dir: Path) -> dict:
        timings = {}
        t0 = time.perf_counter()
        trip = load_trip(trip_dir, with_route=False)
        pt = PreparedTrip(trip)
        feats = build_features(pt)
        timings["preprocess_s"] = time.perf_counter() - t0

        t1 = time.perf_counter()
        feats["p"] = self.score(feats)
        timings["classifier_s"] = time.perf_counter() - t1
        n = len(pt.xy)
        clf_arc = np.full(n, -1)
        if not feats.empty:
            best = feats.loc[feats.groupby("obs")["p"].idxmax()]
            clf_arc[best.obs.to_numpy()] = best.arc.to_numpy()

        t2 = time.perf_counter()
        hmm_arc, breaks = viterbi(pt.net, pt.xy, pt.t, feats, self.hmm)
        route = reconstruct_route(pt.net, hmm_arc)
        timings["hmm_s"] = time.perf_counter() - t2
        timings["total_s"] = time.perf_counter() - t0
        return {"trip_id": trip.trip_id, "pt": pt, "cands": feats, "clf_arc": clf_arc,
                "hmm_arc": hmm_arc, "route": route, "breaks": breaks, "timings": timings,
                "cleaning": trip.cleaning}


def evaluate_trip(result: dict, trip_dir: Path) -> dict:
    """Server-side scoring against the reference route (never fed back to prediction)."""
    pt = result["pt"]
    ref_route = load_route(trip_dir)
    lab = label_trip(pt, ref_route)
    rel = lab.reliable.to_numpy()
    ref = lab.ref_arc.to_numpy()
    out = {
        "n_obs": int(len(ref)), "n_reliable": int(rel.sum()),
        "clf_top1": float((result["clf_arc"][rel] == ref[rel]).mean()) if rel.any() else None,
        "hmm_top1": float((result["hmm_arc"][rel] == ref[rel]).mean()) if rel.any() else None,
        "clf_continuity": route_continuity(result["clf_arc"].tolist(), pt.net),
        "hmm_continuity": route_continuity(result["hmm_arc"].tolist(), pt.net),
        "route_continuity": route_continuity(result["route"], pt.net),
        "hmm_breaks": int(result["breaks"]),
    }
    hit = result["cands"].groupby("obs")["arc"].apply(set)
    in_c = np.array([ref[i] in hit.get(i, set()) for i in range(len(ref))])
    out["candidate_miss_rate"] = float((~in_c[rel]).mean()) if rel.any() else None
    out.update(route_overlap(result["route"], ref_route, pt.net.length))
    clf_route = reconstruct_route(pt.net, result["clf_arc"])
    out.update({f"clf_{k}": v for k, v in route_overlap(clf_route, ref_route, pt.net.length).items()})
    out["ref_route"] = [int(a) for a in ref_route]
    out["ref_arc_per_obs"] = ref
    out["reliable"] = rel
    return out


def arcs_to_lonlat(pt: PreparedTrip, arc_ids) -> list[list[list[float]]]:
    """Polyline(s) [[lat, lon], ...] for a connected arc sequence; breaks where disconnected."""
    net = pt.net
    lines, cur = [], []
    prev_to = None
    for a in arc_ids:
        a = int(a)
        f, t = int(net.arcs[a, 0]), int(net.arcs[a, 1])
        if prev_to is None or f != prev_to:
            if cur:
                lines.append(cur)
            cur = [[float(net.nodes_lonlat[f, 1]), float(net.nodes_lonlat[f, 0])]]
        cur.append([float(net.nodes_lonlat[t, 1]), float(net.nodes_lonlat[t, 0])])
        prev_to = t
    if cur:
        lines.append(cur)
    return lines
