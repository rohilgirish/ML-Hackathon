"""PathSnap FastAPI backend (Laptop 1).

Run:  PATHSNAP_EVAL_DIR=/path/to/evaluator_trips uvicorn api.main:app --host 0.0.0.0 --port 8000

The prediction pipeline only reads .track/.nodes/.arcs. The .route file of the selected
trip is read *after* prediction, server-side, to compute accuracy; reference arcs and
reference geometry are never returned to the UI.
"""
import logging
import sys
import time
from pathlib import Path

import numpy as np
import shapely
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathsnap import config  # noqa: E402
from pathsnap.data_io import list_trips  # noqa: E402
from pathsnap.pipeline import PathSnapMatcher, arcs_to_lonlat, evaluate_trip  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    stream=sys.stdout)
log = logging.getLogger("pathsnap.api")

app = FastAPI(title="PathSnap API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

EVAL_DIR = Path(config.EVAL_DIR)
matcher = PathSnapMatcher()
log.info("Loaded model: %s | HMM: %s", matcher.meta["model"], matcher.hmm.to_dict())
log.info("Serving evaluation trips from %s", EVAL_DIR)


class PredictRequest(BaseModel):
    trip_id: str


@app.get("/health")
def health():
    return {"status": "ok", "model": matcher.meta["model"], "eval_dir": str(EVAL_DIR)}


@app.get("/trips")
def trips():
    out = []
    for t in list_trips(EVAL_DIR)[:10]:
        track = EVAL_DIR / t / f"{t}.track"
        n = sum(1 for _ in open(track)) if track.exists() else 0
        out.append({"trip_id": t, "n_points": n})
    log.info("GET /trips -> %d trips", len(out))
    return {"trips": out}


@app.post("/predict")
def predict(req: PredictRequest):
    trip_dir = EVAL_DIR / req.trip_id
    if req.trip_id not in list_trips(EVAL_DIR):
        raise HTTPException(404, f"unknown trip {req.trip_id}")
    log.info("POST /predict trip=%s : running pipeline", req.trip_id)
    t0 = time.perf_counter()
    res = matcher.predict_trip(trip_dir)
    pt = res["pt"]
    ev = evaluate_trip(res, trip_dir)  # server-side scoring only
    elapsed = time.perf_counter() - t0

    # nearby road network for display (input data, not labels)
    hull = shapely.buffer(shapely.linestrings(pt.xy), 120) if len(pt.xy) > 1 else shapely.buffer(shapely.points(pt.xy[0]), 120)
    near = pt.net.tree.query(hull, predicate="intersects")
    net_lines = [[[float(pt.net.nodes_lonlat[f, 1]), float(pt.net.nodes_lonlat[f, 0])],
                  [float(pt.net.nodes_lonlat[t, 1]), float(pt.net.nodes_lonlat[t, 0])]]
                 for f, t in pt.net.arcs[near[:4000]]]
    metrics = {k: ev[k] for k in ("n_obs", "n_reliable", "clf_top1", "hmm_top1", "clf_continuity",
                                  "hmm_continuity", "route_continuity", "route_precision", "route_recall",
                                  "route_f1", "candidate_miss_rate", "hmm_breaks")}
    log.info("POST /predict trip=%s : %d fixes | accuracy clf=%.4f hmm=%.4f | route F1=%.4f | "
             "continuity=%.3f | %.2fs", req.trip_id, metrics["n_obs"], metrics["clf_top1"] or 0,
             metrics["hmm_top1"] or 0, metrics["route_f1"], metrics["route_continuity"], elapsed)
    return {
        "trip_id": req.trip_id,
        "gps": [[float(lat), float(lon)] for lon, lat in pt.lonlat],
        "predicted_route": arcs_to_lonlat(pt, res["route"]),
        "classifier_only_route": arcs_to_lonlat(pt, [a for a in res["clf_arc"] if a >= 0]),
        "predicted_arc_per_point": res["hmm_arc"].tolist(),
        "road_network": net_lines,
        "metrics": metrics,
        "timings": {**{k: round(v, 3) for k, v in res["timings"].items()}, "request_s": round(elapsed, 3)},
        "model": matcher.meta["model"],
    }
