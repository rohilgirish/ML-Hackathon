"""Observation preprocessing, candidate generation and candidate feature engineering.

Each training row is one (GPS fix, candidate arc) pair. The classifier scores how likely the
candidate is the road segment the vehicle was on; the HMM later combines these scores with
network connectivity.
"""
import numpy as np
import pandas as pd
import shapely

from . import config
from .data_io import Trip
from .geo import LocalProjection
from .network import RoadNetwork

FEATURES = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio", "gauss_share",
    "heading_cos", "heading_absdiff", "speed", "heading_valid",
    "arc_length", "proj_t", "proj_inside",
    "n_cands", "n_within_20",
    "ctx_mean_dist", "ctx_max_dist", "ctx_rank", "ctx_minus_min",
    "ctx_heading_cos",
    "from_in_deg", "to_out_deg", "has_reverse",
    "dt_prev", "dt_next",
]

CTX_WINDOW = 3      # neighbouring fixes on each side used for context features
GAUSS_SIGMA = 10.0  # metres, only used to build a soft distance-share feature


class PreparedTrip:
    """Projected observations + network + candidate table for one trip (no route needed)."""

    def __init__(self, trip: Trip):
        self.trip_id = trip.trip_id
        self.proj = LocalProjection.for_points(trip.track[:, :2])
        self.net = RoadNetwork(trip.nodes, trip.arcs, self.proj)
        self.t = trip.track[:, 2]
        self.xy = self.proj.forward(trip.track[:, 0], trip.track[:, 1])
        self.lonlat = trip.track[:, :2]
        self._kinematics()
        self.cands = generate_candidates(self)

    def _kinematics(self):
        xy, t = self.xy, self.t
        n = len(xy)
        nxt = np.r_[np.arange(1, n), n - 1]
        prv = np.r_[0, np.arange(0, n - 1)]
        d = xy[nxt] - xy[prv]
        dt = np.maximum(t[nxt] - t[prv], 1e-6)
        self.heading = np.arctan2(d[:, 0], d[:, 1])
        self.speed = np.hypot(*d.T) / dt
        self.dt_prev = np.r_[0.0, np.diff(t)]
        self.dt_next = np.r_[np.diff(t), 0.0]


def generate_candidates(pt: PreparedTrip) -> pd.DataFrame:
    net, xy = pt.net, pt.xy
    n = len(xy)
    if n == 0 or net.n_arcs == 0:
        return pd.DataFrame(columns=["obs", "arc"])
    pts = shapely.points(xy)
    obs_idx, arc_idx = net.tree.query(pts, predicate="dwithin", distance=config.CAND_RADIUS_M)
    dist, _, _ = net.project(xy[obs_idx], arc_idx)
    df = pd.DataFrame({"obs": obs_idx, "arc": arc_idx, "dist": dist})
    # fallback for fixes with no arc in radius (GPS outliers, sparse networks)
    missing = np.setdiff1d(np.arange(n), obs_idx)
    if len(missing):
        rows = []
        for i in missing:
            near = net.tree.query_nearest(pts[i], all_matches=True)
            # expand to K nearest by searching a growing radius around the nearest one
            d0 = shapely.distance(pts[i], net.geoms[near]).min()
            ids = net.tree.query(pts[i], predicate="dwithin", distance=d0 + 30.0)
            dd = net.project(np.repeat(xy[i:i + 1], len(ids), 0), ids)[0]
            keep = np.argsort(dd)[:config.CAND_FALLBACK_K]
            rows.append(pd.DataFrame({"obs": i, "arc": ids[keep], "dist": dd[keep]}))
        df = pd.concat([df] + rows, ignore_index=True)
    df = df.sort_values(["obs", "dist"], kind="stable")
    df = df[df.groupby("obs").cumcount() < config.CAND_MAX_K].reset_index(drop=True)
    return df


def _angle_diff(a, b):
    return np.abs((a - b + np.pi) % (2 * np.pi) - np.pi)


def build_features(pt: PreparedTrip) -> pd.DataFrame:
    c = pt.cands.copy()
    if c.empty:
        return pd.DataFrame(columns=["obs", "arc"] + FEATURES)
    net = pt.net
    o, a = c["obs"].to_numpy(), c["arc"].to_numpy()
    dist, tproj, off = net.project(pt.xy[o], a)
    c["dist"] = dist
    c["offset"] = off
    g = c.groupby("obs")["dist"]
    dmin = g.transform("min").to_numpy()
    c["dist_rank"] = g.rank(method="first").to_numpy()
    c["dist_minus_min"] = dist - dmin
    c["dist_ratio"] = dist / (dmin + 1.0)
    w = np.exp(-0.5 * (dist / GAUSS_SIGMA) ** 2)
    c["gauss_share"] = w / (pd.Series(w).groupby(o).transform("sum").to_numpy() + 1e-12)

    hdiff = _angle_diff(pt.heading[o], net.bearing[a])
    c["heading_cos"] = np.cos(hdiff)
    c["heading_absdiff"] = np.degrees(hdiff)
    c["speed"] = pt.speed[o]
    c["heading_valid"] = (pt.speed[o] > 1.0).astype(float)

    c["arc_length"] = net.length[a]
    c["proj_t"] = np.clip(tproj, -2, 3)
    c["proj_inside"] = ((tproj >= 0) & (tproj <= 1)).astype(float)
    c["n_cands"] = c.groupby("obs")["arc"].transform("size").to_numpy()
    c["n_within_20"] = pd.Series(dist <= 20).groupby(o).transform("sum").to_numpy()

    # temporal context: how well does this arc explain the neighbouring fixes too?
    n = len(pt.xy)
    ds, hs = [], []
    for k in range(-CTX_WINDOW, CTX_WINDOW + 1):
        if k == 0:
            continue
        j = np.clip(o + k, 0, n - 1)
        ds.append(net.project(pt.xy[j], a)[0])
        hs.append(np.cos(_angle_diff(pt.heading[j], net.bearing[a])) * (pt.speed[j] > 1.0))
    ds = np.vstack(ds + [dist])
    c["ctx_mean_dist"] = ds.mean(0)
    c["ctx_max_dist"] = ds.max(0)
    gc = c.groupby("obs")["ctx_mean_dist"]
    c["ctx_rank"] = gc.rank(method="first").to_numpy()
    c["ctx_minus_min"] = c["ctx_mean_dist"].to_numpy() - gc.transform("min").to_numpy()
    c["ctx_heading_cos"] = np.vstack(hs).mean(0)

    c["from_in_deg"] = net.in_deg[net.arcs[a, 0]]
    c["to_out_deg"] = net.out_deg[net.arcs[a, 1]]
    c["has_reverse"] = net.has_reverse[a].astype(float)
    c["dt_prev"] = np.minimum(pt.dt_prev[o], 600)
    c["dt_next"] = np.minimum(pt.dt_next[o], 600)
    return c
