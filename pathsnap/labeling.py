"""Derive a per-observation reference arc from the chronological reference route.

The .route file lists the arcs traversed, not which arc each GPS fix belongs to. We align
fixes to route positions with a monotone dynamic programme: fix i is assigned route
position j(i) with j non-decreasing in time, minimising the total point-to-arc distance.
Monotonicity matters when a route revisits an area (loops, U-turns, parallel streets).
"""
import numpy as np

from .network import RoadNetwork


def align_to_route(obs_xy: np.ndarray, route: np.ndarray, net: RoadNetwork):
    n, m = len(obs_xy), len(route)
    if n == 0 or m == 0:
        return np.full(n, -1), np.full(n, np.inf), np.full(n, -1)
    # cost[i, j] = distance from fix i to route arc j (vectorised, chunked for memory)
    cost = np.empty((n, m), dtype=np.float32)
    for s in range(0, n, 512):
        pts = obs_xy[s:s + 512]
        P = np.repeat(pts, m, axis=0)
        A = np.tile(route, len(pts))
        cost[s:s + len(pts)] = net.project(P, A)[0].reshape(len(pts), m)

    back = np.empty((n, m), dtype=np.int32)
    D = cost[0].astype(np.float64)
    back[0] = np.arange(m)
    for i in range(1, n):
        # best predecessor position k <= j  (prefix minimum)
        best = np.minimum.accumulate(D)
        idx = np.arange(m)
        is_new = np.r_[True, D[1:] < best[:-1]]
        arg = np.maximum.accumulate(np.where(is_new, idx, 0)).astype(np.int32)
        back[i] = arg
        D = best + cost[i]
    j = int(np.argmin(D))
    pos = np.empty(n, dtype=np.int64)
    for i in range(n - 1, -1, -1):
        pos[i] = j
        j = back[i, j]
    labels = route[pos]
    return labels, cost[np.arange(n), pos].astype(float), pos
