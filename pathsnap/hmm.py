"""HMM map matching with Viterbi decoding.

Hidden state at time i  : one candidate road arc for GPS fix i (with the fix's projection
                          offset along that arc).
Emission  log P(z_i|s)  : alpha * log p_clf(s | z_i) -- the trained classifier's probability
                          that the candidate is the true arc.
Transition log P(s'|s)  : -|d_route(s, s') - d_gc(z_{i-1}, z_i)| / (beta * sqrt(dt))
                          (Newson & Krumm 2009): the driven network distance between the two
                          projected positions should be close to the straight-line distance
                          between the fixes. Unreachable pairs (beyond a distance bound) get
                          -inf, which forbids disconnected jumps.
If every transition at a step is impossible (disconnected candidates, or a GPS gap longer
than max_gap_s) the chain is broken and restarted from the best previous state, so a
decode always returns an arc for every fix.
"""
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .network import RoadNetwork


@dataclass
class HMMParams:
    alpha: float = 1.0        # weight of classifier emission
    beta: float = 5.0         # metres; scale of the transition exponential
    topk: int = 8             # candidates per fix kept as HMM states
    back_tol: float = 15.0    # metres of backward movement on the same arc tolerated (GPS noise)
    limit_base: float = 100.0 # route-distance bound = limit_base + limit_gc*gc + limit_dt*dt
    limit_gc: float = 2.0
    limit_dt: float = 5.0
    limit_max: float = 5000.0 # hard cap on the route-distance search (metres)
    max_gap_s: float = 120.0  # longer GPS gaps start a new, independently decoded segment

    def to_dict(self):
        return asdict(self)


def _route_dist(net: RoadNetwork, a, oa, b, ob, limit, tol, cache):
    if a == b and ob >= oa - tol:
        return max(ob - oa, 0.0)
    u, v = int(net.arcs[a, 1]), int(net.arcs[b, 0])
    key = (u, v)
    nd = cache.get(key)
    if nd is None or (np.isinf(nd[0]) and nd[1] < limit):
        nd = (net.node_dist(u, v, limit), limit)
        cache[key] = nd
    d = (net.length[a] - oa) + nd[0] + ob
    return d if d <= limit else np.inf


def viterbi(net: RoadNetwork, xy: np.ndarray, t: np.ndarray, cands: pd.DataFrame,
            params: HMMParams):
    """cands needs columns obs, arc, offset, p. Returns (arc per obs, n_breaks)."""
    n = len(xy)
    c = cands.sort_values(["obs", "p"], ascending=[True, False])
    c = c[c.groupby("obs").cumcount() < params.topk]
    groups = {o: g for o, g in c.groupby("obs", sort=True)}
    steps = [o for o in range(n) if o in groups]
    if not steps:
        return np.full(n, -1), 0
    arcs_l, offs_l, back_l = [], [], []
    cache = {}
    breaks = 0
    prev = None
    delta = None
    for o in steps:
        g = groups[o]
        A = g["arc"].to_numpy()
        Off = g["offset"].to_numpy()
        em = params.alpha * np.log(np.clip(g["p"].to_numpy(), 1e-6, 1.0))
        if prev is None:
            delta = em
            back = np.full(len(A), -1)
        else:
            po, PA, POff = prev
            gc = float(np.hypot(*(xy[o] - xy[po])))
            dt = max(float(t[o] - t[po]), 1.0)
            limit = min(params.limit_base + params.limit_gc * gc + params.limit_dt * dt, params.limit_max)
            scale = params.beta * np.sqrt(dt)
            T = np.full((len(PA), len(A)), -np.inf)
            for i in range(len(PA) if dt <= params.max_gap_s else 0):
                for j in range(len(A)):
                    d = _route_dist(net, PA[i], POff[i], A[j], Off[j], limit, params.back_tol, cache)
                    if np.isfinite(d):
                        T[i, j] = -abs(d - gc) / scale
            M = delta[:, None] + T
            back = M.argmax(0)
            best = M.max(0)
            if not np.isfinite(best).any():
                breaks += 1  # disconnected: restart chain from best previous state
                back = np.full(len(A), int(delta.argmax()))
                best = np.full(len(A), delta.max())
            delta = best + em
            delta = delta - np.nanmax(delta[np.isfinite(delta)])  # keep numbers bounded
        arcs_l.append(A)
        offs_l.append(Off)
        back_l.append(back)
        prev = (o, A, Off)
    # backtrack
    out = np.full(n, -1)
    j = int(np.argmax(delta))
    for k in range(len(steps) - 1, -1, -1):
        out[steps[k]] = arcs_l[k][j]
        j = back_l[k][j]
    return out, breaks


def reconstruct_route(net: RoadNetwork, arc_seq, max_gap_m: float = 3000.0) -> list[int]:
    """Turn per-fix arcs into a connected arc sequence by inserting shortest paths."""
    s = [int(a) for a in arc_seq if a >= 0]
    s = [a for i, a in enumerate(s) if i == 0 or a != s[i - 1]]
    route = s[:1]
    for b in s[1:]:
        a = route[-1]
        u, v = int(net.arcs[a, 1]), int(net.arcs[b, 0])
        if u != v:
            path = net.path_arcs(u, v, max_gap_m)
            if path:
                route.extend(path)
        route.append(b)
    return route
