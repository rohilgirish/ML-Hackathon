"""Road network: projected geometry, spatial index and bounded shortest paths."""
import heapq

import numpy as np
import shapely
from scipy.sparse import csr_matrix

from .geo import LocalProjection


class RoadNetwork:
    def __init__(self, nodes_lonlat: np.ndarray, arcs: np.ndarray, proj: LocalProjection):
        self.proj = proj
        self.nodes_lonlat = nodes_lonlat
        self.xy = proj.forward(nodes_lonlat[:, 0], nodes_lonlat[:, 1])
        self.arcs = arcs.astype(np.int64)
        self.n_nodes, self.n_arcs = len(nodes_lonlat), len(arcs)
        a, b = self.arcs[:, 0], self.arcs[:, 1]
        self.p0, self.p1 = self.xy[a], self.xy[b]
        d = self.p1 - self.p0
        self.length = np.hypot(d[:, 0], d[:, 1])
        self.bearing = np.arctan2(d[:, 0], d[:, 1])  # radians clockwise from north
        self.geoms = shapely.linestrings(np.stack([self.p0, self.p1], axis=1))
        self.tree = shapely.STRtree(self.geoms)

        self.out_deg = np.bincount(a, minlength=self.n_nodes)
        self.in_deg = np.bincount(b, minlength=self.n_nodes)
        # has_reverse: does arc (b -> a) exist for arc (a -> b)?
        keys = a * self.n_nodes + b
        rev = b * self.n_nodes + a
        self.has_reverse = np.isin(rev, keys)
        # adjacency in CSR form: node -> outgoing arcs
        order = np.argsort(a, kind="stable")
        self._adj_arcs = order
        self._adj_ptr = np.r_[0, np.cumsum(self.out_deg)]
        self._sp_cache: dict[int, tuple[float, dict]] = {}

    # ---------- geometry ----------
    def project(self, pts: np.ndarray, arc_ids: np.ndarray):
        """Project points onto arcs (pairwise). Returns (distance, t_unclipped, offset_along_arc)."""
        p0, p1, L = self.p0[arc_ids], self.p1[arc_ids], self.length[arc_ids]
        d = p1 - p0
        L2 = np.maximum(L ** 2, 1e-9)
        t = ((pts - p0) * d).sum(1) / L2
        tc = np.clip(t, 0, 1)
        foot = p0 + d * tc[:, None]
        dist = np.hypot(*(pts - foot).T)
        return dist, t, tc * L

    def point_on_arc(self, arc_id: int, offset: float) -> np.ndarray:
        L = max(self.length[arc_id], 1e-9)
        return self.p0[arc_id] + (self.p1[arc_id] - self.p0[arc_id]) * min(max(offset / L, 0), 1)

    # ---------- connectivity ----------
    def out_arcs(self, node: int) -> np.ndarray:
        return self._adj_arcs[self._adj_ptr[node]:self._adj_ptr[node + 1]]

    def sp_from(self, src: int, limit: float) -> dict:
        """Bounded Dijkstra from a node. Cached; returns {node: (dist, pred_arc)}."""
        cached = self._sp_cache.get(src)
        if cached is not None and cached[0] >= limit:
            return cached[1]
        dist = {src: (0.0, -1)}
        heap = [(0.0, src)]
        done = set()
        while heap:
            du, u = heapq.heappop(heap)
            if u in done:
                continue
            done.add(u)
            for e in self.out_arcs(u):
                v = self.arcs[e, 1]
                nd = du + self.length[e]
                if nd <= limit and (v not in dist or nd < dist[v][0]):
                    dist[v] = (nd, int(e))
                    heapq.heappush(heap, (nd, v))
        self._sp_cache[src] = (limit, dist)
        return dist

    def node_dist(self, u: int, v: int, limit: float) -> float:
        r = self.sp_from(u, limit).get(v)
        return r[0] if r is not None and r[0] <= limit else np.inf

    def path_arcs(self, u: int, v: int, limit: float) -> list[int] | None:
        """Arcs of the shortest path u -> v (empty if u == v, None if unreachable within limit)."""
        tree = self.sp_from(u, limit)
        if v not in tree:
            return None
        path = []
        while v != u:
            e = tree[v][1]
            path.append(e)
            v = int(self.arcs[e, 0])
        return path[::-1]

    def components(self):
        """Weakly connected components and isolated nodes (for EDA)."""
        from scipy.sparse.csgraph import connected_components
        g = csr_matrix((np.ones(self.n_arcs), (self.arcs[:, 0], self.arcs[:, 1])),
                       shape=(self.n_nodes, self.n_nodes))
        n, labels = connected_components(g, directed=True, connection="weak")
        deg = self.in_deg + self.out_deg
        return n, labels, int((deg == 0).sum())
