"""Reading and cleaning the raw per-trip files (.track, .nodes, .arcs, .route)."""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray            # (N, 3) lon, lat, t  -- cleaned
    nodes: np.ndarray            # (V, 2) lon, lat
    arcs: np.ndarray             # (E, 2) from_node, to_node
    route: np.ndarray | None     # (R,) arc ids, None when not loaded
    cleaning: dict = field(default_factory=dict)


def list_trips(part_dir: Path) -> list[str]:
    return sorted(p.name for p in Path(part_dir).iterdir() if p.is_dir() and not p.name.startswith("."))


def _load(path: Path, dtype=float, ncols=None) -> np.ndarray:
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, ncols or 1), dtype=dtype)
    return np.loadtxt(path, dtype=dtype, ndmin=2)


def clean_track(raw: np.ndarray) -> tuple[np.ndarray, dict]:
    """Drop non-finite / implausible fixes, sort by time and drop duplicate timestamps."""
    info = {"raw_points": int(len(raw))}
    ok = np.isfinite(raw).all(axis=1)
    ok &= (np.abs(raw[:, 0]) <= 180) & (np.abs(raw[:, 1]) <= 90)
    ok &= ~((raw[:, 0] == 0) & (raw[:, 1] == 0))
    info["invalid_dropped"] = int((~ok).sum())
    tr = raw[ok]
    order = np.argsort(tr[:, 2], kind="stable")
    info["out_of_order"] = int((np.diff(tr[:, 2]) < 0).sum())
    tr = tr[order]
    keep = np.r_[True, np.diff(tr[:, 2]) > 0]
    info["duplicate_ts_dropped"] = int((~keep).sum())
    tr = tr[keep]
    info["clean_points"] = int(len(tr))
    return tr, info


def load_trip(trip_dir: Path, with_route: bool = True) -> Trip:
    """Load one trip. `with_route=False` is what the prediction pipeline uses, so that the
    reference route can never reach the model."""
    trip_dir = Path(trip_dir)
    tid = trip_dir.name
    raw = _load(trip_dir / f"{tid}.track", ncols=3)
    track, info = clean_track(raw)
    nodes = _load(trip_dir / f"{tid}.nodes", ncols=2)
    arcs = _load(trip_dir / f"{tid}.arcs", dtype=int, ncols=2)
    route = None
    if with_route:
        route = _load(trip_dir / f"{tid}.route", dtype=int, ncols=1).ravel()
    return Trip(tid, track, nodes, arcs, route, info)


def load_route(trip_dir: Path) -> np.ndarray:
    trip_dir = Path(trip_dir)
    return _load(trip_dir / f"{trip_dir.name}.route", dtype=int, ncols=1).ravel()
