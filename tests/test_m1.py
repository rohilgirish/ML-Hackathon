"""Comprehensive test suite for M1 (Data Loading, Geo, Network, Labels, Candidates, Features).

Verifies all components of the M1 pipeline:
1. Data loading and cleaning (.track, .nodes, .arcs, .route)
2. Local equirectangular projection (LocalProjection)
3. RoadNetwork graph construction and bounded shortest paths
4. Monotone label alignment (align_to_route)
5. Kinematics and candidate generation (generate_candidates)
6. All 24 engineered features (build_features)
7. Dataset construction (build_trip_table)
8. Absence of NaN / infinite values
9. Evaluation boundary enforcement (train and student_test only)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pathsnap import config
from pathsnap.data_io import list_trips, load_trip, load_route, clean_track, Trip
from pathsnap.geo import LocalProjection, haversine_m
from pathsnap.network import RoadNetwork
from pathsnap.labeling import align_to_route
from pathsnap.features import PreparedTrip, generate_candidates, build_features, FEATURES
from pathsnap.dataset import build_trip_table, label_trip


def test_imports():
    print("[1/9] Testing imports...", end=" ")
    import pathsnap
    from pathsnap import (
        config,
        data_io,
        geo,
        network,
        labeling,
        features,
        dataset,
    )
    assert len(features.FEATURES) == 24, f"Expected 24 features, got {len(features.FEATURES)}"
    print("PASS")


def test_data_loading():
    print("[2/9] Testing data loading and cleaning...", end=" ")
    train_trips = list_trips(config.TRAIN_DIR)
    assert len(train_trips) == 80, f"Expected 80 training trips, found {len(train_trips)}"
    
    test_trips = list_trips(config.STUDENT_TEST_DIR)
    assert len(test_trips) == 10, f"Expected 10 student-test trips, found {len(test_trips)}"
    
    # Test loading a specific trip
    tid = train_trips[0]
    trip = load_trip(config.TRAIN_DIR / tid, with_route=True)
    assert isinstance(trip, Trip)
    assert trip.track.shape[1] == 3, "Track must have 3 columns (lon, lat, t)"
    assert trip.nodes.shape[1] == 2, "Nodes must have 2 columns (lon, lat)"
    assert trip.arcs.shape[1] == 2, "Arcs must have 2 columns (from_node, to_node)"
    assert trip.route is not None and len(trip.route) > 0, "Route must not be empty"
    
    # Test loading with with_route=False (prediction mode safeguard)
    trip_noroute = load_trip(config.TRAIN_DIR / tid, with_route=False)
    assert trip_noroute.route is None, "Route must be None when with_route=False"
    print(f"PASS (loaded trip {tid}: {len(trip.track)} points, {len(trip.nodes)} nodes, {len(trip.arcs)} arcs)")


def test_projection():
    print("[3/9] Testing local equirectangular projection...", end=" ")
    tid = list_trips(config.TRAIN_DIR)[0]
    trip = load_trip(config.TRAIN_DIR / tid)
    pts = trip.track[:, :2]
    proj = LocalProjection.for_points(pts)
    xy = proj.forward(pts[:, 0], pts[:, 1])
    inv_pts = proj.inverse(xy)
    
    max_err_deg = np.max(np.abs(pts - inv_pts))
    assert max_err_deg < 1e-9, f"Projection roundtrip error too high: {max_err_deg}"
    
    # Check haversine
    dist_h = haversine_m(pts[0, 0], pts[0, 1], pts[1, 0], pts[1, 1])
    dist_proj = np.hypot(*(xy[1] - xy[0]))
    rel_err = abs(dist_h - dist_proj) / max(dist_h, 1e-6)
    assert rel_err < 0.005, f"Projection distance error exceeds 0.5%: {rel_err}"
    print(f"PASS (roundtrip error: {max_err_deg:.2e} deg, rel dist diff: {rel_err:.4%})")


def test_road_network():
    print("[4/9] Testing RoadNetwork construction and graph methods...", end=" ")
    tid = list_trips(config.TRAIN_DIR)[0]
    trip = load_trip(config.TRAIN_DIR / tid)
    proj = LocalProjection.for_points(trip.track[:, :2])
    net = RoadNetwork(trip.nodes, trip.arcs, proj)
    
    assert net.n_nodes == len(trip.nodes)
    assert net.n_arcs == len(trip.arcs)
    assert len(net.geoms) == len(trip.arcs)
    assert net.has_reverse.shape[0] == net.n_arcs
    
    # Test projection of points onto arcs
    p0 = trip.track[:5, :2]
    p0_xy = proj.forward(p0[:, 0], p0[:, 1])
    dists, t_unclipped, offsets = net.project(p0_xy, np.zeros(5, dtype=int))
    assert len(dists) == 5
    assert (dists >= 0).all()
    
    # Test Dijkstra bounded shortest path
    src_node = int(trip.arcs[0, 0])
    sp = net.sp_from(src_node, limit=500.0)
    assert src_node in sp
    assert sp[src_node][0] == 0.0
    
    ncomp, labels, isolated = net.components()
    assert ncomp >= 1
    print(f"PASS ({net.n_nodes} nodes, {net.n_arcs} arcs, {net.has_reverse.mean():.1%} two-way, {ncomp} components)")


def test_label_generation():
    print("[5/9] Testing monotone dynamic programming label alignment...", end=" ")
    tid = list_trips(config.TRAIN_DIR)[0]
    trip = load_trip(config.TRAIN_DIR / tid, with_route=True)
    proj = LocalProjection.for_points(trip.track[:, :2])
    net = RoadNetwork(trip.nodes, trip.arcs, proj)
    xy = proj.forward(trip.track[:, 0], trip.track[:, 1])
    
    labels, dists, pos = align_to_route(xy, trip.route, net)
    assert len(labels) == len(xy)
    assert len(dists) == len(xy)
    assert len(pos) == len(xy)
    # Check monotonicity
    assert (np.diff(pos) >= 0).all(), "Label alignment route positions must be monotonically non-decreasing!"
    
    # Check reliability cutoff
    reliable = dists <= config.LABEL_MAX_DIST_M
    rel_rate = reliable.mean()
    assert rel_rate > 0.8, f"Expected high reliability on normal trip, got {rel_rate:.1%}"
    print(f"PASS (monotone positions confirmed, reliable rate: {rel_rate:.1%})")


def test_candidate_generation():
    print("[6/9] Testing candidate generation and fallback...", end=" ")
    tid = list_trips(config.TRAIN_DIR)[0]
    trip = load_trip(config.TRAIN_DIR / tid)
    pt = PreparedTrip(trip)
    cands = pt.cands
    
    assert not cands.empty
    assert "obs" in cands.columns
    assert "arc" in cands.columns
    assert "dist" in cands.columns
    
    # Every observation must have at least 1 candidate (via radius or fallback)
    obs_covered = set(cands["obs"].unique())
    all_obs = set(range(len(trip.track)))
    assert obs_covered == all_obs, f"Missing candidates for {len(all_obs - obs_covered)} observations"
    
    # Maximum candidates constraint
    max_cands_per_obs = cands.groupby("obs").size().max()
    assert max_cands_per_obs <= config.CAND_MAX_K, f"Exceeded max candidates: {max_cands_per_obs} > {config.CAND_MAX_K}"
    print(f"PASS (100% fixes covered, max candidates/fix: {max_cands_per_obs} <= {config.CAND_MAX_K})")


def test_features():
    print("[7/9] Testing 24 engineered features...", end=" ")
    tid = list_trips(config.TRAIN_DIR)[0]
    trip = load_trip(config.TRAIN_DIR / tid)
    pt = PreparedTrip(trip)
    feat_df = build_features(pt)
    
    for f in FEATURES:
        assert f in feat_df.columns, f"Missing feature: {f}"
    
    assert len(FEATURES) == 24
    
    # Verify groupings:
    # 1. Distance-based
    dist_feats = ["dist", "dist_rank", "dist_minus_min", "dist_ratio", "gauss_share"]
    # 2. Heading and movement
    heading_feats = ["heading_cos", "heading_absdiff", "speed", "heading_valid"]
    # 3. Geometry and projection
    geom_feats = ["arc_length", "proj_t", "proj_inside", "n_cands", "n_within_20"]
    # 4. Temporal-context
    ctx_feats = ["ctx_mean_dist", "ctx_max_dist", "ctx_rank", "ctx_minus_min", "ctx_heading_cos"]
    # 5. Road-network topology & temporal gap
    topo_feats = ["from_in_deg", "to_out_deg", "has_reverse", "dt_prev", "dt_next"]
    
    all_grouped = dist_feats + heading_feats + geom_feats + ctx_feats + topo_feats
    assert set(all_grouped) == set(FEATURES), "Mismatch in feature groups"
    
    # Check for NaN / Inf
    sub = feat_df[FEATURES]
    n_nans = sub.isna().sum().sum()
    n_infs = np.isinf(sub.to_numpy()).sum()
    assert n_nans == 0, f"Found {n_nans} NaN values in features"
    assert n_infs == 0, f"Found {n_infs} infinite values in features"
    print(f"PASS (all 24 features present, 0 NaNs, 0 Infs)")


def test_dataset_construction():
    print("[8/9] Testing dataset table construction for one trip...", end=" ")
    tid = list_trips(config.TRAIN_DIR)[0]
    feats, labs = build_trip_table(config.TRAIN_DIR / tid)
    
    assert "y" in feats.columns
    assert "reliable" in feats.columns
    assert "ref_arc" in feats.columns
    assert "in_cands" in labs.columns
    
    # Check candidate hit rate on reliable points
    rel_labs = labs[labs["reliable"]]
    hit_rate = rel_labs["in_cands"].mean()
    assert hit_rate > 0.95, f"Candidate hit rate too low: {hit_rate:.1%}"
    print(f"PASS (features: {feats.shape}, labels: {labs.shape}, candidate hit rate: {hit_rate:.2%})")


def test_student_test_compatibility():
    print("[9/9] Testing compatibility on student-test partition...", end=" ")
    tid = list_trips(config.STUDENT_TEST_DIR)[0]
    trip = load_trip(config.STUDENT_TEST_DIR / tid, with_route=False)
    pt = PreparedTrip(trip)
    feats = build_features(pt)
    
    assert len(feats) > 0
    assert set(FEATURES).issubset(feats.columns)
    assert feats[FEATURES].isna().sum().sum() == 0
    print(f"PASS (student-test trip {tid}: {len(trip.track)} points, {len(feats)} candidate rows)")


def main():
    print("=" * 70)
    print("Running PathSnap M1 Full Integration & Validation Suite")
    print("=" * 70)
    test_imports()
    test_data_loading()
    test_projection()
    test_road_network()
    test_label_generation()
    test_candidate_generation()
    test_features()
    test_dataset_construction()
    test_student_test_compatibility()
    print("=" * 70)
    print("ALL 9 M1 INTEGRATION AND VALIDATION TESTS PASSED PERFECTLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
