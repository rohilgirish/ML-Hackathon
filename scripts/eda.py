"""Step 1-2: dataset inventory, data-quality checks and exploratory plots/maps.

Partitions used: train (80 trips) and student_test (10 trips). Evaluator data is never read.
Outputs: results/eda_inventory.csv, results/eda_summary.json, results/figures/eda_*.png,
         results/maps/eda_*.html
"""
import json
import sys
from pathlib import Path

import folium
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shapely  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathsnap import config  # noqa: E402
from pathsnap.data_io import list_trips, load_trip  # noqa: E402
from pathsnap.dataset import build_partition  # noqa: E402
from pathsnap.geo import LocalProjection  # noqa: E402
from pathsnap.network import RoadNetwork  # noqa: E402

FIG = config.FIG_DIR
FIG.mkdir(parents=True, exist_ok=True)
config.MAP_DIR.mkdir(parents=True, exist_ok=True)


def inspect_trip(part, part_dir, tid):
    d = part_dir / tid
    files = {ext: (d / f"{tid}.{ext}") for ext in ("track", "nodes", "arcs", "route")}
    raw = np.loadtxt(files["track"], ndmin=2)
    trip = load_trip(d)
    tr = trip.track
    proj = LocalProjection.for_points(tr[:, :2])
    net = RoadNetwork(trip.nodes, trip.arcs, proj)
    ncomp, comp, isolated = net.components()
    xy = proj.forward(tr[:, 0], tr[:, 1])
    dt = np.diff(tr[:, 2])
    step = np.hypot(*np.diff(xy, axis=0).T)
    r = trip.route
    ref_cont = float(np.mean(trip.arcs[r[:-1], 1] == trip.arcs[r[1:], 0])) if len(r) > 1 else 1.0
    comp_sizes = np.bincount(comp)
    return {
        "partition": part, "trip": tid,
        **{f"{k}_bytes": f.stat().st_size if f.exists() else 0 for k, f in files.items()},
        "track_cols": raw.shape[1], "raw_points": len(raw), "nan_values": int(np.isnan(raw).sum()),
        "invalid_dropped": trip.cleaning["invalid_dropped"], "duplicate_ts": trip.cleaning["duplicate_ts_dropped"],
        "out_of_order": trip.cleaning["out_of_order"],
        "points": len(tr), "duration_s": float(tr[-1, 2] - tr[0, 2]),
        "dt_median": float(np.median(dt)), "dt_frac_1s": float(np.mean(np.isclose(dt, 1.0))),
        "dt_max": float(dt.max()), "gaps_gt_5s": int((dt > 5).sum()), "gaps_gt_30s": int((dt > 30).sum()),
        "max_step_m": float(step.max()), "max_speed_mps": float((step / np.maximum(dt, 1e-6)).max()),
        "track_length_km": float(step.sum() / 1000),
        "lon_min": tr[:, 0].min(), "lon_max": tr[:, 0].max(), "lat_min": tr[:, 1].min(), "lat_max": tr[:, 1].max(),
        "extent_km": float(np.hypot(*(xy.max(0) - xy.min(0))) / 1000),
        "nodes": net.n_nodes, "arcs": net.n_arcs, "two_way_frac": float(net.has_reverse.mean()),
        "arc_len_median_m": float(np.median(net.length)), "components": ncomp,
        "largest_component_frac": float(comp_sizes.max() / net.n_nodes), "isolated_nodes": isolated,
        "route_arcs": len(r), "route_unique_arcs": len(set(r.tolist())), "route_connected_frac": ref_cont,
        "route_length_km": float(net.length[r].sum() / 1000),
    }


def network_near_track(net: RoadNetwork, xy, buffer_m=150):
    hull = shapely.buffer(shapely.linestrings(xy), buffer_m)
    return net.tree.query(hull, predicate="intersects")


def folium_trip_map(part_dir, tid, out_html, labels=None):
    trip = load_trip(part_dir / tid)
    tr = trip.track
    proj = LocalProjection.for_points(tr[:, :2])
    net = RoadNetwork(trip.nodes, trip.arcs, proj)
    xy = proj.forward(tr[:, 0], tr[:, 1])
    near = network_near_track(net, xy)
    m = folium.Map(location=[tr[:, 1].mean(), tr[:, 0].mean()], zoom_start=14, tiles="cartodbpositron")
    fg_net = folium.FeatureGroup(name=f"Road network near track ({len(near)} arcs)")
    for a in near:
        f, t = trip.arcs[a]
        folium.PolyLine([[trip.nodes[f, 1], trip.nodes[f, 0]], [trip.nodes[t, 1], trip.nodes[t, 0]]],
                        color="#9e9e9e", weight=2, opacity=0.7).add_to(fg_net)
    fg_ref = folium.FeatureGroup(name="Reference route")
    for a in trip.route:
        f, t = trip.arcs[a]
        folium.PolyLine([[trip.nodes[f, 1], trip.nodes[f, 0]], [trip.nodes[t, 1], trip.nodes[t, 0]]],
                        color="#2e7d32", weight=5, opacity=0.8).add_to(fg_ref)
    fg_gps = folium.FeatureGroup(name="Raw GPS fixes")
    for i, (lon, lat, t) in enumerate(tr[::2]):
        folium.CircleMarker([lat, lon], radius=2, color="#1565c0", fill=True, fill_opacity=0.9,
                            tooltip=f"fix {2*i}, t={t:.0f}s").add_to(fg_gps)
    for fg in (fg_net, fg_ref, fg_gps):
        fg.add_to(m)
    folium.Marker([tr[0, 1], tr[0, 0]], tooltip="Start", icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([tr[-1, 1], tr[-1, 0]], tooltip="End", icon=folium.Icon(color="red", icon="stop")).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[tr[:, 1].min(), tr[:, 0].min()], [tr[:, 1].max(), tr[:, 0].max()]])
    m.save(out_html)


def main():
    rows = []
    for part, pdir in [("train", config.TRAIN_DIR), ("student_test", config.STUDENT_TEST_DIR)]:
        for t in list_trips(pdir):
            rows.append(inspect_trip(part, pdir, t))
    inv = pd.DataFrame(rows)
    inv.to_csv(config.RESULTS_DIR / "eda_inventory.csv", index=False)

    _, tr_l = build_partition("train")
    _, te_l = build_partition("student_test")
    labs = pd.concat([tr_l.assign(partition="train"), te_l.assign(partition="student_test")])
    rel = labs.groupby(["partition", "trip"]).agg(reliable_frac=("reliable", "mean"),
                                                   ref_dist_median=("ref_dist", "median"),
                                                   ref_dist_p95=("ref_dist", lambda s: s.quantile(0.95))).reset_index()
    inv = inv.merge(rel, on=["partition", "trip"])
    inv.to_csv(config.RESULTS_DIR / "eda_inventory.csv", index=False)

    summ = {}
    for part, g in inv.groupby("partition"):
        summ[part] = {
            "trips": int(len(g)), "points_total": int(g.points.sum()),
            "points_per_trip": g.points.describe().round(1).to_dict(),
            "nan_values_total": int(g.nan_values.sum()), "invalid_dropped": int(g.invalid_dropped.sum()),
            "duplicate_ts": int(g.duplicate_ts.sum()), "out_of_order": int(g.out_of_order.sum()),
            "dt_frac_1s_mean": float(g.dt_frac_1s.mean()),
            "trips_with_gap_gt_30s": int((g.gaps_gt_30s > 0).sum()), "max_gap_s": float(g.dt_max.max()),
            "nodes_median": float(g.nodes.median()), "arcs_median": float(g.arcs.median()),
            "arcs_max": int(g.arcs.max()), "two_way_frac_mean": float(g.two_way_frac.mean()),
            "components_median": float(g.components.median()), "isolated_nodes_total": int(g.isolated_nodes.sum()),
            "route_connected_frac_mean": float(g.route_connected_frac.mean()),
            "trips_with_disconnected_route": int((g.route_connected_frac < 1).sum()),
            "trips_with_repeated_route_arcs": int((g.route_unique_arcs < g.route_arcs).sum()),
            "label_reliable_frac": float(labs[labs.partition == part].reliable.mean()),
            "candidate_miss_rate": float((~labs[(labs.partition == part) & labs.reliable].in_cands).mean()),
        }
    json.dump(summ, open(config.RESULTS_DIR / "eda_summary.json", "w"), indent=2, default=float)
    print(json.dumps(summ, indent=1, default=float))

    # ---------- figures ----------
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})
    colors = {"train": "#3b6ea8", "student_test": "#d9822b"}

    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for part, g in inv.groupby("partition"):
        ax[0].hist(g.points, bins=25, alpha=0.7, label=part, color=colors[part])
        ax[1].hist(g.duration_s / 60, bins=25, alpha=0.7, label=part, color=colors[part])
        ax[2].hist(g.route_arcs, bins=25, alpha=0.7, label=part, color=colors[part])
    ax[0].set(title="GPS fixes per trip", xlabel="fixes")
    ax[1].set(title="Trip duration", xlabel="minutes")
    ax[2].set(title="Reference-route length", xlabel="arcs")
    ax[0].legend()
    fig.tight_layout(); fig.savefig(FIG / "eda_trip_sizes.png"); plt.close(fig)

    # sampling intervals (train)
    dts = []
    for t in list_trips(config.TRAIN_DIR):
        tr = load_trip(config.TRAIN_DIR / t, with_route=False).track
        dts.append(np.diff(tr[:, 2]))
    dts = np.concatenate(dts)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    vals, cnts = np.unique(np.minimum(np.round(dts), 11), return_counts=True)
    ax[0].bar(vals, cnts, color="#3b6ea8")
    ax[0].set_yscale("log"); ax[0].set(title="Sampling interval (train, 11 = ≥11 s)", xlabel="seconds", ylabel="count (log)")
    g = inv[inv.partition == "train"].sort_values("dt_max", ascending=False)
    ax[1].bar(range(len(g)), g.dt_max, color="#3b6ea8")
    ax[1].set_yscale("log"); ax[1].set(title="Largest time gap per training trip", xlabel="trip (sorted)", ylabel="seconds (log)")
    fig.tight_layout(); fig.savefig(FIG / "eda_sampling.png"); plt.close(fig)

    # GPS-to-reference distance and label reliability
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for part in ("train", "student_test"):
        d = labs[labs.partition == part].ref_dist.clip(upper=150)
        ax[0].hist(d, bins=75, alpha=0.6, density=True, label=part, color=colors[part])
    ax[0].axvline(config.LABEL_MAX_DIST_M, color="k", ls="--", lw=1)
    ax[0].axvline(config.CAND_RADIUS_M, color="#c62828", ls=":", lw=1)
    ax[0].set(title="Distance GPS fix → aligned reference arc", xlabel="metres (clipped at 150)", ylabel="density")
    ax[0].legend()
    g = inv.sort_values("reliable_frac")
    ax[1].bar(range(len(g)), g.reliable_frac, color=[colors[p] for p in g.partition])
    ax[1].set(title="Share of fixes within 100 m of the reference route", xlabel="trip (sorted)", ylim=(0, 1.02))
    fig.tight_layout(); fig.savefig(FIG / "eda_gps_noise.png"); plt.close(fig)

    # missing values table as figure
    fig, ax = plt.subplots(figsize=(8, 2.6))
    mv = pd.DataFrame({
        "NaN coordinates/timestamps": inv.groupby("partition").nan_values.sum(),
        "invalid coords dropped": inv.groupby("partition").invalid_dropped.sum(),
        "duplicate timestamps": inv.groupby("partition").duplicate_ts.sum(),
        "gaps > 5 s": inv.groupby("partition").gaps_gt_5s.sum(),
        "gaps > 30 s": inv.groupby("partition").gaps_gt_30s.sum(),
        "empty files": inv.groupby("partition").apply(lambda g: int((g[[c for c in g if c.endswith('_bytes')]] == 0).sum().sum()), include_groups=False),
    }).T
    ax.axis("off")
    tb = ax.table(cellText=mv.values, rowLabels=mv.index, colLabels=mv.columns, loc="center", cellLoc="center")
    tb.scale(1, 1.4)
    ax.set_title("Missing / invalid value analysis (counts)")
    fig.tight_layout(); fig.savefig(FIG / "eda_missing_values.png"); plt.close(fig)

    # world map of trip locations + network size
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for part, g in inv.groupby("partition"):
        ax[0].scatter((g.lon_min + g.lon_max) / 2, (g.lat_min + g.lat_max) / 2, s=18, label=part, color=colors[part])
        ax[1].scatter(g.nodes, g.arcs, s=18, label=part, color=colors[part])
    ax[0].set(title="Trip locations (centroids)", xlabel="longitude", ylabel="latitude", xlim=(-180, 180), ylim=(-60, 75))
    ax[1].set(title="Road-network size per trip", xlabel="nodes", ylabel="arcs (directed)", xscale="log", yscale="log")
    ax[0].legend()
    fig.tight_layout(); fig.savefig(FIG / "eda_locations_network.png"); plt.close(fig)

    # trajectory + road network plot for two trips
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, (pdir, tid) in zip(axes, [(config.TRAIN_DIR, "00000003"), (config.TRAIN_DIR, "00000002")]):
        trip = load_trip(pdir / tid)
        proj = LocalProjection.for_points(trip.track[:, :2])
        net = RoadNetwork(trip.nodes, trip.arcs, proj)
        xy = proj.forward(trip.track[:, 0], trip.track[:, 1])
        near = network_near_track(net, xy, 300)
        for a in near:
            ax.plot(*np.c_[net.p0[a], net.p1[a]], color="#bdbdbd", lw=0.6)
        for a in trip.route:
            ax.plot(*np.c_[net.p0[a], net.p1[a]], color="#2e7d32", lw=2.2)
        ax.scatter(*xy.T, s=2, color="#1565c0", zorder=3)
        ax.set_aspect("equal"); ax.set(title=f"Trip {tid} (train): GPS (blue), reference (green), network (grey)",
                                       xlabel="x east (m)", ylabel="y north (m)")
        ax.title.set_fontsize(9)
    fig.tight_layout(); fig.savefig(FIG / "eda_trajectories.png"); plt.close(fig)

    # connectivity
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].hist(inv.components, bins=30, color="#3b6ea8"); ax[0].set(title="Weakly connected components per network", xlabel="components")
    ax[1].hist(inv.largest_component_frac, bins=30, color="#3b6ea8"); ax[1].set(title="Largest component share of nodes", xlabel="fraction")
    fig.tight_layout(); fig.savefig(FIG / "eda_connectivity.png"); plt.close(fig)

    folium_trip_map(config.TRAIN_DIR, "00000003", config.MAP_DIR / "eda_train_00000003.html")
    folium_trip_map(config.TRAIN_DIR, "00000002", config.MAP_DIR / "eda_train_00000002_anomaly.html")
    print("EDA done")


if __name__ == "__main__":
    main()
