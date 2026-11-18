"""Figures explaining two data-quality issues (partitions: train + student_test):
  1. time gaps in the GPS tracks
  2. GPS fixes not covered by the reference route (> LABEL_MAX_DIST_M from it). Two causes:
     long stays in a small area after the route ends (trips 54, 33) and driving on roads that
     the reference route omits (e.g. trip 2)
Outputs: results/figures/gaps_*.png, results/figures/uncovered_*.png, results/maps/uncovered_train_00000054.html
"""
import sys
from pathlib import Path

import folium
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathsnap import config  # noqa: E402
from pathsnap.data_io import list_trips, load_trip  # noqa: E402
from pathsnap.dataset import build_partition  # noqa: E402
from pathsnap.geo import LocalProjection  # noqa: E402
from pathsnap.network import RoadNetwork  # noqa: E402

FIG = config.FIG_DIR
BLUE, ORANGE, RED, GREEN, GREY = "#3b6ea8", "#d9822b", "#c62828", "#2e7d32", "#9aa0a6"
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})
BUCKETS = [(1.5, 5, "2–5 s"), (5, 30, "5–30 s"), (30, 120, "30 s–2 min"), (120, np.inf, "> 2 min")]
BUCKET_COLORS = ["#c6dbef", "#6baed6", "#2171b5", RED]


def gap_table():
    rows = []
    for part, pdir in [("train", config.TRAIN_DIR), ("student_test", config.STUDENT_TEST_DIR)]:
        for t in list_trips(pdir):
            tr = load_trip(pdir / t, with_route=False).track
            dt = np.diff(tr[:, 2])
            r = {"partition": part, "trip": t, "points": len(tr), "max_gap_s": dt.max(),
                 "missing_s": float(np.clip(dt - 1, 0, None).sum()), "duration_s": tr[-1, 2] - tr[0, 2]}
            for lo, hi, lab in BUCKETS:
                r[lab] = int(((dt > lo) & (dt <= hi)).sum())
            rows.append(r)
    return pd.DataFrame(rows)


def plot_gaps(g):
    labs = [b[2] for b in BUCKETS]
    # 1) gap counts by size per partition + per-trip stacked bars for affected trips
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.6), gridspec_kw={"width_ratios": [1, 2.6]})
    parts = ["train", "student_test"]
    x = np.arange(len(labs))
    for k, p in enumerate(parts):
        v = g[g.partition == p][labs].sum().to_numpy()
        ax[0].bar(x + (k - 0.5) * 0.4, v, 0.4, label=p, color=[BLUE, ORANGE][k])
        for i, c in enumerate(v):
            ax[0].text(x[i] + (k - 0.5) * 0.4, c, str(c), ha="center", va="bottom", fontsize=7)
    ax[0].set_xticks(x, labs); ax[0].set_yscale("log")
    ax[0].set(title="How many time gaps, by size", ylabel="number of gaps (log scale)"); ax[0].legend()

    aff = g[g[labs[1:]].sum(1) > 0].sort_values("max_gap_s", ascending=False)
    bottom = np.zeros(len(aff))
    names = [f"{int(t)}" + ("*" if p == "student_test" else "") for t, p in zip(aff.trip, aff.partition)]
    for lab, c in zip(labs, BUCKET_COLORS):
        ax[1].bar(names, aff[lab], bottom=bottom, color=c, label=lab)
        bottom += aff[lab].to_numpy()
    ax[1].set_yscale("symlog", linthresh=10)
    ax[1].set(title=f"Gaps per trip ({len(aff)} trips have a gap > 5 s; sorted by largest gap; * = student_test)",
              xlabel="trip id", ylabel="number of gaps")
    ax[1].tick_params(axis="x", rotation=90, labelsize=7)
    ax[1].legend(title="gap size", ncol=4, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "gaps_overview.png"); plt.close(fig)

    # 2) largest gap per trip vs thresholds
    fig, ax = plt.subplots(figsize=(13, 4))
    s = g.sort_values("max_gap_s", ascending=False).reset_index(drop=True)
    ax.bar(range(len(s)), s.max_gap_s, color=[ORANGE if p == "student_test" else BLUE for p in s.partition])
    for y, txt in [(30, "30 s"), (120, "120 s: HMM restarts the chain"), (3600, "1 hour")]:
        ax.axhline(y, color="k", ls="--", lw=0.8)
        ax.text(len(s) - 1, y * 1.15, txt, ha="right", fontsize=8)
    for i in range(3):
        r = s.iloc[i]
        ax.annotate(f"trip {int(r.trip)}: {r.max_gap_s/3600:.1f} h", (i, r.max_gap_s), xytext=(i + 4, r.max_gap_s),
                    fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.set_yscale("log")
    ax.set(title="Largest time gap in each trip (blue = train, orange = student_test)",
           xlabel="trips, sorted by largest gap", ylabel="seconds (log scale)")
    fig.tight_layout(); fig.savefig(FIG / "gaps_largest_per_trip.png"); plt.close(fig)

    # 3) what a gap looks like: timeline + map for one representative trip
    tid = "00000006"
    trip = load_trip(config.TRAIN_DIR / tid)
    tr = trip.track
    dt = np.r_[1, np.diff(tr[:, 2])]
    proj = LocalProjection.for_points(tr[:, :2])
    xy = proj.forward(tr[:, 0], tr[:, 1])
    fig, ax = plt.subplots(1, 2, figsize=(15, 5), gridspec_kw={"width_ratios": [1.5, 1]})
    ax[0].plot(tr[:, 2] - tr[0, 2], dt, color=BLUE, lw=0.8)
    big = np.where(dt > 5)[0]
    ax[0].scatter(tr[big, 2] - tr[0, 2], dt[big], color=RED, s=18, zorder=3, label="gap > 5 s")
    ax[0].set_yscale("log")
    ax[0].set(title=f"Train trip {tid}: time between consecutive GPS fixes", xlabel="time since start (s)",
              ylabel="seconds since previous fix (log)"); ax[0].legend()
    ax[1].plot(*xy.T, color=BLUE, lw=1, label="GPS track")
    for i in big:
        ax[1].plot(*xy[i - 1:i + 1].T, color=RED, lw=2.5)
        ax[1].scatter(*xy[i - 1:i + 1].T, color=RED, s=14, zorder=3)
        if dt[i] >= 20:
            ax[1].annotate(f"{dt[i]:.0f} s", xy[i], fontsize=8, color=RED, xytext=(5, 5), textcoords="offset points")
    ax[1].plot([], [], color=RED, lw=2.5, label="jump across a gap (no fixes)")
    ax[1].set_aspect("equal"); ax[1].legend(fontsize=8)
    ax[1].set(title="Where the gaps happen on the map", xlabel="x east (m)", ylabel="y north (m)")
    fig.tight_layout(); fig.savefig(FIG / "gaps_example_trip.png"); plt.close(fig)


def plot_uncovered():
    _, tr_l = build_partition("train")
    _, te_l = build_partition("student_test")
    labs = pd.concat([tr_l.assign(partition="train"), te_l.assign(partition="student_test")])
    per = labs.groupby(["partition", "trip"]).agg(n=("reliable", "size"), covered=("reliable", "sum")).reset_index()
    per["uncovered"] = per.n - per.covered
    per["share"] = per.uncovered / per.n

    # 1) per-trip uncovered fixes
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.4))
    s = per[per.uncovered > 0].sort_values("uncovered", ascending=False)
    ax[0].bar([str(int(t)) + ("*" if p == "student_test" else "") for t, p in zip(s.trip, s.partition)], s.uncovered,
              color=[ORANGE if p == "student_test" else RED for p in s.partition])
    for i, (u, sh) in enumerate(zip(s.uncovered, s.share)):
        ax[0].text(i, u, f"{sh:.0%}", ha="center", va="bottom", fontsize=7)
    ax[0].set_yscale("log")
    ax[0].set(title=f"GPS fixes NOT covered by the reference route (> {config.LABEL_MAX_DIST_M:.0f} m away)\n"
                    f"{len(s)} of {len(per)} trips affected; label = share of that trip's fixes (* = student_test)",
              xlabel="trip id", ylabel="uncovered fixes (log)")
    ax[0].tick_params(axis="x", rotation=90, labelsize=7)
    tot = per.groupby("partition")[["covered", "uncovered"]].sum().loc[["train", "student_test"]]
    ax[1].barh(tot.index, tot.covered, color=GREEN, label="covered (used for training / scoring)")
    ax[1].barh(tot.index, tot.uncovered, left=tot.covered, color=RED, label="uncovered (excluded)")
    for i, (c, u) in enumerate(zip(tot.covered, tot.uncovered)):
        ax[1].text(c + u, i, f"  {u:,} uncovered ({u/(c+u):.1%})", va="center", fontsize=9)
    ax[1].set(title="All fixes per partition", xlabel="GPS fixes"); ax[1].legend(loc="lower right", fontsize=8)
    ax[1].set_xlim(0, tot.sum(1).max() * 1.35)
    fig.tight_layout(); fig.savefig(FIG / "uncovered_overview.png"); plt.close(fig)

    # 2) map views: two kinds of uncovered fixes
    #    54 / 33: a long stay in a small area after the route ends (likely parked, GPS jitter)
    #    2: real driving on roads that the reference route does not include
    from matplotlib.lines import Line2D
    cases = [("00000054", "long stay in a small area after the route ends"),
             ("00000033", "long stay in one spot after the route ends"),
             ("00000002", "driving on roads missing from the reference route")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    for ax, (tid, why) in zip(axes, cases):
        trip = load_trip(config.TRAIN_DIR / tid)
        tr = trip.track
        proj = LocalProjection.for_points(tr[:, :2])
        net = RoadNetwork(trip.nodes, trip.arcs, proj)
        xy = proj.forward(tr[:, 0], tr[:, 1])
        rel = tr_l[tr_l.trip == tid].sort_values("obs").reliable.to_numpy()
        for a in trip.route:
            ax.plot(*np.c_[net.p0[a], net.p1[a]], color=GREEN, lw=3, alpha=0.8)
        ax.scatter(*xy[rel].T, s=2, color=BLUE)
        ax.scatter(*xy[~rel].T, s=2, color=RED)
        ax.scatter(*xy[0], marker="^", s=70, color="k", zorder=4)
        ax.scatter(*xy[-1], marker="s", s=50, color="k", zorder=4)
        u = np.where(~rel)[0]
        p = xy[~rel]
        spread = np.percentile(np.hypot(*(p - np.median(p, 0)).T), 95)
        hours = (tr[u[-1], 2] - tr[u[0], 2]) / 3600
        if spread < 500:
            c = np.median(p, 0)
            ax.annotate(f"{len(u):,} uncovered fixes\nwithin ~{spread:.0f} m\nover {hours:.1f} h",
                        c, xytext=(30, 40), textcoords="offset points", fontsize=8, color=RED,
                        arrowprops=dict(arrowstyle="->", color=RED))
        handles = [Line2D([], [], color=GREEN, lw=3, label="reference route"),
                   Line2D([], [], marker="o", ls="", color=BLUE, ms=5, label=f"covered fixes ({rel.mean():.0%})"),
                   Line2D([], [], marker="o", ls="", color=RED, ms=5, label=f"uncovered fixes ({(~rel).mean():.0%})"),
                   Line2D([], [], marker="^", ls="", color="k", ms=7, label="start"),
                   Line2D([], [], marker="s", ls="", color="k", ms=6, label="end")]
        ax.legend(handles=handles, fontsize=8, loc="best")
        ax.set_aspect("equal")
        ax.set(title=f"Train trip {tid}: {why}", xlabel="x east (m)", ylabel="y north (m)")
        ax.title.set_fontsize(10)
    fig.suptitle("Why some GPS fixes are not covered by the reference route (> 100 m away)", fontsize=12)
    fig.tight_layout(); fig.savefig(FIG / "uncovered_example_trips.png"); plt.close(fig)

    # 3) interactive folium map for trip 54
    tid = "00000054"
    trip = load_trip(config.TRAIN_DIR / tid)
    rel = tr_l[tr_l.trip == tid].sort_values("obs").reliable.to_numpy()
    tr = trip.track
    m = folium.Map(location=[tr[:, 1].mean(), tr[:, 0].mean()], tiles="cartodbpositron")
    fg_r = folium.FeatureGroup(name="Reference route")
    for a in trip.route:
        f, t = trip.arcs[a]
        folium.PolyLine([[trip.nodes[f, 1], trip.nodes[f, 0]], [trip.nodes[t, 1], trip.nodes[t, 0]]],
                        color=GREEN, weight=6, opacity=0.8).add_to(fg_r)
    fg_c = folium.FeatureGroup(name="GPS fixes covered by route")
    fg_u = folium.FeatureGroup(name="GPS fixes NOT covered (excluded)")
    for i in range(0, len(tr), 10):
        folium.CircleMarker([tr[i, 1], tr[i, 0]], radius=2, color=BLUE if rel[i] else RED, fill=True,
                            fill_opacity=0.9, tooltip=f"fix {i}").add_to(fg_c if rel[i] else fg_u)
    for fg in (fg_r, fg_c, fg_u):
        fg.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.fit_bounds([[tr[:, 1].min(), tr[:, 0].min()], [tr[:, 1].max(), tr[:, 0].max()]])
    m.save(config.MAP_DIR / f"uncovered_train_{tid}.html")
    return per


def main():
    g = gap_table()
    g.to_csv(config.RESULTS_DIR / "data_gaps_per_trip.csv", index=False)
    plot_gaps(g)
    per = plot_uncovered()
    per.to_csv(config.RESULTS_DIR / "uncovered_fixes_per_trip.csv", index=False)
    print("wrote gaps_*.png, uncovered_*.png and uncovered_train_00000054.html")


if __name__ == "__main__":
    main()
