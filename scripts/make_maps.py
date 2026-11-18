"""Folium maps of the final pipeline on student-testing trips: raw GPS, road network,
reference route, classifier-only and HMM-predicted routes, with layer control and legend."""
import sys
from pathlib import Path

import folium
import shapely
from branca.element import MacroElement, Template

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathsnap import config  # noqa: E402
from pathsnap.data_io import list_trips  # noqa: E402
from pathsnap.pipeline import PathSnapMatcher, arcs_to_lonlat, evaluate_trip  # noqa: E402

LEGEND = """{% macro html(this, kwargs) %}
<div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; background: white; padding: 8px 12px;
 border-radius: 6px; box-shadow: 0 1px 4px rgba(0,0,0,.3); font: 12px sans-serif; line-height: 1.7">
<b>{{this.title}}</b><br>
<span style="display:inline-block;width:18px;height:4px;background:#9aa0a6"></span> road network<br>
<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#1565c0;margin:0 5px"></span> raw GPS<br>
<span style="display:inline-block;width:18px;height:4px;background:#2e7d32"></span> reference route<br>
<span style="display:inline-block;width:18px;height:4px;background:#8e44ad"></span> classifier only<br>
<span style="display:inline-block;width:18px;height:4px;background:#e4572e"></span> predicted (HMM)
</div>{% endmacro %}"""


def trip_map(matcher, part_dir, tid, out, zoom_on_disagreement=False):
    res = matcher.predict_trip(part_dir / tid)
    ev = evaluate_trip(res, part_dir / tid)
    pt = res["pt"]
    lat, lon = pt.lonlat[:, 1], pt.lonlat[:, 0]
    m = folium.Map(location=[lat.mean(), lon.mean()], tiles="cartodbpositron", zoom_start=14)
    near = pt.net.tree.query(shapely.buffer(shapely.linestrings(pt.xy), 150), predicate="intersects")
    groups = {
        "Road network": (arcs_to_lonlat(pt, near), dict(color="#9aa0a6", weight=1.5, opacity=0.7)),
        "Reference route": (arcs_to_lonlat(pt, ev["ref_route"]), dict(color="#2e7d32", weight=7, opacity=0.6)),
        "Classifier only": (arcs_to_lonlat(pt, [a for a in res["clf_arc"] if a >= 0]), dict(color="#8e44ad", weight=3, dash_array="4 6")),
        "Predicted route (HMM)": (arcs_to_lonlat(pt, res["route"]), dict(color="#e4572e", weight=3.5)),
    }
    for name, (lines, style) in groups.items():
        fg = folium.FeatureGroup(name=name, show=zoom_on_disagreement or name != "Classifier only")
        for ln in lines:
            folium.PolyLine(ln, **style).add_to(fg)
        fg.add_to(m)
    fg = folium.FeatureGroup(name="Raw GPS fixes")
    for i in range(0, len(lat), 2):
        folium.CircleMarker([lat[i], lon[i]], radius=2, color="#1565c0", fill=True, fill_opacity=0.9,
                            tooltip=f"fix {i}").add_to(fg)
    fg.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    leg = MacroElement()
    leg._template = Template(LEGEND)
    leg.title = f"Trip {tid}: HMM acc {ev['hmm_top1']:.3f} | clf acc {ev['clf_top1']:.3f} | route F1 {ev['route_f1']:.3f}"
    m.get_root().add_child(leg)
    if zoom_on_disagreement:
        # centre on the 60-fix window where classifier-only and HMM disagree most
        import numpy as np
        dis = (res["clf_arc"] != res["hmm_arc"]).astype(float)
        c = int(np.argmax(np.convolve(dis, np.ones(60), mode="same")))
        lo, hi = max(c - 30, 0), min(c + 30, len(lat) - 1)
        m.fit_bounds([[lat[lo:hi].min(), lon[lo:hi].min()], [lat[lo:hi].max(), lon[lo:hi].max()]])
    else:
        m.fit_bounds([[lat.min(), lon.min()], [lat.max(), lon.max()]])
    m.save(out)
    print(tid, f"hmm={ev['hmm_top1']:.4f} clf={ev['clf_top1']:.4f} routeF1={ev['route_f1']:.4f}")


def main():
    config.MAP_DIR.mkdir(parents=True, exist_ok=True)
    matcher = PathSnapMatcher()
    for tid in list_trips(config.STUDENT_TEST_DIR):
        trip_map(matcher, config.STUDENT_TEST_DIR, tid, config.MAP_DIR / f"student_test_{tid}.html")
    trip_map(matcher, config.STUDENT_TEST_DIR, "00000056", config.MAP_DIR / "student_test_00000056_zoom_clf_vs_hmm.html",
             zoom_on_disagreement=True)


if __name__ == "__main__":
    main()
