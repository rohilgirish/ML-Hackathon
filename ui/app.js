// PathSnap evaluator UI (Laptop 2). Talks only to the FastAPI backend on Laptop 1.
const params = new URLSearchParams(location.search);
const DEFAULT_API = params.get("api") || localStorage.getItem("pathsnap_api") || "http://localhost:8000";
const $ = (id) => document.getElementById(id);
$("api-url").value = DEFAULT_API;

const map = L.map("map", { preferCanvas: true }).setView([20, 0], 2);
L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
  attribution: "&copy; OpenStreetMap contributors",
  maxZoom: 19,
}).addTo(map);
let layers = {};
let control = null;

const legend = L.control({ position: "bottomright" });
legend.onAdd = () => {
  const d = L.DomUtil.create("div", "legend");
  d.innerHTML = '<i style="background:#9aa0a6"></i>Road network<br>' +
    '<i class="dot" style="background:#2b6cb0"></i>Raw GPS<br>' +
    '<i style="background:#e4572e"></i>Predicted route (HMM)<br>' +
    '<i style="background:#8e44ad"></i>Classifier only';
  return d;
};
legend.addTo(map);

const api = () => $("api-url").value.replace(/\/+$/, "");

async function connect() {
  const st = $("api-status");
  st.textContent = "connecting…"; st.className = "status";
  try {
    localStorage.setItem("pathsnap_api", api());
    const h = await fetch(api() + "/health").then((r) => r.json());
    const t = await fetch(api() + "/trips").then((r) => r.json());
    st.textContent = "connected · " + h.model; st.className = "status ok";
    const ul = $("trip-list");
    ul.innerHTML = "";
    t.trips.forEach((trip, i) => {
      const li = document.createElement("li");
      const b = document.createElement("button");
      b.innerHTML = `<span>Trip ${i + 1} · ${trip.trip_id}</span><small>${trip.n_points} fixes</small>`;
      b.onclick = () => run(trip.trip_id, b);
      li.appendChild(b); ul.appendChild(li);
      if (params.get("trip") === trip.trip_id) run(trip.trip_id, b);
    });
  } catch (e) {
    st.textContent = "cannot reach API"; st.className = "status bad";
  }
}

const pct = (x) => (x == null ? "–" : (100 * x).toFixed(1) + "%");

async function run(tripId, btn) {
  document.querySelectorAll("#trip-list button").forEach((b) => b.classList.remove("active"));
  btn.classList.add("active");
  $("busy").hidden = false;
  try {
    const r = await fetch(api() + "/predict", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ trip_id: tripId }),
    });
    if (!r.ok) throw new Error(await r.text());
    draw(await r.json());
  } catch (e) {
    alertBox("Prediction failed: " + e.message);
  } finally {
    $("busy").hidden = true;
  }
}

function alertBox(msg) { $("api-status").textContent = msg; $("api-status").className = "status bad"; }

function draw(res) {
  Object.values(layers).forEach((l) => map.removeLayer(l));
  if (control) map.removeControl(control);
  layers = {
    "Road network": L.layerGroup(res.road_network.map((s) => L.polyline(s, { color: "#9aa0a6", weight: 1.5, opacity: 0.7 }))),
    "Classifier only": L.layerGroup(res.classifier_only_route.map((s) => L.polyline(s, { color: "#8e44ad", weight: 3, opacity: 0.8, dashArray: "4 6" }))),
    "Predicted route (HMM)": L.layerGroup(res.predicted_route.map((s) => L.polyline(s, { color: "#e4572e", weight: 4, opacity: 0.95 }))),
    "Raw GPS": L.layerGroup(res.gps.filter((_, i) => i % 2 === 0).map((p) => L.circleMarker(p, { radius: 2, color: "#2b6cb0", weight: 0, fillOpacity: 0.55 }))),
  };
  ["Road network", "Raw GPS", "Predicted route (HMM)"].forEach((k) => layers[k].addTo(map));
  control = L.control.layers(null, layers, { collapsed: false, position: "topright" }).addTo(map);
  map.fitBounds(L.latLngBounds(res.gps), { padding: [30, 30] });

  const m = res.metrics;
  $("metrics").hidden = false;
  $("res-trip").textContent = "· " + res.trip_id;
  $("m-hmm").textContent = pct(m.hmm_top1);
  $("m-clf").textContent = pct(m.clf_top1);
  $("m-f1").textContent = pct(m.route_f1);
  $("m-cont").textContent = pct(m.route_continuity);
  const rows = [
    ["GPS fixes", m.n_obs], ["Fixes scored (reliable labels)", m.n_reliable],
    ["Route precision", pct(m.route_precision)], ["Route recall", pct(m.route_recall)],
    ["Continuity – classifier only", pct(m.clf_continuity)], ["Continuity – HMM per-fix", pct(m.hmm_continuity)],
    ["Candidate miss rate", pct(m.candidate_miss_rate)], ["HMM chain breaks", m.hmm_breaks],
    ["Preprocess + features", res.timings.preprocess_s + " s"], ["Classifier", res.timings.classifier_s + " s"],
    ["HMM / Viterbi", res.timings.hmm_s + " s"], ["Total request", res.timings.request_s + " s"],
  ];
  $("m-table").innerHTML = rows.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");
}

$("api-form").addEventListener("submit", (e) => { e.preventDefault(); connect(); });
connect();
