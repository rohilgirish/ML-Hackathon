# PathSnap — GPS Map-Matching and Route Reconstruction

PathSnap is an end-to-end machine learning system for map-matching noisy vehicle GPS trajectories onto directed road networks. It matches GPS observations to candidate road segments using a learned candidate classifier and reconstructs physically drivable trajectories using Hidden Markov Model (HMM) sequence decoding with Viterbi optimization.

```
.track/.nodes/.arcs ──► clean + project ──► candidate arcs (≤50 m) ──► 24 features
      ──► Candidate Classifier P(arc is correct) ──► HMM/Viterbi over network distances ──► connected route
```

---

## 1. Team Responsibilities & Work Split

The project is structured into four distinct, self-contained modules:

| Module | Focus Area | Key Files | Status |
|---|---|---|---|
| **M1** | **Data Loading, Geo Projections, Label Alignment & Feature Engineering** | `pathsnap/{data_io,geo,network,labeling,features,dataset,config}.py`, `scripts/eda.py`, `docs/M1_DOCUMENTATION.md` | **Integrated & Validated** |
| **M2** | Model Experimentation & Comparison (8 Models) | `pathsnap/models.py`, `pathsnap/metrics.py`, `scripts/train_models.py` | Teammate assigned |
| **M3** | Bayesian Optimization & HMM/Viterbi Decoder | `pathsnap/hmm.py`, `scripts/bayes_opt.py`, `scripts/hmm_eval.py` | Teammate assigned |
| **M4** | API, Interactive UI & Technical Report | `api/main.py`, `ui/`, `report/` | Teammate assigned |

Detailed work breakdown and presentation preparation are documented in [split.md](file:///C:/Users/Samrudh0045/Studies/Sem5/ML/ML%20Hackathon/split.md) and [evaluation_split.md](file:///C:/Users/Samrudh0045/Studies/Sem5/ML/ML%20Hackathon/evaluation_split.md).

---

## 2. Module 1 (M1) Core Components

Module 1 establishes the mathematical and data foundation for all subsequent modeling:

1. **Data I/O & Cleaning (`pathsnap/data_io.py`):** Loads `.track`, `.nodes`, `.arcs`, and `.route` files; filters out-of-bounds coordinates, stably sorts timestamps, and drops duplicates. Supports `with_route=False` to strictly prevent ground-truth leakage during evaluation.
2. **Local Metric Projection (`pathsnap/geo.py`):** Fast vectorised local equirectangular projection centered at trip medians, ensuring $<0.05\%$ scale distortion with zero external GIS runtime overhead.
3. **Road Network & Graph Topology (`pathsnap/network.py`):** Directed graph representation in Compressed Sparse Row (CSR) format, `shapely.STRtree` R-tree spatial indexing, bidirectional road detection, and bounded Dijkstra shortest-path algorithms.
4. **Monotone DP Label Generation (`pathsnap/labeling.py`):** Dynamic programming algorithm aligning GPS fixes to route arcs with guaranteed forward-in-time monotonicity ($O(N \cdot M)$ complexity). Identifies and filters unreliable fixes ($>100\text{ m}$).
5. **Candidate Generation & Spatial Filtering (`pathsnap/features.py`):** 50 m spatial radius query (up to 12 candidates) with automatic nearest-arc fallback, achieving 98.6% training and 99.88% student-test coverage.
6. **24 Engineered Features (`pathsnap/features.py`):**
   - **Distance (5):** `dist`, `dist_rank`, `dist_minus_min`, `dist_ratio`, `gauss_share`
   - **Heading & Kinematics (4):** `heading_cos`, `heading_absdiff`, `speed`, `heading_valid`
   - **Geometry & Projection (5):** `arc_length`, `proj_t`, `proj_inside`, `n_cands`, `n_within_20`
   - **Temporal Context (5):** `ctx_mean_dist`, `ctx_max_dist`, `ctx_rank`, `ctx_minus_min`, `ctx_heading_cos`
   - **Topology & Gaps (5):** `from_in_deg`, `to_out_deg`, `has_reverse`, `dt_prev`, `dt_next`
7. **Dataset Construction (`pathsnap/dataset.py`):** Multi-threaded parquet generation with caching and GroupKFold cross-validation splits.
8. **Exploratory Data Analysis (`scripts/eda.py`):** Comprehensive dataset inventory, quality analysis, figure generation, and interactive Folium maps.

For in-depth mathematical formulations and empirical findings, see [docs/M1_DOCUMENTATION.md](file:///C:/Users/Samrudh0045/Studies/Sem5/ML/ML%20Hackathon/docs/M1_DOCUMENTATION.md).

---

## 3. Directory Layout

```
.
├── pathsnap/                      # Core Library
│   ├── __init__.py                # Package exports
│   ├── config.py                  # Paths, candidate constants, label thresholds
│   ├── data_io.py                 # File I/O, Trip dataclass, track cleaning
│   ├── dataset.py                 # Candidate-level dataset construction & caching
│   ├── features.py                # Kinematics, candidate generation & 24 features
│   ├── geo.py                     # Local equirectangular projection & haversine
│   ├── labeling.py                # Monotone DP label alignment & reliability cut
│   └── network.py                 # RoadNetwork, spatial R-tree, Dijkstra shortest paths
├── scripts/
│   └── eda.py                     # Full exploratory data analysis & Folium map generator
├── tests/
│   └── test_m1.py                 # 9-stage validation test suite for M1
├── results/                       # Verified M1 outputs
│   ├── eda_inventory.csv          # Per-trip metrics across train and student-test
│   ├── eda_summary.json           # Aggregated statistics and data-quality metrics
│   ├── figures/                   # EDA figures (sampling, noise, trajectories, etc.)
│   └── maps/                      # Interactive Folium maps (HTML)
├── docs/
│   └── M1_DOCUMENTATION.md        # Comprehensive technical report for M1
├── split.md                       # 4-member hackathon task split
├── evaluation_split.md            # Viva presentation guide & questions
├── requirements.txt               # Dependency specifications
└── .gitignore                     # Git ignore rules (prevents tracking raw data & caches)
```

---

## 4. Setup & Quickstart

### 4.1 Prerequisites
- Python 3.10+ (tested on Python 3.11 and 3.14).
- Dependencies listed in `requirements.txt`.

### 4.2 Installation
```bash
# Clone or navigate to the repository
cd "C:\Users\Samrudh0045\Studies\Sem5\ML\ML Hackathon"

# Install dependencies
pip install -r requirements.txt
```

### 4.3 Dataset Placement
Raw data is distributed separately and must not be committed to Git. The dataset should be extracted to:
```
Student folder/
├── train/          # 80 training trips (00000001, ...)
└── student_test/   # 10 student-testing trips (00000000, ...)
```
`pathsnap/config.py` automatically checks both `Student folder/` in the workspace root and `../Student folder/`, or can be overridden via `PATHSNAP_DATA_DIR`.

---

## 5. Verification & Testing

To verify the integrity of the integrated M1 contribution, execute the validation test suite:

```bash
python tests/test_m1.py
```

All 9 test stages will execute and validate:
- Package imports and 24-feature schema.
- Data loading across 80 training and 10 student-test trips.
- Local projection accuracy (roundtrip error $<10^{-9}$ degrees).
- Road network geometry, spatial index, and Dijkstra paths.
- Monotone DP route alignment and reliability cutoffs.
- Candidate generation and outlier fallback bounds.
- All 24 engineered features with zero NaN and zero infinite values.
- Parquet table construction and candidate hit rates ($>98\%$).
- Student-test partition compatibility and ground-truth isolation.

To run the exploratory data analysis script:
```bash
python scripts/eda.py
```
This updates the inventory CSV, summary JSON, figures in `results/figures/`, and interactive maps in `results/maps/`.
