# PathSnap — GPS Map-Matching and Route Reconstruction

PathSnap is an end-to-end machine learning system for map-matching noisy vehicle GPS trajectories onto directed road networks. It matches GPS observations to candidate road segments using a learned candidate classifier and reconstructs physically drivable trajectories using Hidden Markov Model (HMM) sequence decoding with Viterbi optimization.

```
.track/.nodes/.arcs ──► clean + project ──► candidate arcs (≤50 m) ──► 24 features
      ──► Candidate Classifier P(arc is correct) ──► HMM/Viterbi over network distances ──► connected route
```

---

## 1. Team Responsibilities & Work Split

The project is structured into four distinct, self-contained modules according to `evaluation_split.md`:

| Module | Focus Area | Key Files | Status |
|---|---|---|---|
| **M1** | **Data Loading, Geo Projections, Label Alignment & Feature Engineering** | `pathsnap/{data_io,geo,network,labeling,features,dataset,config}.py`, `scripts/eda.py`, `docs/M1_DOCUMENTATION.md` | **Integrated & Validated** |
| **M2** | Logistic Regression & KNN Baselines | `evaluation_split.md` | Teammate assigned |
| **M3** | **Candidate Classifiers: Random Forest & MLP, Evaluator-Test & Figures** | `pathsnap/models.py`, `scripts/{train_models,eval_evaluator_test,make_figures}.py`, `docs/M3_DOCUMENTATION.md` | **Integrated & Validated** |
| **M4** | LightGBM Selection & Final System Pipeline | `evaluation_split.md` | Teammate assigned |

Detailed work breakdown and presentation preparation are documented in [docs/M1_DOCUMENTATION.md](file:///C:/Users/Lenovo/Desktop/ML%20hackathon/docs/M1_DOCUMENTATION.md), [docs/M3_DOCUMENTATION.md](file:///C:/Users/Lenovo/Desktop/ML%20hackathon/docs/M3_DOCUMENTATION.md), and [evaluation_split.md](file:///C:/Users/Lenovo/Desktop/ML%20hackathon/evaluation_split.md).

---

## 2. Module 1 (M1) Core Components

1. **Data I/O & Cleaning (`pathsnap/data_io.py`):** Loads `.track`, `.nodes`, `.arcs`, and `.route` files; filters out-of-bounds coordinates, stably sorts timestamps, and drops duplicates. Supports `with_route=False` to strictly prevent ground-truth leakage during evaluation.
2. **Local Metric Projection (`pathsnap/geo.py`):** Fast vectorised local equirectangular projection centered at trip medians, ensuring $<0.05\%$ scale distortion with zero external GIS runtime overhead.
3. **Road Network & Graph Topology (`pathsnap/network.py`):** Directed graph representation in Compressed Sparse Row (CSR) format, `shapely.STRtree` R-tree spatial indexing, bidirectional road detection, and bounded Dijkstra shortest-path algorithms.
4. **Monotone DP Label Generation (`pathsnap/labeling.py`):** Dynamic programming algorithm aligning GPS fixes to route arcs with guaranteed forward-in-time monotonicity.
5. **Candidate Generation & Spatial Filtering (`pathsnap/features.py`):** 50 m spatial radius query (up to 12 candidates) with automatic nearest-arc fallback.
6. **24 Engineered Features (`pathsnap/features.py`):** Distance (5), Heading & Kinematics (4), Geometry & Projection (5), Temporal Context (5), Topology & Gaps (5).
7. **Exploratory Data Analysis (`scripts/eda.py`):** Comprehensive dataset inventory, quality analysis, figure generation, and interactive Folium maps.

---

## 3. Module 3 (M3) Core Components

Module 3 delivers the candidate road segment classification models:

1. **Random Forest Classifier (`pathsnap/models.py`):** 
   - 150 bagged decision trees, `min_samples_leaf=5`, `max_features="sqrt"`.
   - Highest test precision ($95.32\%$) and lowest log loss ($0.0705$).
2. **Multi-Layer Perceptron (MLP) (`pathsnap/models.py`):**
   - 2-layer funnel architecture ($24 \to 64 \to 32 \to 1$), `StandardScaler`, ReLU activation, Adam optimizer with early stopping at Epoch 37.
   - Ultra-fast real-time inference: **$0.81\text{ ms}$ per 1,000 fixes ($11.6\times$ faster than Random Forest)**.
3. **Evaluation on Unseen Benchmarks:**
   - **Student-Test (10 trips):** Random Forest Top-1 ($89.30\%$), MLP Top-1 ($89.31\%$).
   - **Official Evaluator-Test (10 trips):** Random Forest Top-1 (**$90.07\%$**), MLP Top-1 ($88.93\%$).
4. **Diagnostic & Route Visualizations (`scripts/make_figures.py`):**
   - 5-fold CV stability boxplots, train vs. test generalization gaps, candidate confusion matrices, MLP epoch loss curves, and GIS map-matching trajectory plots (`cmp_actual_vs_predicted_path.png` and `evaluator_test_actual_vs_predicted.png`).

---

## 4. Benchmark Summary

| Model | Family | Student-Test Top-1 | Student-Test Macro F1 | Evaluator-Test Top-1 | Evaluator-Test Macro F1 | Inference Latency |
|---|---|---|---|---|---|---|
| **Random Forest** | Ensemble (bagging) | 0.8930 | **0.9360** | **0.9007** | **0.9384** | 9.4 ms / 1k |
| **MLP** | Neural network | **0.8931** | 0.9312 | 0.8893 | 0.9315 | **0.81 ms / 1k** |

---

## 5. Verification & Testing

Execute both Member 1 and Member 3 validation test suites:

```bash
# 1. Run M1 data pipeline and feature tests (9 stages)
python tests/test_m1.py

# 2. Run M3 model zoo and benchmark tests (5 stages)
python -m unittest tests/test_m3.py

# 3. Re-train models and evaluate live
python scripts/train_models.py

# 4. Evaluate on the official evaluator_test dataset
python scripts/eval_evaluator_test.py

# 5. Generate all diagnostic and trajectory figures
python scripts/make_figures.py
```
