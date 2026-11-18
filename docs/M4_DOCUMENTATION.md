# PathSnap Module 4 (M4) Technical Documentation
**Gradient Boosted Classifier (LightGBM), Bayesian Optimization, HMM/Viterbi Route Reconstruction, FastAPI Backend & Evaluator Web UI**

---

## 1. Role & Problem Formulation (Member 4)

### 1.1 Objective
Member 4 owns the end-to-end integration and delivery of PathSnap, spanning:
1. **Model 4 (LightGBM):** Design, training, and diagnostics of the gradient boosting candidate classifier.
2. **Bayesian Hyperparameter Optimization:** GroupKFold-aware tuning using Optuna's Tree-structured Parzen Estimator (TPE).
3. **Hidden Markov Model (HMM) & Viterbi Decoding:** Sequence-level map-matching enforcing physical road network connectivity.
4. **FastAPI Backend (Laptop 1):** Low-latency REST API executing real-time inference and server-side ground-truth evaluation.
5. **Evaluator Interface (Laptop 2):** Interactive Leaflet-based web UI displaying trajectories, road network layers, and live metrics.
6. **Technical Report & Visualizations:** Automated generation of the consolidated technical report and Folium geographic maps.

---

## 2. Model 4: LightGBM (Gradient Boosted Trees)

### 2.1 Theoretical Framework
Unlike bagging methods (e.g. Random Forest) that train independent trees in parallel, Gradient Boosting builds decision trees **sequentially**. Each subsequent tree fits the negative gradient (residual) of the loss function with respect to the current ensemble output:
$$F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \eta \cdot h_m(\mathbf{x})$$
where $\eta = 0.05$ is the shrinkage rate (learning rate) and $h_m$ is a regression tree fitted to the pseudo-residuals:
$$r_{im} = -\left[ \frac{\partial \mathcal{L}(y_i, F(\mathbf{x}_i))}{\partial F(\mathbf{x}_i)} \right]_{F=F_{m-1}}$$
The objective is binary cross-entropy (log loss), optimized using second-order Taylor approximations.

### 2.2 Efficiency Innovations
1. **Histogram-based Binning:** Continuous features are discretized into 256 discrete integer bins. Split finding scales with $O(\#\text{bins})$ instead of $O(\#\text{samples} \log \#\text{samples})$, drastically accelerating training on ~440,000 candidate rows.
2. **Leaf-wise (Best-first) Growth:** Trees grow by splitting the leaf that achieves the maximal loss reduction, yielding deeper, more asymmetric trees that minimize log loss faster than level-wise architectures.

### 2.3 Hyperparameter Specification (`pathsnap.models`)
```python
lgb.LGBMClassifier(
    n_estimators=400,        # 400 boosting stages
    learning_rate=0.05,      # Shrinkage factor
    num_leaves=63,           # Maximum leaves per tree (depth ~6)
    subsample=0.8,           # Stochastic row subsampling (bagging fraction)
    subsample_freq=1,        # Subsample every iteration
    colsample_bytree=0.8,    # Feature fraction per tree
    random_state=42,         # Reproducibility
    n_jobs=-1,
    verbose=-1
)
```

### 2.4 Overfitting Diagnostics
In empirical training histories (`results/figures/cmp_training_histories.png`), the held-out validation log loss reaches a global minimum at **round 130** ($0.0666$) before exhibiting a slight upward trajectory, while training loss monotonically decreases. This mild overfitting profile directly motivated Bayesian hyperparameter tuning.

---

## 3. Bayesian Hyperparameter Optimization

### 3.1 Optimization Design
* **Search Algorithm:** Optuna's Tree-structured Parzen Estimator (TPE), modeling $p(\mathbf{\theta} \mid y)$ to prioritize high-potential parameter configurations.
* **Objective Function:** Mean 5-fold `GroupKFold` Macro F1 across the 80 training trips.
* **Trials:** 40 trials, with a median pruner terminating underperforming parameter sets early.
* **Search Space:**
  - `n_estimators`: $[100, 600]$
  - `learning_rate`: $[0.01, 0.2]$ (log-uniform)
  - `num_leaves`: $[15, 127]$
  - `min_child_samples`: $[5, 100]$
  - `subsample`: $[0.5, 1.0]$
  - `colsample_bytree`: $[0.5, 1.0]$
  - `reg_alpha` & `reg_lambda`: $[10^{-8}, 10.0]$ (log-uniform)

---

## 4. Sequential Route Reconstruction (HMM & Viterbi)

### 4.1 Motivation
Independent candidate classification evaluates each GPS observation in isolation, frequently resulting in non-adjacent road segments or illegal direction reversals. The HMM enforces global sequence consistency and topological connectivity.

### 4.2 State, Emission, and Transition Formulation
* **Hidden States ($S_i$):** Top-$K$ candidate road segments ($K=8$) within 50 m of GPS fix $i$.
* **Emission Probability:** Derived directly from the calibrated LightGBM candidate probability:
  $$\log P_{\text{emit}}(z_i \mid s_{i,k}) = \alpha \cdot \log p_{i,k}, \quad \alpha = 1.0$$
* **Transition Probability:** Penalizes discrepancy between network routing distance $d_{\text{net}}$ and Euclidean great-circle distance $d_{\text{gc}}$:
  $$\log P_{\text{trans}}(s_{i,k} \to s_{i+1,l}) = -\frac{|d_{\text{net}}(s_{i,k}, s_{i+1,l}) - d_{\text{gc}}(z_i, z_{i+1})|}{\beta \cdot \sqrt{\Delta t}}, \quad \beta = 2.0$$
  where $d_{\text{net}}$ is computed via bounded Dijkstra search on the road network graph.
* **Gap & Break Handling:** If $\Delta t > 120\text{ s}$ or if no candidate is topologically reachable within bounding distance limits, the chain resets to prevent unphysical interpolation.
* **Decoding:** Viterbi dynamic programming determines the globally optimal sequence of arcs with $O(T \cdot K^2)$ complexity.

---

## 5. Two-Laptop Distributed Deployment Architecture

### 5.1 Architecture Overview
```
[ Laptop 2: Web UI (Browser) ]
         │
         │ HTTP POST /predict {"trip_id": "00000015"}
         ▼
[ Laptop 1: FastAPI Backend (:8000) ]
   ├── PathSnapMatcher (Preprocess -> LightGBM -> HMM/Viterbi)
   ├── Evaluator Scorer (Server-side accuracy against .route)
   └── JSON Response (GPS points, predicted polylines, aggregate metrics)
```

### 5.2 API Specification (`api/main.py`)
1. **`GET /health`:** Health check returning model identifier and active evaluation directory.
2. **`GET /trips`:** Lists all 10 evaluation trips with observation counts.
3. **`POST /predict`:** Accepts `{"trip_id": str}`, executes inference, calculates server-side evaluation metrics, and returns map polylines.

### 5.3 Data Leakage Safeguards
* **Ground-Truth Isolation:** `predict_trip` loads data using `with_route=False`. The prediction model never receives ground-truth `.route` arcs.
* **Server-Side Scoring:** Evaluation against reference routes occurs strictly after prediction in `evaluate_trip`.
* **Client Boundaries:** The client UI receives only aggregate accuracy metrics and predicted geometry; no reference route geometry is transmitted.
* **Live Logging:** Real-time requests, fix counts, accuracy, and latency are printed live to the Laptop 1 console.

---

## 6. Empirical Benchmark Results

### 6.1 5-Model Comparison (`student_test` - 10 Trips)

| Model | Family | CV Macro F1 | Student-Test Top-1 | Macro F1 | Log Loss | Route Continuity | Inference Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Linear | 0.9315 ± 0.014 | 0.8894 | 0.9315 | 0.0838 | 0.939 | 0.2 ms / 1k |
| **KNN** | Instance | 0.9269 ± 0.014 | 0.8805 | 0.9269 | 0.1032 | 0.914 | 143.2 ms / 1k |
| **Random Forest** | Bagging | **0.9375** ± 0.013 | 0.8930 | 0.9360 | 0.0705 | **0.945** | 9.4 ms / 1k |
| **LightGBM ✅** | Boosting | 0.9367 ± 0.014 | **0.9033** | **0.9404** | **0.0666** | 0.944 | 16.5 ms / 1k |
| **MLP** | Neural Net | 0.9320 ± 0.014 | 0.8931 | 0.9312 | 0.0761 | 0.944 | 0.8 ms / 1k |

### 6.2 Official Evaluator Test Set Benchmark (`evaluator_test` - 10 Trips, 19,674 Points)

| Metric | Classifier Only | **HMM / Viterbi Reconstructed** | Relative Gain |
| :--- | :---: | :---: | :---: |
| **Top-1 Accuracy (Weighted)** | 90.06% | **`93.79%`** | **+3.73%** |
| **Route F1 (Length-weighted)** | 88.20% | **`97.80%`** | **+9.60%** |
| **Route Continuity** | 94.40% | **`100.0%` (1.000)** | **Zero Disconnected Breaks** |
| **Mean Request Latency** | — | **~1.1 s** | Real-time capable |

---

## 7. Verification and Reproducibility

Execute the Member 4 automated test suite:
```bash
python tests/test_m4.py
```

Launch the distributed deployment:
```bash
# Laptop 1 (API Server)
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

# Laptop 2 (Web UI)
cd ui && python -m http.server 5500
# Browser: http://localhost:5500/?api=http://<LAPTOP1_IP>:8000
```
