# PathSnap Module 3 (M3) Technical Documentation
**Candidate Road Segment Classifiers: Random Forest & Multi-Layer Perceptron (MLP)**

---

## 1. Role & Problem Formulation (Member 3)

### 1.1 Objective
Member 3 is responsible for the design, training, validation, and benchmarking of two fundamental, contrasting machine learning paradigms for map-matching candidate classification:
1. **Model 3: Random Forest Classifier** (Tree-based Bagging Ensemble)
2. **Model 5: Multi-Layer Perceptron (MLP)** (Deep Connectionist Feedforward Neural Network)

### 1.2 Mathematical Formulation
For each GPS observation $i \in \{1, \dots, N\}$, spatial indexing generates a candidate set $\mathcal{C}_i = \{a_{i,1}, \dots, a_{i, K_i}\}$ of road arcs within a $50\text{ m}$ search radius ($K_i \le 12$).
Each `(fix, arc)` pair is parameterized by a 24-dimensional feature vector $\mathbf{x} \in \mathbb{R}^{24}$.

The binary classification objective is:
$$f: \mathbb{R}^{24} \longrightarrow [0, 1], \quad f(\mathbf{x}) = \hat{P}(y = 1 \mid \mathbf{x})$$
where:
* $y = 1$: The candidate arc is the ground-truth segment traversed by the vehicle ($a = a^*$).
* $y = 0$: The candidate arc is a distractor (nearby parallel street, perpendicular crossroad, or opposite-direction lane).

For each GPS fix $i$, the final reconstructed road segment is chosen via the Top-1 decision rule:
$$\hat{a}_i^* = \arg\max_{a \in \mathcal{C}_i} f(\mathbf{x}_{i, a})$$

---

## 2. Model 1: Random Forest Classifier

### 2.1 Theoretical Framework
A single decision tree exhibits low bias but high variance, easily memorizing sensor noise and GPS multipath reflections. 
Random Forest mitigates variance through **Bootstrap Aggregation (Bagging)** and **Random Subspace Projection**:
$$\hat{P}_{\text{RF}}(y=1 \mid \mathbf{x}) = \frac{1}{B} \sum_{b=1}^B T_b(\mathbf{x})$$
where $B = 150$ de-correlated decision trees.

### 2.2 Hyperparameter Specification (`pathsnap.models`)
```python
RandomForestClassifier(
    n_estimators=150,       # 150 independent decision trees
    min_samples_leaf=5,     # Regularization: >= 5 samples per leaf
    max_features="sqrt",    # Evaluate sqrt(24) ≈ 5 random features at each split
    n_jobs=-1,              # Full multi-core CPU parallelization
    random_state=42         # Deterministic reproducibility
)
```

### 2.3 Hyperparameter Justification
* `n_estimators=150`: Maximizes variance reduction while keeping inference within acceptable limits ($9.4\text{ ms}$ per 1k fixes).
* `min_samples_leaf=5`: Suppresses overfitting to outlier GPS multipath spikes by forbidding single-sample leaf partitions.
* `max_features='sqrt'`: Decorrelates individual trees by forcing splits across non-dominant features (e.g. heading, context) rather than solely relying on `dist`.

---

## 3. Model 2: Multi-Layer Perceptron (MLP)

### 3.1 Architecture & Pipeline
MLP provides continuous, non-linear function approximation through dense layered transformations:
$$\mathbf{h}_1 = \text{ReLU}(W_1 \tilde{\mathbf{x}} + \mathbf{b}_1), \quad \mathbf{h}_2 = \text{ReLU}(W_2 \mathbf{h}_1 + \mathbf{b}_2), \quad \hat{y} = \sigma(\mathbf{w}_3^T \mathbf{h}_2 + b_3)$$
where $\tilde{\mathbf{x}} = \text{StandardScaler}(\mathbf{x}) = \frac{\mathbf{x} - \boldsymbol{\mu}}{\boldsymbol{\sigma}}$.

```
Input (24 Features) -> StandardScaler -> Dense(64, ReLU) -> Dense(32, ReLU) -> Dense(1, Sigmoid)
```

### 3.2 Hyperparameter Specification (`pathsnap.models`)
```python
make_pipeline(
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=(64, 32), # 2-layer funnel architecture
        alpha=1e-4,                  # L2 weight regularization penalty
        batch_size=512,              # Mini-batch gradient descent
        learning_rate_init=1e-3,     # Initial Adam learning rate
        max_iter=60,                 # Maximum training epochs
        early_stopping=True,         # Validation-guided early termination
        validation_fraction=0.1,     # 10% stratified internal validation split
        n_iter_no_change=6,          # Patience of 6 epochs
        random_state=42
    )
)
```

### 3.3 Training Optimization & Early Stopping
* **StandardScaler Preprocessing:** Unscaled input features span wildly different magnitudes (`dist`: $0\text{--}50\text{ m}$ vs `heading_cos`: $[-1, +1]$). Scaling to zero mean and unit variance prevents gradient explosion and ensures isotropic parameter updates.
* **Adam Optimizer:** Uses adaptive first and second moment estimations ($\beta_1 = 0.9, \beta_2 = 0.999$) for fast, stable convergence.
* **Early Stopping:** Training terminated at **Epoch 37** when internal validation accuracy plateaued at $97.41\%$ with training loss declining from $0.1129$ to $0.0626$.

---

## 4. Empirical Evaluation Protocol

1. **Cross-Validation:** 5-fold `GroupKFold` over the 80 training trips ($442,498$ training samples across $62,648$ GPS observations). Trips are strictly held out to prevent autocorrelation leakage.
2. **Student Test Benchmark:** Evaluated out-of-sample on the 10 student-testing trips ($147,062$ candidate rows across $19,160$ GPS observations).
3. **Blind Evaluator Test Benchmark:** Evaluated on the 10 official evaluator-test trips ($146,691$ candidate rows across $19,674$ GPS observations).

---

## 5. Experimental Results & Benchmark Tables

### 5.1 Student-Test Benchmark (`results/model_comparison.csv`)

| Metric | Random Forest (Bagging) | MLP (Neural Network) | Analysis |
| :--- | :---: | :---: | :--- |
| **CV Macro F1 (Mean ± SD)** | **$0.9375 \pm 0.0133$** | $0.9320 \pm 0.0135$ | Both highly stable across folds |
| **CV Top-1 Accuracy** | **$88.86\%$** | $88.39\%$ | Robust cross-validation generalization |
| **CV Log Loss** | **$0.0790$** | $0.0858$ | RF provides sharper probabilities |
| **Train Macro F1** | $0.9723$ | $0.9489$ | Mild RF fit; minimal MLP gap ($0.018$) |
| **Student-Test Top-1** | $89.30\%$ | **$89.31\%$** | **MLP slightly leads (+0.01%)** |
| **Student-Test Macro F1** | **$0.9360$** | $0.9312$ | RF maintains superior precision |
| **Student-Test Log Loss** | **$0.0705$** | $0.0761$ | RF has lower cross-entropy error |
| **Route Continuity** | **$94.48\%$** | $94.38\%$ | Over 94.4% sequential continuity |
| **Inference Latency** | $9.45\text{ ms}$ / 1k fixes | **$0.81\text{ ms}$ / 1k fixes** | **MLP is $11.6\times$ faster** |

---

### 5.2 Official Blind Evaluator-Test Benchmark (`results/evaluator_test_comparison.csv`)

| Metric | Random Forest | MLP | Analysis on Unseen Data |
| :--- | :---: | :---: | :--- |
| **Top-1 Accuracy** | **$90.07\%$** | $88.93\%$ | **Passed 90% accuracy milestone!** |
| **Macro F1** | **$0.9384$** | $0.9315$ | High precision and recall |
| **Precision** | **$95.32\%$** | $93.27\%$ | Filters out false side streets |
| **Recall** | $92.49\%$ | **$93.02\%$** | High coverage of true road arcs |
| **Log Loss** | **$0.0748$** | $0.0796$ | Sharp candidate calibration |
| **Route Continuity** | $94.61\%$ | **$95.12\%$** | Continuous topological paths |
| **Inference Latency** | $21.41\text{ ms}$ / 1k fixes | **$1.68\text{ ms}$ / 1k fixes** | **MLP is $12.7\times$ faster** |

---

## 6. Generated Visualizations & Artifacts

All figures are compiled in `results/figures/`:
1. `cmp_test_top1_f1.png`: Out-of-sample Top-1 Accuracy and Macro F1 comparison.
2. `cmp_cv_macro_f1.png`: 5-fold cross-validation Macro F1 with error bars ($\mu \pm \sigma$).
3. `cmp_cv_folds_box.png`: Fold-by-fold stability boxplot confirming stability across diverse trip complexities.
4. `cmp_train_vs_test.png`: Generalization gap diagnostics (Train vs Test Macro F1).
5. `cmp_logloss_inference.png`: Calibration quality vs Production inference latency trade-off.
6. `cmp_confusion_matrices.png`: Candidate-level confusion matrices on 147k test rows ($>98\%$ specificity).
7. `cmp_training_histories.png`: MLP Adam loss curve and early stopping at Epoch 37.
8. `cmp_actual_vs_predicted_path.png`: Full trajectory and zoomed intersection comparison on student-test.
9. `evaluator_test_actual_vs_predicted.png`: Map-matching trajectory verification on the blind evaluator test set.

---

## 7. Verification and Reproducibility

Execute the full Member 3 test suite:
```bash
python tests/test_m3.py
```

Re-train and evaluate both models live:
```bash
# Train on 80 trips and evaluate on student_test
python scripts/train_models.py

# Evaluate on the official evaluator_test partition
python scripts/eval_evaluator_test.py

# Regenerate all visual figures
python scripts/make_figures.py
```
