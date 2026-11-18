"""Candidate classifiers for Member 3 (M3): Random Forest & Multi-Layer Perceptron (MLP)."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .config import SEED

# name -> (family, factory, key configuration, max training rows or None)
MODEL_ZOO = {
    "RandomForest": ("Ensemble (bagging)", lambda: RandomForestClassifier(
        n_estimators=150, min_samples_leaf=5, max_features="sqrt", n_jobs=-1, random_state=SEED),
        "150 trees, min_samples_leaf=5, sqrt features", None),
    "MLP": ("Neural network", lambda: make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=512, learning_rate_init=1e-3,
        max_iter=60, early_stopping=True, validation_fraction=0.1, n_iter_no_change=6,
        random_state=SEED)), "64-32 ReLU, adam, early stopping", None),
}
