"""Candidate classifiers compared in the experiments (one per algorithm family)."""
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import SEED

# name -> (family, factory, key configuration, max training rows or None)
MODEL_ZOO = {
    "LogisticRegression": ("Linear", lambda: make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
        "L2, C=1.0, standardised", None),
    "GaussianNB": ("Probabilistic", lambda: GaussianNB(var_smoothing=1e-9),
        "var_smoothing=1e-9", None),
    "KNN": ("Instance-based", lambda: make_pipeline(
        StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1)),
        "k=25, distance-weighted, standardised, 60k-row reference set", 60_000),
    "DecisionTree": ("Tree", lambda: DecisionTreeClassifier(max_depth=12, min_samples_leaf=20,
        random_state=SEED), "max_depth=12, min_samples_leaf=20", None),
    "RandomForest": ("Ensemble (bagging)", lambda: RandomForestClassifier(
        n_estimators=150, min_samples_leaf=5, max_features="sqrt", n_jobs=-1, random_state=SEED),
        "150 trees, min_samples_leaf=5, sqrt features", None),
    "LightGBM": ("Boosting", lambda: lgb.LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, random_state=SEED, verbose=-1, n_jobs=-1),
        "400 trees, lr=0.05, 63 leaves", None),
    "XGBoost": ("Boosting", lambda: xgb.XGBClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=7, subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", eval_metric="logloss", random_state=SEED, n_jobs=-1),
        "400 trees, lr=0.05, depth=7", None),
    "MLP": ("Neural network", lambda: make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=512, learning_rate_init=1e-3,
        max_iter=60, early_stopping=True, validation_fraction=0.1, n_iter_no_change=6,
        random_state=SEED)), "64-32 ReLU, adam, early stopping", None),
}
