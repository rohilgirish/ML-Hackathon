"""Step 4: Bayesian hyperparameter optimisation (Optuna TPE) of the selected model (LightGBM).

Objective: mean macro-F1 over 5-fold GroupKFold on the 80 training trips (trip-grouped).
Every trial is logged to results/optuna_trials.csv and an SQLite study (results/optuna.db).
The tuned model is then refit on all training trips and compared with the untuned default
on the identical student-testing partition. Out-of-fold probabilities of the tuned model
are saved for tuning the HMM decoder without leakage.
"""
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathsnap import config  # noqa: E402
from pathsnap.dataset import build_partition  # noqa: E402
from pathsnap.features import FEATURES  # noqa: E402
from pathsnap.metrics import candidate_metrics, top1_accuracy, top1_predictions  # noqa: E402
from pathsnap.models import MODEL_ZOO  # noqa: E402
from train_models import continuity_for, training_rows  # noqa: E402

N_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 30

SEARCH_SPACE = {
    "n_estimators": ("int", 100, 800),
    "learning_rate": ("float_log", 0.01, 0.2),
    "num_leaves": ("int_log", 15, 255),
    "min_child_samples": ("int_log", 5, 300),
    "subsample": ("float", 0.5, 1.0),
    "colsample_bytree": ("float", 0.4, 1.0),
    "reg_lambda": ("float_log", 1e-3, 10.0),
    "reg_alpha": ("float_log", 1e-3, 10.0),
}


def suggest(trial):
    p = {}
    for k, (kind, lo, hi) in SEARCH_SPACE.items():
        if kind == "int":
            p[k] = trial.suggest_int(k, lo, hi)
        elif kind == "int_log":
            p[k] = trial.suggest_int(k, lo, hi, log=True)
        elif kind == "float_log":
            p[k] = trial.suggest_float(k, lo, hi, log=True)
        else:
            p[k] = trial.suggest_float(k, lo, hi)
    return p


def make(params):
    return lgb.LGBMClassifier(subsample_freq=1, random_state=config.SEED, verbose=-1, n_jobs=-1, **params)


def main():
    tr_f, tr_l = build_partition("train")
    te_f, te_l = build_partition("student_test")
    train = training_rows(tr_f)
    X, y, groups = train[FEATURES].to_numpy(np.float32), train.y.to_numpy(), train.trip.to_numpy()
    folds = list(GroupKFold(n_splits=config.CV_FOLDS).split(X, y, groups))

    def cv_score(model_factory, return_oof=False):
        f1s, oof = [], np.zeros(len(y))
        for k, (a, b) in enumerate(folds):
            m = model_factory()
            m.fit(X[a], y[a])
            p = m.predict_proba(X[b])[:, 1]
            oof[b] = p
            f1s.append(candidate_metrics(y[b], p)["macro_f1"])
            yield k, float(np.mean(f1s)), f1s, oof

    def objective(trial):
        params = suggest(trial)
        t0 = time.time()
        for k, mean_f1, f1s, _ in cv_score(lambda: make(params)):
            trial.report(mean_f1, k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("fold_f1", f1s)
        trial.set_user_attr("cv_time_s", time.time() - t0)
        return mean_f1

    storage = f"sqlite:///{config.RESULTS_DIR / 'optuna.db'}"
    study = optuna.create_study(study_name="lightgbm_macro_f1", direction="maximize", storage=storage,
                                load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=config.SEED, n_startup_trials=8),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
    # enqueue the untuned configuration as trial 0 for a fair reference point
    if len(study.trials) == 0:
        study.enqueue_trial({"n_estimators": 400, "learning_rate": 0.05, "num_leaves": 63,
                             "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8,
                             "reg_lambda": 1e-3, "reg_alpha": 1e-3})
    t0 = time.time()
    remaining = N_TRIALS - len([t for t in study.trials if t.state != optuna.trial.TrialState.WAITING])
    if remaining > 0:
        study.optimize(objective, n_trials=remaining)
    opt_time = time.time() - t0
    study.trials_dataframe().to_csv(config.RESULTS_DIR / "optuna_trials.csv", index=False)
    best = study.best_params
    print("best", study.best_value, best)

    # ---- compare untuned vs tuned on identical partitions ----
    untuned_factory = MODEL_ZOO["LightGBM"][1]
    rows = []
    for label, factory in [("untuned", untuned_factory), ("tuned", lambda: make(best))]:
        *_, (k, cv_f1, f1s, _) = list(cv_score(factory))
        m = factory()
        t1 = time.time()
        m.fit(X, y)
        fit_t = time.time() - t1
        t1 = time.time()
        p = m.predict_proba(te_f[FEATURES].to_numpy(np.float32))[:, 1]
        inf_t = time.time() - t1
        te = te_f.assign(p=p)
        rel = te[te.reliable]
        cm = candidate_metrics(rel.y.to_numpy(), rel.p.to_numpy())
        pred = top1_predictions(te)
        rows.append({"model": f"LightGBM ({label})", "cv_macro_f1": cv_f1, "cv_macro_f1_std": float(np.std(f1s, ddof=1)),
                     "test_top1": top1_accuracy(pred, te_l), "test_macro_f1": cm["macro_f1"],
                     "test_log_loss": cm["log_loss"], "test_route_continuity": continuity_for(pred, config.STUDENT_TEST_DIR),
                     "fit_time_s": fit_t, "inference_ms_per_1k_obs": 1e6 * inf_t / len(te_l)})
        if label == "tuned":
            te[["trip", "obs", "arc", "p"]].to_parquet(config.RESULTS_DIR / "student_test_scores" / "LightGBM_tuned.parquet")
            config.ARTIFACT_DIR.mkdir(exist_ok=True)
            import joblib
            joblib.dump(m, config.ARTIFACT_DIR / "model.joblib")
            (config.ARTIFACT_DIR / "model_meta.json").write_text(json.dumps({
                "model": "LightGBM (Optuna-tuned)", "features": FEATURES, "params": best,
                "seed": config.SEED, "candidate_radius_m": config.CAND_RADIUS_M,
                "candidate_max_k": config.CAND_MAX_K, "trained_on": "80 training trips (stride 3, reliable labels)",
            }, indent=2))
    cmp = pd.DataFrame(rows)
    cmp["optimisation_time_s"] = opt_time
    cmp["n_trials"] = len(study.trials)
    cmp.to_csv(config.RESULTS_DIR / "tuned_vs_untuned.csv", index=False)
    print(cmp.to_string())
    # out-of-fold probabilities for *every* fix of each held-out training trip (for HMM tuning)
    full_p = np.zeros(len(tr_f))
    for a, b in folds:
        held = set(groups[b])
        m = make(best)
        m.fit(X[a], y[a])
        mask = tr_f.trip.isin(held).to_numpy()
        full_p[mask] = m.predict_proba(tr_f.loc[mask, FEATURES].to_numpy(np.float32))[:, 1]
    tr_f[["trip", "obs", "arc"]].assign(p=full_p).to_parquet(config.RESULTS_DIR / "train_oof_tuned.parquet")
    json.dump({"search_space": SEARCH_SPACE, "objective": "mean 5-fold GroupKFold macro-F1",
               "sampler": "TPE (seed 42, 8 startup trials)", "pruner": "Median (by fold)",
               "n_trials": len(study.trials), "best_value": study.best_value, "best_params": best,
               "optimisation_time_s": opt_time},
              open(config.RESULTS_DIR / "optuna_summary.json", "w"), indent=2)


if __name__ == "__main__":
    main()
