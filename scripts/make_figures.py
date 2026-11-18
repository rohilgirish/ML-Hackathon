"""Step 6: result figures from the recorded experiment CSVs (no numbers are typed in by hand)."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathsnap import config  # noqa: E402

RS, FIG = config.RESULTS_DIR, config.FIG_DIR
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "-"})
BLUE, ORANGE, GREY, GREEN = "#3b6ea8", "#d9822b", "#9aa0a6", "#2f855a"


def bar(ax, names, vals, err=None, highlight=None, fmt="{:.3f}", color=BLUE):
    cols = [ORANGE if n == highlight else color for n in names]
    ax.bar(names, vals, yerr=err, color=cols, capsize=3)
    for i, v in enumerate(vals):
        ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=7)
    ax.tick_params(axis="x", rotation=35)


def main():
    cmp = pd.read_csv(RS / "model_comparison.csv")
    folds = pd.read_csv(RS / "cv_folds.csv")
    names = cmp.model.tolist()
    sel = "LightGBM"

    fig, ax = plt.subplots(figsize=(8, 3.8))
    bar(ax, names, cmp.cv_macro_f1, cmp.cv_macro_f1_std, sel)
    ax.set(title="5-fold GroupKFold CV macro-F1 (train, mean ± sd)", ylim=(0.8, 0.96))
    fig.tight_layout(); fig.savefig(FIG / "cmp_cv_macro_f1.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    data = [folds[folds.model == n].macro_f1.to_numpy() for n in names]
    ax.boxplot(data, tick_labels=names)
    ax.tick_params(axis="x", rotation=35)
    ax.set(title="Fold-wise CV macro-F1 (stability across folds)")
    fig.tight_layout(); fig.savefig(FIG / "cmp_cv_folds_box.png"); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    bar(ax[0], names, cmp.test_top1, highlight=sel); ax[0].set(title="Student-test Top-1 accuracy (classifier only)", ylim=(0.8, 0.92))
    bar(ax[1], names, cmp.test_macro_f1, highlight=sel); ax[1].set(title="Student-test macro-F1", ylim=(0.8, 0.96))
    fig.tight_layout(); fig.savefig(FIG / "cmp_test_top1_f1.png"); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    bar(ax[0], names, cmp.test_log_loss, highlight=sel); ax[0].set(title="Student-test log loss (lower is better)", yscale="log")
    bar(ax[1], names, cmp.inference_ms_per_1k_obs, highlight=sel, fmt="{:.1f}"); ax[1].set(title="Inference time (ms per 1,000 GPS fixes)", yscale="log")
    fig.tight_layout(); fig.savefig(FIG / "cmp_logloss_inference.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(names))
    ax.bar(x - 0.2, cmp.train_macro_f1, 0.4, label="train (refit)", color=GREY)
    ax.bar(x + 0.2, cmp.test_macro_f1, 0.4, label="student-test", color=BLUE)
    ax.set_xticks(x, names, rotation=35); ax.set(title="Train vs student-test macro-F1 (over/underfitting)", ylim=(0.8, 1.01)); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "cmp_train_vs_test.png"); plt.close(fig)

    # confusion matrices
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
    for ax, n in zip(axes.ravel(), names):
        cm = np.load(RS / "histories" / f"cm_{n}.npy")
        cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n({cmn[i, j]:.2f})", ha="center", va="center",
                        color="white" if cmn[i, j] > 0.6 else "black", fontsize=8)
        ax.set(title=n, xticks=[0, 1], yticks=[0, 1], xticklabels=["not ref", "ref arc"],
               yticklabels=["not ref", "ref arc"], xlabel="predicted", ylabel="true")
        ax.grid(False)
    fig.suptitle("Candidate-level confusion matrices on student-test (threshold 0.5)")
    fig.tight_layout(); fig.savefig(FIG / "cmp_confusion_matrices.png"); plt.close(fig)

    # training histories (only models that actually produce them)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    for ax, n in zip(axes[:2], ["LightGBM", "XGBoost"]):
        h = json.load(open(RS / "histories" / f"{n}.json"))
        ax.plot(h["train"], label="train trips", color=GREY)
        ax.plot(h["valid"], label="10 held-out train trips", color=BLUE)
        ax.set(title=f"{n}: log loss per boosting round", xlabel="boosting round", ylabel="log loss"); ax.legend()
    h = json.load(open(RS / "histories" / "MLP.json"))
    ax = axes[2]
    ax.plot(h["train_loss"], color=GREY, label="training loss")
    ax.set(title="MLP: loss per epoch", xlabel="epoch", ylabel="log loss")
    ax2 = ax.twinx(); ax2.plot(h["valid_score"], color=BLUE, label="validation accuracy"); ax2.set_ylabel("validation accuracy")
    ax.legend(loc="upper center"); ax2.legend(loc="center right")
    fig.tight_layout(); fig.savefig(FIG / "cmp_training_histories.png"); plt.close(fig)

    # Bayesian optimisation
    if (RS / "optuna_trials.csv").exists():
        tr = pd.read_csv(RS / "optuna_trials.csv")
        done = tr[tr.state == "COMPLETE"]
        fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
        ax[0].scatter(done.number, done.value, color=BLUE, s=18, label="completed trial")
        pr = tr[tr.state == "PRUNED"]
        ax[0].scatter(pr.number, [done.value.min()] * len(pr), marker="x", color=GREY, s=18, label="pruned trial")
        ax[0].plot(done.number, done.value.cummax(), color=ORANGE, label="best so far")
        ax[0].set(title="Optuna TPE: CV macro-F1 per trial", xlabel="trial", ylabel="CV macro-F1"); ax[0].legend()
        tv = pd.read_csv(RS / "tuned_vs_untuned.csv")
        metrics = ["cv_macro_f1", "test_top1", "test_macro_f1"]
        x = np.arange(len(metrics))
        for k, (i, r) in enumerate(tv.iterrows()):
            ax[1].bar(x + (k - 0.5) * 0.38, r[metrics], 0.38, label=r.model, color=[GREY, ORANGE][k])
            for j, m in enumerate(metrics):
                ax[1].text(x[j] + (k - 0.5) * 0.38, r[m], f"{r[m]:.4f}", ha="center", va="bottom", fontsize=7)
        ax[1].set_xticks(x, ["CV macro-F1", "test Top-1", "test macro-F1"]); ax[1].set(ylim=(0.85, 0.96), title="Untuned vs tuned LightGBM"); ax[1].legend()
        fig.tight_layout(); fig.savefig(FIG / "bayes_opt.png"); plt.close(fig)

    # HMM vs classifier
    if (RS / "hmm_all_models.csv").exists():
        hm = pd.read_csv(RS / "hmm_all_models.csv")
        fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
        x = np.arange(len(hm))
        for a, (c1, c2, t) in zip(ax, [("clf_top1", "hmm_top1", "Top-1 accuracy"),
                                        ("clf_continuity", "hmm_continuity", "Route continuity"),
                                        ("clf_route_f1", "hmm_route_f1", "Route F1 (length-weighted)")]):
            a.bar(x - 0.2, hm[c1], 0.4, label="classifier only", color=GREY)
            a.bar(x + 0.2, hm[c2], 0.4, label="classifier + HMM", color=ORANGE)
            a.set_xticks(x, hm.model, rotation=35); a.set(title=f"Student-test: {t}", ylim=(max(0, hm[[c1, c2]].min().min() - 0.05), 1.0))
        ax[0].legend()
        fig.tight_layout(); fig.savefig(FIG / "hmm_vs_classifier_all_models.png"); plt.close(fig)

        pt = pd.read_csv(RS / "hmm_vs_classifier_per_trip.csv")
        fig, ax = plt.subplots(figsize=(9, 3.8))
        x = np.arange(len(pt))
        ax.bar(x - 0.2, pt.clf_top1, 0.4, label="classifier only", color=GREY)
        ax.bar(x + 0.2, pt.hmm_top1, 0.4, label="classifier + HMM", color=ORANGE)
        ax.set_xticks(x, pt.trip, rotation=35); ax.set(title="Per-trip Top-1 accuracy, tuned LightGBM (student-test)", ylim=(0.5, 1.0)); ax.legend()
        fig.tight_layout(); fig.savefig(FIG / "hmm_per_trip.png"); plt.close(fig)

        g = pd.read_csv(RS / "hmm_grid.csv")
        piv = g.pivot(index="alpha", columns="beta", values="hmm_top1")
        fig, ax = plt.subplots(figsize=(5, 3.4))
        im = ax.imshow(piv.values, cmap="viridis")
        ax.set_xticks(range(len(piv.columns)), piv.columns); ax.set_yticks(range(len(piv.index)), piv.index)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{piv.values[i, j]:.4f}", ha="center", va="center", color="w", fontsize=8)
        ax.set(xlabel="beta (m)", ylabel="alpha (emission weight)", title="HMM grid: Top-1 on OOF train trips"); ax.grid(False)
        fig.colorbar(im); fig.tight_layout(); fig.savefig(FIG / "hmm_grid.png"); plt.close(fig)
    # feature importance of the final model
    if (config.ARTIFACT_DIR / "model.joblib").exists():
        import joblib
        from pathsnap.features import FEATURES
        m = joblib.load(config.ARTIFACT_DIR / "model.joblib")
        imp = pd.Series(m.booster_.feature_importance("gain"), index=FEATURES).sort_values()
        imp = imp / imp.sum()
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.barh(imp.index, imp.values, color=BLUE)
        ax.set(title="Tuned LightGBM: normalised gain importance", xlabel="share of total gain")
        fig.tight_layout(); fig.savefig(FIG / "final_feature_importance.png"); plt.close(fig)
        imp.sort_values(ascending=False).to_csv(RS / "final_feature_importance.csv", header=["gain_share"])
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
