"""Comparison figures for the 5 models presented at the first evaluation
(one per algorithm family). Reads the recorded results only; nothing is retrained.
Outputs: results/figures/five_models_*.png
"""
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
FIVE = ["LogisticRegression", "KNN", "RandomForest", "LightGBM", "MLP"]
LABELS = ["Logistic\nRegression", "KNN", "Random\nForest", "LightGBM", "MLP"]
BEST = "LightGBM"
BLUE, ORANGE, GREY = "#3b6ea8", "#d9822b", "#9aa0a6"
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25})


def bars(ax, vals, title, err=None, fmt="{:.4f}", ylim=None, log=False, lower_better=False):
    cols = [ORANGE if m == BEST else BLUE for m in FIVE]
    ax.bar(LABELS, vals, yerr=err, color=cols, capsize=3)
    tops = np.asarray(vals) + (np.asarray(err) if err is not None else 0)
    for i, v in enumerate(vals):
        ax.text(i, tops[i], fmt.format(v), ha="center", va="bottom", fontsize=7.5)
    ax.set_title(title + (" (lower is better)" if lower_better else ""), fontsize=10)
    if ylim:
        ax.set_ylim(*ylim)
    if log:
        ax.set_yscale("log")
    ax.tick_params(axis="x", labelsize=8)


def main():
    cmp = pd.read_csv(RS / "model_comparison.csv").set_index("model").loc[FIVE]
    folds = pd.read_csv(RS / "cv_folds.csv")

    # 1. one-page summary dashboard
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    bars(ax[0, 0], cmp.cv_macro_f1, "CV macro-F1 (5-fold GroupKFold, mean ± sd)", cmp.cv_macro_f1_std, ylim=(0.9, 0.96))
    bars(ax[0, 1], cmp.test_top1, "Student-test Top-1 accuracy", ylim=(0.85, 0.92))
    bars(ax[0, 2], cmp.test_macro_f1, "Student-test macro-F1", ylim=(0.9, 0.95))
    bars(ax[1, 0], cmp.test_log_loss, "Student-test log loss", lower_better=True)
    bars(ax[1, 1], cmp.test_route_continuity, "Route continuity (classifier only)", fmt="{:.3f}", ylim=(0.85, 0.97))
    bars(ax[1, 2], cmp.inference_ms_per_1k_obs, "Inference time, ms / 1,000 GPS fixes", fmt="{:.1f}", log=True, lower_better=True)
    fig.suptitle("First evaluation: 5 models, one per algorithm family (orange = selected: LightGBM)", fontsize=13)
    fig.tight_layout(); fig.savefig(FIG / "five_models_summary.png"); plt.close(fig)

    # 2. CV stability across folds
    fig, ax = plt.subplots(figsize=(8, 4))
    data = [folds[folds.model == m].macro_f1.to_numpy() for m in FIVE]
    bp = ax.boxplot(data, tick_labels=LABELS, patch_artist=True)
    for patch, m in zip(bp["boxes"], FIVE):
        patch.set_facecolor(ORANGE if m == BEST else BLUE)
        patch.set_alpha(0.6)
    for i, d in enumerate(data):
        ax.scatter(np.full(len(d), i + 1), d, color="k", s=10, zorder=3)
    ax.set(title="CV macro-F1 per fold (stability)", ylabel="macro-F1")
    fig.tight_layout(); fig.savefig(FIG / "five_models_cv_folds.png"); plt.close(fig)

    # 3. over/underfitting: train vs CV vs test
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(FIVE))
    for k, (col, lab, c) in enumerate([("train_macro_f1", "train (refit)", GREY), ("cv_macro_f1", "CV", BLUE),
                                       ("test_macro_f1", "student-test", ORANGE)]):
        ax.bar(x + (k - 1) * 0.27, cmp[col], 0.27, label=lab, color=c)
        for i, v in enumerate(cmp[col]):
            ax.text(x[i] + (k - 1) * 0.27, v, f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x, LABELS); ax.set(ylim=(0.88, 1.02), title="Train vs CV vs student-test macro-F1 (over/underfitting)")
    ax.legend(loc="upper left", ncol=3)
    fig.tight_layout(); fig.savefig(FIG / "five_models_train_vs_test.png"); plt.close(fig)

    # 4. confusion matrices
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.8))
    for ax, m, lab in zip(axes, FIVE, LABELS):
        cm = np.load(RS / "histories" / f"cm_{m}.npy")
        cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap="Oranges" if m == BEST else "Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n({cmn[i, j]:.2f})", ha="center", va="center",
                        color="white" if cmn[i, j] > 0.6 else "black", fontsize=8)
        ax.set(title=lab.replace("\n", " "), xticks=[0, 1], yticks=[0, 1], xticklabels=["wrong", "true arc"],
               yticklabels=["wrong", "true arc"], xlabel="predicted", ylabel="actual")
        ax.grid(False)
    fig.suptitle("Candidate-level confusion matrices, student-test (threshold 0.5)")
    fig.tight_layout(); fig.savefig(FIG / "five_models_confusion.png"); plt.close(fig)

    # 5. real training histories (only LightGBM and MLP among the 5 produce them)
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    h = json.load(open(RS / "histories" / "LightGBM.json"))
    best = int(np.argmin(h["valid"]))
    ax[0].plot(h["train"], color=GREY, label="training trips")
    ax[0].plot(h["valid"], color=ORANGE, label="10 held-out training trips")
    ax[0].axvline(best, color="k", ls="--", lw=1, label=f"min validation loss (round {best})")
    ax[0].set(title="LightGBM: log loss per boosting round", xlabel="boosting round", ylabel="log loss"); ax[0].legend()
    h = json.load(open(RS / "histories" / "MLP.json"))
    ax[1].plot(h["train_loss"], color=GREY, label="training loss")
    ax[1].set(title="MLP: loss per epoch (early stopping)", xlabel="epoch", ylabel="log loss")
    a2 = ax[1].twinx(); a2.plot(h["valid_score"], color=BLUE, label="validation accuracy"); a2.set_ylabel("validation accuracy")
    ax[1].legend(loc="upper center"); a2.legend(loc="center right")
    fig.tight_layout(); fig.savefig(FIG / "five_models_training_curves.png"); plt.close(fig)

    cmp.reset_index().to_csv(RS / "five_models_comparison.csv", index=False)
    print("wrote five_models_*.png")


if __name__ == "__main__":
    main()
