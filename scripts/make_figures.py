"""Step 6: result figures for Member 3 (M3: Random Forest & MLP) from the recorded experiment CSVs."""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pathsnap import config  # noqa: E402
from pathsnap.data_io import load_trip  # noqa: E402

RS, FIG = config.RESULTS_DIR, config.FIG_DIR
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
})
BLUE, ORANGE, GREY, GREEN = "#3b6ea8", "#d9822b", "#9aa0a6", "#2f855a"


def bar(ax, names, vals, err=None, highlight=None, fmt="{:.3f}", color=BLUE):
    cols = [ORANGE if n == highlight else color for n in names]
    ax.bar(names, vals, yerr=err, color=cols, capsize=4, width=0.45)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.005, fmt.format(v), ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.tick_params(axis="x", labelsize=9)


def main():
    cmp = pd.read_csv(RS / "model_comparison.csv")
    folds = pd.read_csv(RS / "cv_folds.csv")
    names = cmp.model.tolist()

    # 1. 5-fold GroupKFold CV Macro-F1
    fig, ax = plt.subplots(figsize=(6, 3.8))
    bar(ax, names, cmp.cv_macro_f1, cmp.cv_macro_f1_std)
    ax.set(title="5-fold GroupKFold CV Macro-F1 (Mean ± SD)", ylim=(0.85, 0.98))
    fig.tight_layout()
    fig.savefig(FIG / "cmp_cv_macro_f1.png")
    plt.close(fig)

    # 2. Fold-wise CV stability boxplot
    fig, ax = plt.subplots(figsize=(6, 3.8))
    data = [folds[folds.model == n].macro_f1.to_numpy() for n in names]
    ax.boxplot(data, tick_labels=names, patch_artist=True, boxprops=dict(facecolor="#e2e8f0"))
    ax.set(title="Fold-wise CV Macro-F1 Stability across Trips", ylim=(0.91, 0.96))
    fig.tight_layout()
    fig.savefig(FIG / "cmp_cv_folds_box.png")
    plt.close(fig)

    # 3. Student-test Top-1 & Macro-F1
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    bar(ax[0], names, cmp.test_top1, color=BLUE)
    ax[0].set(title="Student-test Top-1 Accuracy", ylim=(0.8, 0.95))
    bar(ax[1], names, cmp.test_macro_f1, color=GREEN)
    ax[1].set(title="Student-test Macro-F1", ylim=(0.85, 0.98))
    fig.tight_layout()
    fig.savefig(FIG / "cmp_test_top1_f1.png")
    plt.close(fig)

    # 4. Student-test Log Loss & Inference Latency
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    bar(ax[0], names, cmp.test_log_loss, color="#6366f1", fmt="{:.4f}")
    ax[0].set(title="Student-test Log Loss (Lower is better)", ylim=(0.0, 0.1))
    bar(ax[1], names, cmp.inference_ms_per_1k_obs, color="#0891b2", fmt="{:.1f}")
    ax[1].set(title="Inference Latency (ms per 1,000 GPS fixes)")
    fig.tight_layout()
    fig.savefig(FIG / "cmp_logloss_inference.png")
    plt.close(fig)

    # 5. Over/Underfitting: Train vs Student-Test Macro-F1
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    x = np.arange(len(names))
    w = 0.3
    b1 = ax.bar(x - w / 2, cmp.train_macro_f1, w, label="Train (Refit on 80 trips)", color=GREY)
    b2 = ax.bar(x + w / 2, cmp.test_macro_f1, w, label="Student-Test (10 trips)", color=BLUE)
    for b in b1:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005, f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    for b in b2:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005, f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, names)
    ax.set(title="Train vs Student-Test Macro-F1 (Overfitting Diagnostics)", ylim=(0.85, 1.02))
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "cmp_train_vs_test.png")
    plt.close(fig)

    # 6. Candidate-level Confusion Matrices
    fig, axes = plt.subplots(1, len(names), figsize=(4.5 * len(names), 3.8))
    if len(names) == 1:
        axes = [axes]
    for ax, n in zip(axes, names):
        cm = np.load(RS / "histories" / f"cm_{n}.npy")
        cmn = cm / cm.sum(1, keepdims=True)
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    f"{cm[i, j]:,}\n({cmn[i, j]:.2f})",
                    ha="center",
                    va="center",
                    color="white" if cmn[i, j] > 0.6 else "black",
                    fontsize=9,
                    fontweight="bold",
                )
        ax.set(
            title=f"{n} Confusion Matrix",
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["Not Ref", "Ref Arc"],
            yticklabels=["Not Ref", "Ref Arc"],
            xlabel="Predicted",
            ylabel="True",
        )
        ax.grid(False)
    fig.suptitle("Candidate-Level Confusion Matrices on Student-Test (Threshold 0.5)", fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "cmp_confusion_matrices.png")
    plt.close(fig)

    # 7. Training History: MLP Loss per Epoch & Early Stopping
    if (RS / "histories" / "MLP.json").exists():
        h = json.load(open(RS / "histories" / "MLP.json"))
        fig, ax = plt.subplots(figsize=(7, 3.8))
        epochs = np.arange(1, len(h["train_loss"]) + 1)
        ax.plot(epochs, h["train_loss"], color="#dc2626", linewidth=2.2, label="Training Cross-Entropy Loss")
        ax.set(
            title="Multi-Layer Perceptron (MLP): Loss per Epoch & Early Stopping",
            xlabel="Training Epoch",
            ylabel="Log Loss (Binary Cross-Entropy)",
        )
        ax2 = ax.twinx()
        ax2.plot(epochs, h["valid_score"], color="#2563eb", linewidth=2.0, linestyle="--", label="Validation Accuracy (10% Holdout)")
        ax2.set_ylabel("Validation Accuracy", color="#2563eb")
        ax.axvline(x=len(epochs), color="#059669", linestyle=":", linewidth=2, label=f"Early Stopped at Epoch {len(epochs)}")
        lines_1, labels_1 = ax.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax.legend(lines_1 + lines_2, labels_1 + labels_2, loc="center right", fontsize=8.5)
        fig.tight_layout()
        fig.savefig(FIG / "cmp_training_histories.png")
        plt.close(fig)

    # 8. Trajectory Map: Actual Ground Truth Path vs Model Predictions
    rf_pq = RS / "student_test_scores" / "RandomForest.parquet"
    mlp_pq = RS / "student_test_scores" / "MLP.parquet"
    tid = "00000000"
    trip_path = config.STUDENT_TEST_DIR / tid
    if rf_pq.exists() and mlp_pq.exists() and trip_path.exists():
        tr = load_trip(trip_path, with_route=True)
        nodes, arcs, route, track = tr.nodes, tr.arcs, tr.route, tr.track

        rf_df = pd.read_parquet(rf_pq)
        rf_t = rf_df[rf_df.trip == tid].sort_values("p", ascending=False).groupby("obs").first().reset_index()
        rf_arcs = rf_t.sort_values("obs")["arc"].to_numpy()

        mlp_df = pd.read_parquet(mlp_pq)
        mlp_t = mlp_df[mlp_df.trip == tid].sort_values("p", ascending=False).groupby("obs").first().reset_index()
        mlp_arcs = mlp_t.sort_values("obs")["arc"].to_numpy()

        def get_segments(arc_ids):
            segs = []
            for a in arc_ids:
                u, v = arcs[a]
                segs.append([nodes[u], nodes[v]])
            return segs

        fig, axes = plt.subplots(1, 2, figsize=(15, 7), dpi=140)

        # Panel 1: Full trip trajectory
        ax1 = axes[0]
        pad = 0.005
        x_min, x_max = track[:, 0].min() - pad, track[:, 0].max() + pad
        y_min, y_max = track[:, 1].min() - pad, track[:, 1].max() + pad

        net_u, net_v = arcs[:, 0], arcs[:, 1]
        in_box = (nodes[net_u, 0] >= x_min) & (nodes[net_u, 0] <= x_max) & (nodes[net_u, 1] >= y_min) & (nodes[net_u, 1] <= y_max)
        box_arcs = np.where(in_box)[0]
        net_segs = get_segments(box_arcs[:8000])

        lc_net = LineCollection(net_segs, colors="#cfd8dc", linewidths=0.8, alpha=0.6, label="Road Network Graph", zorder=1)
        ax1.add_collection(lc_net)

        route_segs = get_segments(route)
        lc_route = LineCollection(route_segs, colors="#2e7d32", linewidths=4.0, label="Actual Ground Truth Route (.route)", alpha=0.8, zorder=3)
        ax1.add_collection(lc_route)

        rf_segs = get_segments(rf_arcs)
        lc_rf = LineCollection(rf_segs, colors="#d9822b", linewidths=1.8, linestyle="--", label="Random Forest Prediction (Top-1)", alpha=0.9, zorder=4)
        ax1.add_collection(lc_rf)

        ax1.scatter(track[:, 0], track[:, 1], s=5, color="#1565c0", alpha=0.4, label="Raw Noisy GPS Points (.track)", zorder=2)
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
        ax1.set_title(f"Trip {tid}: Full Trajectory Map-Matching", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Longitude (°)")
        ax1.set_ylabel("Latitude (°)")
        ax1.legend(loc="upper left", framealpha=0.92, fontsize=8.5)
        ax1.grid(True, linestyle=":", alpha=0.35)

        # Panel 2: Zoomed Intersection Detail
        ax2 = axes[1]
        sub_track = track[280:430]
        sub_rf_arcs = rf_arcs[280:430]
        sub_mlp_arcs = mlp_arcs[280:430]

        zx_min, zx_max = sub_track[:, 0].min() - 0.0015, sub_track[:, 0].max() + 0.0015
        zy_min, zy_max = sub_track[:, 1].min() - 0.0015, sub_track[:, 1].max() + 0.0015

        in_zbox = (nodes[net_u, 0] >= zx_min) & (nodes[net_u, 0] <= zx_max) & (nodes[net_u, 1] >= zy_min) & (nodes[net_u, 1] <= zy_max)
        zbox_arcs = np.where(in_zbox)[0]
        lc_net_zoom = LineCollection(get_segments(zbox_arcs), colors="#b0bec5", linewidths=1.2, alpha=0.6, label="Road Network Arcs", zorder=1)
        ax2.add_collection(lc_net_zoom)

        lc_route_zoom = LineCollection(route_segs, colors="#2e7d32", linewidths=5.0, label="Actual True Road", alpha=0.65, zorder=3)
        ax2.add_collection(lc_route_zoom)

        lc_rf_zoom = LineCollection(get_segments(sub_rf_arcs), colors="#d9822b", linewidths=2.5, linestyle="--", label="Random Forest Prediction", alpha=0.95, zorder=4)
        ax2.add_collection(lc_rf_zoom)

        lc_mlp_zoom = LineCollection(get_segments(sub_mlp_arcs), colors="#8e44ad", linewidths=1.6, linestyle=":", label="MLP Prediction", alpha=0.95, zorder=5)
        ax2.add_collection(lc_mlp_zoom)

        ax2.scatter(sub_track[:, 0], sub_track[:, 1], s=22, color="#1565c0", alpha=0.8, edgecolors="black", linewidths=0.5, label="Raw GPS Fixes (Jittery)", zorder=6)

        ax2.set_xlim(zx_min, zx_max)
        ax2.set_ylim(zy_min, zy_max)
        ax2.set_title("Zoomed Intersection Detail: Snapping GPS onto True Road", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Longitude (°)")
        ax2.set_ylabel("Latitude (°)")
        ax2.legend(loc="lower right", framealpha=0.92, fontsize=8.5)
        ax2.grid(True, linestyle=":", alpha=0.35)

        fig.suptitle("PathSnap Map-Matching Visualization: Actual Ground Truth vs Model Predictions", fontsize=13, fontweight="bold", y=0.98)
        fig.tight_layout()
        fig.savefig(FIG / "cmp_actual_vs_predicted_path.png")
        plt.close(fig)

    print(f"M3 figures written successfully to {FIG}")


if __name__ == "__main__":
    main()
