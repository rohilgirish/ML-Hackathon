"""Evaluation metrics: candidate-level classification, observation-level and route-level."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score


def candidate_metrics(y, p, thr=0.5) -> dict:
    yhat = (p >= thr).astype(int)
    return {
        "macro_f1": f1_score(y, yhat, average="macro"),
        "precision": precision_score(y, yhat, average="macro", zero_division=0),
        "recall": recall_score(y, yhat, average="macro", zero_division=0),
        "pos_f1": f1_score(y, yhat),
        "log_loss": log_loss(y, np.clip(p, 1e-7, 1 - 1e-7), labels=[0, 1]),
    }


def top1_predictions(df: pd.DataFrame, score_col="p") -> pd.DataFrame:
    """Highest-scoring candidate per (trip, obs)."""
    idx = df.groupby(["trip", "obs"], sort=False)[score_col].idxmax()
    return df.loc[idx, ["trip", "obs", "arc", score_col]].rename(columns={"arc": "pred_arc"})


def top1_accuracy(pred: pd.DataFrame, labels: pd.DataFrame) -> float:
    """Over reliable observations. Observations whose reference arc is missing from the
    candidate set (or without prediction) count as errors."""
    m = labels[labels.reliable].merge(pred, on=["trip", "obs"], how="left")
    return float((m["pred_arc"] == m["ref_arc"]).mean())


def route_continuity(arc_seq, net, max_m: float = 100.0) -> float:
    """Fraction of changes of matched arc that are physically drivable: the next arc's start
    node is reachable from the previous arc's end node within `max_m` metres along the
    directed network (0 m = directly connected). At 1 Hz a vehicle can pass a short arc
    between two fixes, hence the small tolerance instead of requiring direct adjacency."""
    s = [int(a) for a in arc_seq if a is not None and a >= 0]
    s = [a for i, a in enumerate(s) if i == 0 or a != s[i - 1]]
    if len(s) < 2:
        return 1.0
    ok = [net.node_dist(int(net.arcs[a, 1]), int(net.arcs[b, 0]), max_m) <= max_m for a, b in zip(s[:-1], s[1:])]
    return float(np.mean(ok))


def route_overlap(pred_route, ref_route, lengths: np.ndarray) -> dict:
    """Length-weighted precision/recall/F1 between the reconstructed and reference arc sets."""
    P, R = set(int(a) for a in pred_route), set(int(a) for a in ref_route)
    inter = sum(lengths[a] for a in P & R)
    lp, lr = sum(lengths[a] for a in P), sum(lengths[a] for a in R)
    prec = inter / lp if lp else 0.0
    rec = inter / lr if lr else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"route_precision": prec, "route_recall": rec, "route_f1": f1}
