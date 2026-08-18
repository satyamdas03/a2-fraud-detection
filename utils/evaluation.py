import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def compute_metrics(y_true, y_pred, y_score):
    """Return a dict of standard classification metrics."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auroc": roc_auc_score(y_true, y_score),
        "average_precision": average_precision_score(y_true, y_score),
    }


def total_cost(y_true, y_pred, cost_fn=100.0, cost_fp=10.0):
    """Compute asymmetric cost of false negatives and false positives."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fn * cost_fn + fp * cost_fp


def cost_per_transaction(y_true, y_pred, cost_fn=100.0, cost_fp=10.0):
    """Average cost across all transactions."""
    return total_cost(y_true, y_pred, cost_fn, cost_fp) / len(y_true)


def find_best_threshold(y_true, y_score, cost_fn=100.0, cost_fp=10.0):
    """Pick threshold on validation set that minimizes total cost."""
    thresholds = np.linspace(0.001, 0.999, 999)
    costs = []
    for thr in thresholds:
        y_pred = (y_score >= thr).astype(int)
        costs.append(total_cost(y_true, y_pred, cost_fn, cost_fp))
    best_idx = int(np.argmin(costs))
    return thresholds[best_idx], costs[best_idx]


def precision_at_k(y_true, y_score, k=100):
    """Precision among the top-k highest scored samples."""
    top_k_idx = np.argsort(y_score)[-k:]
    return precision_score(y_true[top_k_idx], np.ones(k), zero_division=0)


def plot_pr_curve(y_true, y_score, save_path="pr_curve.png"):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"AP = {ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_threshold_cost(y_true, y_score, cost_fn=100.0, cost_fp=10.0, save_path="threshold_cost.png"):
    thresholds = np.linspace(0.001, 0.999, 999)
    costs = [total_cost(y_true, (y_score >= t).astype(int), cost_fn, cost_fp) for t in thresholds]

    best_thr, best_cost = find_best_threshold(y_true, y_score, cost_fn, cost_fp)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(thresholds, costs, label="Total cost")
    ax.axvline(best_thr, color="red", linestyle="--", label=f"Best threshold = {best_thr:.3f}")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Total cost")
    ax.set_title("Cost vs. Classification Threshold")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def evaluate_model(y_true, y_score, threshold=0.5, cost_fn=100.0, cost_fp=10.0):
    """Full evaluation report."""
    y_pred = (y_score >= threshold).astype(int)
    metrics = compute_metrics(y_true, y_pred, y_score)
    metrics["precision_at_100"] = precision_at_k(y_true, y_score, k=100)
    metrics["total_cost"] = total_cost(y_true, y_pred, cost_fn, cost_fp)
    metrics["cost_per_txn"] = cost_per_transaction(y_true, y_pred, cost_fn, cost_fp)
    metrics["threshold"] = threshold
    return metrics
