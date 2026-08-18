import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))

from data_loader import get_splits
from model import FraudDetector
from evaluation import find_best_threshold, evaluate_model, plot_pr_curve, plot_threshold_cost

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "creditcard.csv")

print("Loading data...")
X_train, y_train, X_val, y_val, X_test, y_test, scaler = get_splits(DATA_PATH)
print(f"Train: {X_train.shape}, frauds: {y_train.sum()}")
print(f"Val:   {X_val.shape}, frauds: {y_val.sum()}")
print(f"Test:  {X_test.shape}, frauds: {y_test.sum()}")

print("\nTraining baseline model...")
baseline = FraudDetector(class_weight="balanced", n_estimators=200, learning_rate=0.05)
baseline.fit(X_train, y_train)
val_probs = baseline.predict_proba(X_val)
test_probs = baseline.predict_proba(X_test)

print("\nTuning hyperparameters...")
tuned, best_params, best_ap = FraudDetector().tune(X_train, y_train, n_iter=15, cv=3)
print("Best params:", best_params)
print("Best val AP:", best_ap)
tuned_val_probs = tuned.predict_proba(X_val)
tuned_test_probs = tuned.predict_proba(X_test)

print("\nFinding best threshold on validation set...")
best_thr, _ = find_best_threshold(y_val, tuned_val_probs, cost_fn=100.0, cost_fp=10.0)
print(f"Best threshold: {best_thr:.4f}")

print("\nValidation metrics:")
val_metrics = evaluate_model(y_val, tuned_val_probs, threshold=best_thr)
for k, v in val_metrics.items():
    print(f"  {k}: {v}")

print("\nTest metrics:")
test_metrics = evaluate_model(y_test, tuned_test_probs, threshold=best_thr)
for k, v in test_metrics.items():
    print(f"  {k}: {v}")

print("\nGenerating plots...")
plot_pr_curve(y_test, tuned_test_probs, save_path="a2_fraud_detection/results/pr_curve.png")
plot_threshold_cost(y_val, tuned_val_probs, save_path="a2_fraud_detection/results/threshold_cost.png")
print("Done.")
