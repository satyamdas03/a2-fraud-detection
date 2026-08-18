# Results and Discussion

## 1. Experimental Setup

- Dataset: ULB Credit Card Fraud Detection (284,807 transactions, 492 frauds, 0.172% fraud rate).
- Splits: temporal 60/20/20 by `Time`.
- Model: LightGBM with class balancing, hyperparameters tuned by randomized search (15 candidates, 3-fold CV).
- Threshold: chosen on validation set to minimize total cost with `cost_FN = 100`, `cost_FP = 10`.

## 2. Hyperparameter Search Results

Best parameters from `RandomizedSearchCV`:

```
{'colsample_bytree': 1.0,
 'learning_rate': 0.175,
 'max_depth': 4,
 'min_child_samples': 199,
 'n_estimators': 583,
 'num_leaves': 33,
 'subsample': 1.0}
```

Best cross-validated average precision: **0.8464**.

The search preferred a shallow tree depth (`max_depth=4`) and large `min_child_samples=199`, suggesting the model benefits from strong regularization rather than complex partitions. This is consistent with the heavy class imbalance: deep trees could overfit to the few fraud examples.

## 3. Test Set Performance

| Metric | Value |
|--------|-------|
| Accuracy | 0.9993 |
| Precision | 0.7108 |
| Recall | 0.7867 |
| F1 | 0.7468 |
| AUROC | 0.9660 |
| Average Precision | 0.8064 |
| Total cost (cost_FN=100, cost_FP=10) | 1840 |
| Cost per transaction | 0.0323 |
| Threshold | 0.004 |
| Precision@100 | 0.60 |

## 4. Interpretation

- **Accuracy is misleading:** 99.93% accuracy looks excellent, but a trivial classifier that always predicts "legitimate" would score 99.83%. Precision, recall, and AP are more honest.
- **Ranking quality is strong:** AUROC of 0.966 and AP of 0.806 show the model separates frauds from legitimate transactions well.
- **Operating point is aggressive:** The threshold 0.004 is low because missing a fraud costs 10x more than a false alarm. This pushes recall above precision, catching ~79% of frauds while flagging ~29% of flagged cases as false positives.
- **Top-100 precision is 60%:** If investigators can only review 100 transactions, 60 are likely fraud. This is a practical deployment metric.

## 5. Loss Function vs. Task Objective

The training loss is binary cross-entropy; the business objective is minimizing fraud cost. These are not identical. Cross-entropy produces calibrated probabilities but does not encode the asymmetric cost of false negatives. We bridge the gap in two ways:

1. **Class weights** during training increase the influence of fraud examples.
2. **Threshold optimization** after training selects the operating point that minimizes cost on validation data.

A limitation remains: cross-entropy treats all frauds as equally costly and all false alarms as equally cheap. In reality, a $10,000 fraudulent transaction is worse than a $1 fraud, and repeated false alarms on VIP customers carry reputational cost. A more refined objective would use instance-specific costs.

## 6. Limitations

- **Dataset age:** The data is from 2013. Modern fraud patterns differ.
- **Anonymized features:** `V1`–`V28` are PCA components, so interpretation beyond feature importance is limited.
- **Static model:** Fraudsters adapt. The model would need periodic retraining.
- **No concept drift detection:** We do not monitor distribution shifts between training and live data.

## 7. Future Work

- Add real-time monitoring of feature distributions and model performance.
- Experiment with instance-weighted losses that account for transaction amount.
- Investigate ensemble methods that combine LightGBM with anomaly detection for emerging fraud patterns.
- Calibrate probabilities on the validation set and report confidence intervals.
- Include investigator feedback loop to continuously retrain on confirmed frauds.
