# A3 Presentation: Credit Card Fraud Detection

---

## Slide 1: Title

**Credit Card Fraud Detection with Cost-Sensitive Gradient Boosting**

UTS 32513 — Advanced Data Analytics Algorithms
Assessment 2 / 3

Goal: flag fraudulent card transactions from historical payment data.

---

## Slide 2: Problem Definition

- Input: transaction record with `Time`, `V1`–`V28` (PCA features), `Amount`.
- Output: fraud probability in [0, 1].
- Decision: flag transactions above a learned threshold for investigator review.
- Why ML? Fraud patterns evolve; rules become stale; labeled historical data exists.

Dataset: 284,807 transactions, only 492 frauds (0.172%).

---

## Slide 3: Data Pipeline

- Temporal split: 60% train / 20% validation / 20% test, ordered by `Time`.
- `Amount` standardized using training statistics.
- `Time` used only for splitting, not as a model feature.

This prevents leakage from future transactions and matches production deployment.

---

## Slide 4: Model and Training

- Model: LightGBM gradient boosted decision trees.
- Loss: binary cross-entropy.
- Class weighting: `balanced` to compensate for rare frauds.
- Hyperparameters tuned by randomized search (15 candidates, 3-fold CV) using average precision.

Best settings: depth 4, 583 trees, learning rate 0.175, large leaf regularization.

---

## Slide 5: Evaluation and Threshold

Training loss ≠ business objective.

- Missing fraud is far more expensive than a false alarm.
- We optimize the classification threshold on validation data to minimize total cost:

```
total_cost = 100 * FN + 10 * FP
```

Chosen threshold: 0.004, not 0.5.

---

## Slide 6: Test Results

| Metric | Value |
|--------|-------|
| AUROC | 0.966 |
| Average Precision | 0.806 |
| Precision | 0.711 |
| Recall | 0.787 |
| F1 | 0.747 |
| Accuracy | 0.999 |
| Precision@100 | 0.60 |

Accuracy is misleading; ranking and cost metrics matter more.

---

## Slide 7: Limitations and Future Work

- Dataset is from 2013; modern fraud differs.
- Features are anonymized PCA components.
- Static model; needs retraining for concept drift.
- Future: instance-specific costs by transaction amount, real-time monitoring, feedback loop.

---

## Anticipated Q&A

Q: Why not use SMOTE to balance the data?
A: SMOTE creates synthetic fraud examples that may not follow real feature distributions. Class weighting and threshold tuning avoid inventing fake data.

Q: Why is the threshold so low (0.004)?
A: Because false negatives cost 10x more than false positives. A low threshold catches more frauds at the cost of more false alarms.

Q: Why LightGBM instead of a neural network?
A: For tabular data of this size, tree ensembles generally train faster and perform better than small neural nets, and they are easier to interpret.
