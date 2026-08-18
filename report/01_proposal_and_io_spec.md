# A2 Project Proposal: Credit Card Fraud Detection

## 1. Task Definition

### 1.1 Problem Statement
Build a machine learning system that flags fraudulent credit card transactions in a stream of payment events. The system must output a fraud probability for each transaction so that investigators can prioritize cases.

### 1.2 Why This Task Matters
Fraud losses on credit and debit cards are estimated in the tens of billions of dollars annually. Manual review of every transaction is impossible at scale. A learning-based system can rank transactions by risk, letting analysts focus effort where it has the highest financial impact.

### 1.3 Why Machine Learning Is Appropriate
- The input space is high-dimensional and noisy.
- Fraud patterns evolve; rule-based systems become stale.
- The cost structure is asymmetric: missing a fraud is far more expensive than investigating a false alarm.
- Historical labeled data exists, so supervised learning is feasible.

### 1.4 Input / Output Specifications

#### Training Phase
| Interface | Specification |
|-----------|---------------|
| Input | A table of historical credit card transactions. Each row is a transaction described by numeric features: `Time` (seconds since first transaction), `V1`–`V28` (PCA-transformed principal components), `Amount` (transaction value in USD), and a binary label `Class` (0 = legitimate, 1 = fraud). |
| Output | A trained classifier that maps a feature vector to a fraud probability. |
| Valid input | A pandas DataFrame or NumPy array with 30 feature columns and one label column. |
| Expected output format | A scikit-learn compatible estimator object with `fit`, `predict`, and `predict_proba` methods. |

#### Deployment Phase
| Interface | Specification |
|-----------|---------------|
| Input | A single transaction or batch of transactions represented by the same 30 feature columns used during training. |
| Output | A fraud probability score in the range [0, 1] for each input transaction. |
| Decision rule | A threshold is chosen on a validation set. Scores above the threshold are flagged for review. |

### 1.5 Dataset
The ULB Machine Learning Group Credit Card Fraud Detection dataset: 284,807 transactions from European cardholders in September 2013, with 492 confirmed frauds (0.172%). Features are PCA-transformed for privacy. The dataset is loaded directly from OpenML inside the notebook using `sklearn.datasets.fetch_openml`, so the notebook is self-contained and reproducible.

### 1.6 Research Question
Can a cost-sensitive gradient-boosted classifier, calibrated with a threshold chosen by validation, outperform naive accuracy-based baselines on this heavily imbalanced fraud detection task?

---

## 2. Learning Framework

### 2.1 Hypothesis Space
The model family is gradient-boosted decision trees. Each tree partitions the feature space into axis-aligned regions and assigns a fraud probability. The ensemble combines many weak trees into a strong classifier.

### 2.2 Loss Function
Training optimizes binary cross-entropy:

```
L(y, p) = -[ y log(p) + (1 - y) log(1 - p) ]
```

where `y` is the true label and `p` is the predicted fraud probability. This loss is well-suited to probabilistic binary classification and naturally outputs calibrated scores.

### 2.3 Task Objective vs. Loss Function
The business objective is to minimize total fraud loss. That is not exactly cross-entropy. A missed fraud may cost hundreds of dollars; a false alarm costs an investigator's time. This mismatch is addressed by:
- Applying class weights during training.
- Optimizing the classification threshold on a validation set using a custom cost-weighted metric.
- Reporting precision, recall, F1, AUROC, and average precision alongside accuracy.

### 2.4 Learning Algorithm
Gradient boosting builds trees sequentially. Each new tree predicts the negative gradient of the loss with respect to the current ensemble's predictions. The algorithm is implemented through the LightGBM library, with hyperparameters tuned by random search on the validation set.

---

## 3. Evaluation Strategy

### 3.1 Data Splits
- Train: first 60% of transactions by time.
- Validation: next 20% of transactions by time.
- Test: final 20% of transactions by time.

A temporal split prevents data leakage from future transactions into training, which matters for real-world deployment.

### 3.2 Metrics
| Metric | Role |
|--------|------|
| AUROC | Measures ranking quality across all thresholds. Robust to class imbalance. |
| Average Precision (AP) | Focuses on the rare positive class; more informative than AUROC on imbalanced data. |
| Precision@k | Of the top-k flagged transactions, what fraction are actually fraud. |
| Recall@threshold | What fraction of frauds are caught at a chosen operating point. |
| Custom cost | `(cost_FN * FN + cost_FP * FP) / total_transactions`, used to choose threshold. |

### 3.3 Validation Procedure
- Hyperparameter search uses 5-fold cross-validation on the training set.
- The final model is retrained on the full training set with the best hyperparameters.
- The threshold and cost weights are selected using the validation set.
- Test set is used only once for final unbiased evaluation.
