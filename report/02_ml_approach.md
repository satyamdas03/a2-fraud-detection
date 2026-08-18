# Machine Learning Approach

## 1. Model Choice and Justification

We use gradient boosted decision trees implemented in LightGBM. The reasoning:

- **Tabular data:** Fraud detection runs on structured features. Tree ensembles outperform neural networks on small-to-medium tabular problems (Grinsztajn et al., 2022).
- **Non-linearity:** Individual trees capture feature interactions without hand-engineered products.
- **Scalability:** LightGBM trains quickly on 284k rows and handles class imbalance via `class_weight`.
- **Interpretability:** Feature importance and SHAP values can explain individual predictions, which matters in regulated financial settings.

The model family is an ensemble of decision trees:

```
F(x) = sum_{m=1}^M eta * h_m(x)
```

where each `h_m` is a decision tree and `eta` is the learning rate. The final fraud probability is passed through a sigmoid.

## 2. Hypothesis Space and Parameterization

The hypothesis space is all functions produced by the LightGBM training algorithm given the hyperparameter ranges we search. Key parameters:

| Parameter | Role |
|-----------|------|
| `num_leaves` | Complexity of each tree. More leaves = finer partitions, higher variance. |
| `max_depth` | Maximum tree depth. Constrains interaction order. |
| `n_estimators` | Number of trees in the ensemble. More trees can overfit if learning rate is too high. |
| `learning_rate` | Shrinkage applied to each new tree. Lower values need more trees. |
| `min_child_samples` | Minimum samples in a leaf. Regularizes small leaves. |
| `subsample` / `colsample_bytree` | Stochastic sampling of rows/features per tree. Reduces overfitting. |
| `class_weight="balanced"` | Adjusts loss gradients to compensate for rarity of fraud. |

These map directly to the code in `model.py` and to the `RandomizedSearchCV` block in the notebook.

## 3. Loss Function and Training Procedure

Training minimizes binary cross-entropy:

```
L(y, p) = -[ y log(p) + (1 - y) log(1 - p) ]
```

This is the natural loss for probabilistic binary classification. In LightGBM the gradients are:

```
g_i = p_i - y_i
h_i = p_i (1 - p_i)
```

Each new tree is fit to the negative gradient, and the leaf outputs are adjusted by the Hessian. This second-order optimization is why gradient boosting converges faster than first-order methods.

`class_weight="balanced"` reweights samples inversely by class frequency, so the gradient of a fraud example contributes more than a legitimate example. This is one way to handle imbalance; we combine it with threshold tuning.

## 4. Evaluation Metrics and Their Relevance

| Metric | Why it is used |
|--------|----------------|
| **AUROC** | Threshold-independent ranking quality. Useful when the operating point is unknown. |
| **Average Precision (AP)** | Area under the precision-recall curve. More sensitive to the rare class than AUROC. |
| **Precision / Recall / F1** | Standard point-wise metrics at the chosen threshold. |
| **Total cost** | Custom metric that weights false negatives 10x more than false positives, matching business impact. |
| **Precision@100** | Of the 100 highest-risk transactions, how many are fraud. Directly relevant to investigator capacity. |

## 5. Validation Scheme

- **Temporal split:** 60% train, 20% validation, 20% test, ordered by `Time`. This prevents look-ahead leakage and mimics production deployment.
- **Cross-validation:** 3-fold CV on the training set is used during `RandomizedSearchCV`.
- **Single test use:** The test set is evaluated only after threshold selection, giving an unbiased estimate.

## 6. Data Preprocessing

- `Time` is dropped because it is only useful for splitting, not as a feature.
- `Amount` is standardized with `StandardScaler` fit on the training set.
- `V1`–`V28` are already PCA-transformed in the original dataset, so no further scaling is applied.
- No SMOTE or synthetic oversampling is used; we rely on class weights and threshold tuning to avoid generating unrealistic synthetic fraud examples.
