import numpy as np
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, randint


class FraudDetector:
    """LightGBM classifier for credit card fraud detection."""

    def __init__(self, class_weight="balanced", n_estimators=200, learning_rate=0.05,
                 num_leaves=31, max_depth=-1, random_state=42):
        self.class_weight = class_weight
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.random_state = random_state
        self.model = None

    def _build_estimator(self):
        return lgb.LGBMClassifier(
            objective="binary",
            boosting_type="gbdt",
            class_weight=self.class_weight,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1,
        )

    def fit(self, X, y):
        self.model = self._build_estimator()
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X):
        return self.model.predict(X)

    def tune(self, X, y, n_iter=20, cv=3):
        """Randomized search over key LightGBM hyperparameters."""
        param_dist = {
            "n_estimators": randint(100, 600),
            "learning_rate": loguniform(1e-2, 3e-1),
            "num_leaves": randint(20, 150),
            "max_depth": randint(3, 12),
            "min_child_samples": randint(10, 200),
            "subsample": [0.6, 0.8, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
        }
        base = self._build_estimator()
        search = RandomizedSearchCV(
            base,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring="average_precision",
            cv=cv,
            n_jobs=-1,
            random_state=self.random_state,
            verbose=1,
        )
        search.fit(X, y)
        self.model = search.best_estimator_
        return self, search.best_params_, search.best_score_
