"""Importable estimators for independent home/away goal models.
Keep this module at src/regression/goal_models.py for joblib loading.
"""
import numpy as np
from scipy.stats import poisson
from sklearn.base import BaseEstimator
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class PairedGoalModel(BaseEstimator):
    def __init__(self, columns, alpha=1.0, blend=1.0, binary=False):
        self.columns = columns
        self.alpha = alpha
        self.blend = blend
        self.binary = binary

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        if y.ndim != 2 or y.shape[1] != 2:
            raise ValueError('Expected two goal targets: FTHG and FTAG')
        if not np.isfinite(y).all() or (y < 0).any():
            raise ValueError('Invalid score targets')
        self.models_ = []
        self.means_ = y.mean(axis=0)
        self.prior_ = float(np.mean(y.sum(axis=1) >= 3))
        self.classes_ = np.array([0, 1])
        for i in range(2):
            model = make_pipeline(StandardScaler(), PoissonRegressor(alpha=self.alpha, max_iter=1000))
            model.fit(X[list(self.columns)], y[:, i])
            self.models_.append(model)
        return self

    def predict_goal_rates(self, X):
        rates = np.column_stack([m.predict(X[list(self.columns)]) for m in self.models_])
        return np.clip(self.blend * rates + (1-self.blend) * self.means_, 0.01, 10)

    def predict_proba(self, X):
        # Independent Poisson goal counts imply a Poisson total.
        # Blend with the observed training prior to reduce overconfidence.
        raw_rates = np.column_stack([m.predict(X[list(self.columns)]) for m in self.models_])
        p = poisson.sf(2, np.clip(raw_rates, 0.01, 10).sum(axis=1))
        p = np.clip(self.blend * p + (1-self.blend) * self.prior_, 1e-6, 1-1e-6)
        return np.column_stack([1-p, p])

    def predict(self, X):
        if self.binary:
            return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
        return self.predict_goal_rates(X).sum(axis=1)
