import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV

class SoftVotingEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, models, weights=None):
        self.models  = models
        self.weights = weights or [1.0] * len(models)

    def fit(self, X, y):
        for name, model in self.models:
            print(f"  Fitting {name}...")
            model.fit(X, y)
        return self

    def predict_proba(self, X):
        probas = np.array([w * m.predict_proba(X)
                           for (_, m), w in zip(self.models, self.weights)])
        return probas.sum(axis=0) / sum(self.weights)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)

    @classmethod
    def from_types(cls, model_types, weights=None):
        from models.classifier import build_model
        models = [(t, build_model(t)) for t in model_types]
        return cls(models, weights)
