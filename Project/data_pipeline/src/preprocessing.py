"""Leakage-free preprocessing: every statistic is learned in fit() on the TRAIN split only."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from schema import CAT_COLS, INT_COLS, MISSING


class Cleaner(BaseEstimator, TransformerMixin):
    """'?'/NaN -> missing; numeric: median impute + clip to train 1st/99th percentile;
    categorical: most-frequent (mode) impute. All statistics come from the train data in fit()."""

    def __init__(self, num_cols=None, cat_cols=None, q_low=0.01, q_high=0.99):
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.q_low = q_low
        self.q_high = q_high

    def _as_missing(self, X):
        X = X.copy()
        for c in self.cat_cols:
            X[c] = X[c].astype(object).where(X[c].notna(), MISSING).astype(str).str.strip()
            X[c] = X[c].replace(MISSING, np.nan)
        for c in self.num_cols:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype(float)
        return X

    def fit(self, X, y=None):
        X = self._as_missing(X)
        self.medians_ = {c: float(X[c].median()) for c in self.num_cols}
        self.modes_ = {c: X[c].mode(dropna=True).iloc[0] for c in self.cat_cols}
        filled = X[self.num_cols].fillna(pd.Series(self.medians_))
        self.low_ = {c: float(filled[c].quantile(self.q_low)) for c in self.num_cols}
        self.high_ = {c: float(filled[c].quantile(self.q_high)) for c in self.num_cols}
        return self

    def transform(self, X):
        X = self._as_missing(X)
        for c in self.num_cols:
            X[c] = X[c].fillna(self.medians_[c]).clip(self.low_[c], self.high_[c])
        for c in self.cat_cols:
            X[c] = X[c].fillna(self.modes_[c])
        return X


class DerivedFeatures(BaseEstimator, TransformerMixin):
    """capital_net = capital-gain - capital-loss;  hours_per_age = hours-per-week / (age - 16)."""

    def __init__(self, enabled=True):
        self.enabled = enabled

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.enabled:
            X["capital_net"] = X["capital-gain"] - X["capital-loss"]
            # age >= 17 in the schema, so (age - 16) >= 1; clipping keeps age >= 17 as well
            X["hours_per_age"] = X["hours-per-week"] / (X["age"] - 16).clip(lower=1)
        return X


def build_preprocessor(derived=True, q=(0.01, 0.99)):
    num = list(INT_COLS) + (["capital_net", "hours_per_age"] if derived else [])
    encode = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
    ])
    return Pipeline([
        ("clean", Cleaner(num_cols=list(INT_COLS), cat_cols=list(CAT_COLS), q_low=q[0], q_high=q[1])),
        ("derive", DerivedFeatures(enabled=derived)),
        ("encode", encode),
    ])
