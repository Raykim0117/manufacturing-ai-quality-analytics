"""Reusable train-fitted feature selection, median imputation and scaling."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted


class MissingnessSelector(TransformerMixin, BaseEstimator):
    """Keep measured feature order; learn missingness exclusively during fit."""

    def __init__(self, feature_names: tuple[str, ...]):
        self.feature_names = feature_names

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "MissingnessSelector":
        values = check_array(X, force_all_finite="allow-nan")
        if values.shape[1] != len(self.feature_names):
            raise ValueError("Input width does not match raw feature names")
        self.n_features_in_ = values.shape[1]
        self.feature_names_in_ = np.asarray(self.feature_names, dtype=object)
        self.missing_fraction_ = np.isnan(values).mean(axis=0)
        self.support_ = self.missing_fraction_ <= 0.5
        if not self.support_.any():
            raise ValueError("No features remain after training missingness selection")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "support_")
        values = check_array(X, force_all_finite="allow-nan")
        if values.shape[1] != self.n_features_in_:
            raise ValueError("Input width does not match fitted raw schema")
        return values[:, self.support_]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self, "support_")
        if input_features is not None and not np.array_equal(input_features, self.feature_names_in_):
            raise ValueError("Input feature order does not match the fitted raw schema")
        return self.feature_names_in_[self.support_]


def make_preprocessing(feature_names: tuple[str, ...]) -> Pipeline:
    """Fit this entire pipeline once on training measurements only."""
    return Pipeline([
        ("missingness", MissingnessSelector(feature_names)),
        ("imputer", SimpleImputer(strategy="median")),
        ("constants", VarianceThreshold(threshold=0.0)),
        ("scaler", StandardScaler()),
    ])
