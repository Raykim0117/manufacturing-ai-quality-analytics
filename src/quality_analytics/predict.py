"""Single-sample inference using only a trusted, already fitted bundle."""

from dataclasses import dataclass
import math
from pathlib import Path
import pickle
from typing import Any

import numpy as np

RAW_FEATURE_COUNT = 590
RAW_FEATURE_NAMES = tuple(f"feature_{index:03d}" for index in range(RAW_FEATURE_COUNT))


@dataclass(frozen=True)
class FrozenPredictor:
    model: Any
    preprocessing: Any
    threshold: float
    model_version: str
    dataset_version: str
    raw_feature_names: tuple[str, ...]
    positive_class_index: int

    @classmethod
    def load(cls, model_path: Path) -> "FrozenPredictor":
        # Pickle is trusted local configuration, never request-supplied content.
        with model_path.open("rb") as source:
            bundle = pickle.load(source)
        metadata = bundle["metadata"]
        names = tuple(metadata["raw_feature_names"])
        if names != RAW_FEATURE_NAMES or metadata["raw_feature_count"] != RAW_FEATURE_COUNT:
            raise ValueError("Frozen bundle must contain the ordered 590-feature raw schema")
        preprocessing = bundle["preprocessing"]
        retained = preprocessing.get_feature_names_out(names).tolist()
        if retained != metadata["retained_feature_names"]:
            raise ValueError("Frozen retained feature order differs from bundle metadata")
        threshold = float(bundle["threshold"])
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Frozen threshold must be finite and between zero and one")
        if threshold != metadata["threshold"]:
            raise ValueError("Frozen threshold differs from bundle metadata")
        classes = list(bundle["model"].classes_)
        if classes != [0, 1]:
            raise ValueError("Frozen model must use PASS=0 and FAIL=1")
        return cls(bundle["model"], preprocessing, threshold,
                   metadata["model_version"], metadata["dataset_version"],
                   names, classes.index(1))

    def score(self, features: list[float | None]) -> float:
        values = np.asarray(features, dtype=float).reshape(1, -1)
        if values.shape[1] != RAW_FEATURE_COUNT or np.isinf(values).any():
            raise ValueError("Measurements must match the finite/null raw feature schema")
        transformed = self.preprocessing.transform(values)
        failure_score = float(self.model.predict_proba(transformed)[0, self.positive_class_index])
        if not math.isfinite(failure_score) or not 0 <= failure_score <= 1:
            raise ValueError("Frozen model returned an invalid failure_score")
        return failure_score
