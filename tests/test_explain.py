"""Check frozen state, original feature IDs, SHAP dimensions and split isolation."""

from dataclasses import replace
import json
from pathlib import Path
import pickle
import struct
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quality_analytics import explain
from quality_analytics.dataset import read_local_dataset
from quality_analytics.split import load_persisted_split


class OfflineShapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = read_local_dataset(ROOT / "data/normalized/secom.jsonl")
        artifact_root = ROOT / "artifacts" / cls.dataset.dataset_version
        cls.bundle_path = artifact_root / "selected-model-v1/selected.pkl"
        cls.split_path = artifact_root / "logistic-regression-v1/split_membership.json"
        cls.bundle_bytes = cls.bundle_path.read_bytes()
        cls.bundle = pickle.loads(cls.bundle_bytes)
        cls.membership = load_persisted_split(cls.dataset, cls.split_path)

    def test_samples_are_fixed_capped_and_permitted_splits_only(self):
        first = explain.fixed_indices(self.dataset, self.membership, "validation", 100)
        repeated = explain.fixed_indices(self.dataset, self.membership, "validation", 100)
        np.testing.assert_array_equal(first, repeated)
        self.assertEqual(len(first), 100)
        self.assertEqual(len(set(first)), 100)
        self.assertTrue(all(self.membership[self.dataset.sample_ids[i]] == "validation" for i in first))
        background = explain.fixed_indices(self.dataset, self.membership, "train", 64)
        self.assertTrue(all(self.membership[self.dataset.sample_ids[i]] == "train" for i in background))
        self.assertFalse(set(first) & set(background))
        for split, cap in (("test", 5), ("validation", 101), ("train", 65), ("validation", 0)):
            with self.subTest(split=split, cap=cap), self.assertRaises(ValueError):
                explain.fixed_indices(self.dataset, self.membership, split, cap)

    def test_frozen_model_preprocessing_and_split_are_reused_without_fitting(self):
        original_transform = Pipeline.transform
        original_call = explain.shap.TreeExplainer.__call__
        transformed_inputs, explained_inputs = [], []

        def record_transform(pipeline, X):
            transformed_inputs.append(X.copy())
            return original_transform(pipeline, X)

        def record_call(explainer, X, **kwargs):
            explained_inputs.append(X.copy())
            return original_call(explainer, X, **kwargs)

        split_before = self.split_path.read_bytes(), self.split_path.stat().st_mtime_ns
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(XGBClassifier, "fit", side_effect=AssertionError("model retrained")), \
                patch.object(Pipeline, "fit", side_effect=AssertionError("preprocessing refitted")), \
                patch.object(Pipeline, "fit_transform", side_effect=AssertionError("preprocessing refitted")), \
                patch("quality_analytics.split.create_membership", side_effect=AssertionError("split regenerated")), \
                patch.object(Pipeline, "transform", new=record_transform), \
                patch.object(explain.shap.TreeExplainer, "__call__", new=record_call):
            result = explain.explain_validation(self.dataset, self.bundle_path, self.split_path,
                                                Path(temporary), sample_size=12, background_size=8)
            summary = (Path(temporary) / "shap_summary.png").read_bytes()
            self.assertEqual(summary[:8], b"\x89PNG\r\n\x1a\n")
            width, height = struct.unpack(">II", summary[16:24])
            self.assertGreater(width, 500)
            self.assertGreater(height, 500)
            saved = json.loads((Path(temporary) / "shap_feature_importance.json").read_text(encoding="utf-8"))
            self.assertEqual(saved, result["global"])
        background = explain.fixed_indices(self.dataset, self.membership, "train", 8)
        validation = explain.fixed_indices(self.dataset, self.membership, "validation", 12)
        self.assertEqual(len(transformed_inputs), 2)
        np.testing.assert_equal(transformed_inputs[0], self.dataset.features[background])
        np.testing.assert_equal(transformed_inputs[1], self.dataset.features[validation])
        self.assertEqual(len(explained_inputs), 1)
        self.assertEqual(explained_inputs[0].shape, (12, 450))
        self.assertEqual(result["global"]["shap_shape"], [12, 450])
        self.assertEqual(result["global"]["feature_names"], self.bundle["metadata"]["retained_feature_names"])
        self.assertEqual(len(result["global"]["feature_importance"]), 450)
        self.assertEqual(len(result["global"]["top_20_features"]), 20)
        means = [item["mean_abs_shap"] for item in result["global"]["feature_importance"]]
        self.assertEqual(means, sorted(means, reverse=True))
        self.assertFalse(result["global"]["test_evaluated"])
        local = result["local"]
        self.assertEqual(self.membership[local["sample_id"]], "validation")
        self.assertGreaterEqual(local["failure_score"], self.bundle["threshold"])
        self.assertEqual(local["feature_order"], self.bundle["metadata"]["retained_feature_names"])
        self.assertEqual(len(local["feature_contributions"]), 450)
        self.assertAlmostEqual(local["base_value"] + local["shap_value_sum"],
                               local["model_raw_margin"], places=5)
        self.assertEqual(self.bundle_path.read_bytes(), self.bundle_bytes)
        self.assertEqual(split_before, (self.split_path.read_bytes(), self.split_path.stat().st_mtime_ns))

    def test_feature_order_mismatch_is_rejected_before_transform(self):
        bundle = pickle.loads(self.bundle_bytes)
        bundle["metadata"]["retained_feature_names"] = list(reversed(bundle["metadata"]["retained_feature_names"]))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "misordered.pkl"
            path.write_bytes(pickle.dumps(bundle))
            with patch.object(Pipeline, "transform", side_effect=AssertionError("transformed")), \
                    self.assertRaisesRegex(ValueError, "Feature order"):
                explain.explain_validation(self.dataset, path, self.split_path, Path(temporary), 12, 8)

    def test_bad_shap_dimensions_are_rejected(self):
        fake_explainer = Mock()
        fake_explainer.return_value = Mock(values=np.zeros((12, 449)))
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            explain.shap, "TreeExplainer", return_value=fake_explainer
        ), self.assertRaisesRegex(ValueError, "SHAP dimensions"):
            explain.explain_validation(self.dataset, self.bundle_path, self.split_path, Path(temporary), 12, 8)

    def test_test_rows_do_not_affect_explanations(self):
        features = self.dataset.features.copy()
        test = np.array([self.membership[sid] == "test" for sid in self.dataset.sample_ids])
        # Poison test measurements after identity validation; any accidental
        # preprocessing or scoring of them fails, without changing real data.
        features[test] = np.inf
        poisoned = replace(self.dataset, features=features)
        with tempfile.TemporaryDirectory() as temporary:
            first = explain.explain_validation(self.dataset, self.bundle_path, self.split_path,
                                                Path(temporary) / "first", 12, 8)
            second = explain.explain_validation(poisoned, self.bundle_path, self.split_path,
                                                 Path(temporary) / "second", 12, 8)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
