"""Critical split, preprocessing, metric, and baseline persistence checks."""

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quality_analytics.dataset import dataset_from_rows, read_local_dataset
from quality_analytics.preprocessing import make_preprocessing
from quality_analytics.split import create_membership, load_or_create_split, split_summary
from quality_analytics.train import evaluate_validation, train_baseline


def sample_rows() -> list[dict]:
    return [{
        "dataset_version": "fixture-v1", "sample_id": f"fixture-v1:{i:04d}",
        "test_time": "2008-07-19 11:55:00", "failure_label": int(i % 5 == 0),
        "feature_000": float(i), "feature_001": None if i % 7 == 0 else float(i % 9),
        "feature_002": float(i % 3),
    } for i in range(1, 101)]


class DatasetTests(unittest.TestCase):
    def test_local_and_bigquery_rows_have_same_schema_and_fingerprint(self):
        rows = sample_rows()
        cloud_rows = deepcopy(rows)
        for row in cloud_rows:
            row["test_time"] = datetime.fromisoformat(row["test_time"])
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "raw.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            local = read_local_dataset(path)
        cloud = dataset_from_rows(list(reversed(cloud_rows)))
        self.assertEqual(local.fingerprint, cloud.fingerprint)
        self.assertEqual(local.sample_ids, cloud.sample_ids)
        self.assertEqual(local.feature_names, ("feature_000", "feature_001", "feature_002"))
        np.testing.assert_equal(local.features, cloud.features)
        self.assertEqual(local.features.shape, (100, 3))

    def test_invalid_normalized_rows_are_rejected(self):
        for replacement in ({"failure_label": -1}, {"feature_000": float("inf")},
                            {"feature_000": float("nan")}, {"feature_000": True},
                            {"test_time": "2008-07-19T11:55:00+09:00"},
                            {"dataset_version": "other-version"}):
            rows = sample_rows()
            rows[0].update(replacement)
            with self.subTest(replacement=replacement), self.assertRaises(ValueError):
                dataset_from_rows(rows)
        rows = sample_rows()
        rows[0]["sample_id"] = rows[1]["sample_id"]
        with self.assertRaisesRegex(ValueError, "unique"):
            dataset_from_rows(rows)


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.dataset = dataset_from_rows(sample_rows())

    def test_deterministic_split_ignores_read_order(self):
        first = create_membership(self.dataset)
        self.assertEqual(first, create_membership(self.dataset))
        self.assertEqual(first, create_membership(dataset_from_rows(list(reversed(sample_rows())))))

    def test_stratification_and_no_overlap(self):
        membership = create_membership(self.dataset)
        summary = split_summary(self.dataset, membership)
        self.assertEqual(summary, {"train": {"rows": 60, "fail": 12, "pass": 48},
                                   "validation": {"rows": 20, "fail": 4, "pass": 16},
                                   "test": {"rows": 20, "fail": 4, "pass": 16}})
        groups = [{sid for sid, split in membership.items() if split == name}
                  for name in ("train", "validation", "test")]
        self.assertEqual(set.union(*groups), set(self.dataset.sample_ids))
        for i, group in enumerate(groups):
            for other in groups[i + 1:]:
                self.assertFalse(group & other)

    def test_persisted_membership_is_reused_and_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "split.json"
            first = load_or_create_split(self.dataset, path)
            before = path.read_bytes(), path.stat().st_mtime_ns
            self.assertEqual(first, load_or_create_split(self.dataset, path))
            self.assertEqual(before, (path.read_bytes(), path.stat().st_mtime_ns))
            changed = sample_rows()
            changed[0]["feature_000"] = 999.0
            with self.assertRaisesRegex(ValueError, "Persisted split"):
                load_or_create_split(dataset_from_rows(changed), path)
            saved = json.loads(path.read_text(encoding="utf-8"))
            saved["membership"][self.dataset.sample_ids[0]] = "unknown"
            path.write_text(json.dumps(saved), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_or_create_split(self.dataset, path)


class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.names = tuple(f"feature_{i:03d}" for i in range(5))
        self.train = np.array([
            [np.nan, np.nan, 1, 7, 10],
            [np.nan, np.nan, np.nan, 7, 20],
            [np.nan, np.nan, 3, 7, 30],
            [np.nan, 9, np.nan, 7, 40],
        ])

    def test_train_only_missingness_medians_constants_and_scaling(self):
        pipeline = make_preprocessing(self.names)
        transformed = pipeline.fit_transform(self.train)
        self.assertEqual(pipeline.get_feature_names_out().tolist(), ["feature_002", "feature_004"])
        np.testing.assert_array_equal(pipeline.named_steps["missingness"].support_,
                                      [False, False, True, True, True])
        np.testing.assert_array_equal(pipeline.named_steps["imputer"].statistics_, [2, 7, 25])
        np.testing.assert_array_equal(pipeline.named_steps["scaler"].mean_, [2, 25])
        np.testing.assert_allclose(transformed.mean(axis=0), [0, 0], atol=1e-15)
        np.testing.assert_allclose(transformed.std(axis=0), [1, 1])
        before = pickle.dumps(pipeline)
        # Held-out columns reverse training missingness/constant patterns.
        validation = np.array([[100, 200, np.nan, 999, 1000], [101, 201, 4, 1000, 2000]])
        result = pipeline.transform(validation)
        self.assertEqual(result.shape, (2, 2))
        self.assertEqual(result[0, 0], 0.0)  # Uses training median 2.
        self.assertEqual(pickle.dumps(pipeline), before)

    def test_feature_order_is_preserved_and_mismatches_are_rejected(self):
        pipeline = make_preprocessing(self.names).fit(self.train)
        self.assertEqual(pipeline.get_feature_names_out(self.names).tolist(),
                         ["feature_002", "feature_004"])
        with self.assertRaisesRegex(ValueError, "feature order"):
            pipeline.get_feature_names_out(tuple(reversed(self.names)))
        with self.assertRaisesRegex(ValueError, "width"):
            pipeline.transform(np.ones((2, 4)))

    def test_no_usable_features_fails(self):
        for values in (np.full((4, 5), np.nan), np.ones((4, 5))):
            with self.subTest(values=values), self.assertRaises(ValueError):
                make_preprocessing(self.names).fit(values)


class BaselineTests(unittest.TestCase):
    def test_primary_metric_is_trapezoidal_pr_auc(self):
        labels = np.array([0, 1, 0, 1])
        scores = np.array([0.1, 0.4, 0.35, 0.8])
        metrics = evaluate_validation(labels, scores)
        self.assertAlmostEqual(metrics["pr_auc"], 1.0)
        # A ranking with imperfect precision distinguishes AP from trapezoids.
        scores = np.array([0.1, 0.35, 0.4, 0.8])
        metrics = evaluate_validation(labels, scores)
        self.assertAlmostEqual(metrics["pr_auc"], 19 / 24)
        self.assertNotAlmostEqual(metrics["pr_auc"], average_precision_score(labels, scores))
        self.assertEqual(metrics["confusion_matrix"], [[2, 0], [1, 1]])
        self.assertEqual(metrics["threshold"], 0.5)
        zero = evaluate_validation(labels, np.array([0.1, 0.2, 0.3, 0.4]))
        self.assertEqual(zero["precision"], 0)

    def test_only_train_is_fitted_validation_transformed_and_test_untouched(self):
        dataset = dataset_from_rows(sample_rows())
        membership = create_membership(dataset)
        train = np.array([membership[sid] == "train" for sid in dataset.sample_ids])
        validation = np.array([membership[sid] == "validation" for sid in dataset.sample_ids])
        original_fit = Pipeline.fit_transform
        original_transform = Pipeline.transform
        fit_calls, transform_calls = [], []

        def record_fit(pipeline, X):
            fit_calls.append(X.copy())
            return original_fit(pipeline, X)

        def record_transform(pipeline, X):
            transform_calls.append(X.copy())
            return original_transform(pipeline, X)

        # Plain class-method wrappers preserve sklearn descriptors and pickling.
        with tempfile.TemporaryDirectory() as temporary, patch.object(
            Pipeline, "fit_transform", new=record_fit,
        ), patch.object(Pipeline, "transform", new=record_transform):
            train_baseline(dataset, Path(temporary), {"backend": "test"})
            self.assertEqual(len(fit_calls), 1)
            np.testing.assert_equal(fit_calls[0], dataset.features[train])
            # Validation is transformed before and after artifact reload.
            self.assertEqual(len(transform_calls), 2)
            for values in transform_calls:
                np.testing.assert_equal(values, dataset.features[validation])

    def test_artifact_round_trip_and_held_out_mutations_do_not_affect_fitting(self):
        dataset = dataset_from_rows(sample_rows())
        membership = create_membership(dataset)
        changed_rows = sample_rows()
        for row in changed_rows:
            if membership[row["sample_id"]] != "train":
                row["feature_000"] = 1000000.0
                row["feature_001"] = None
                row["feature_002"] = -999.0
        with tempfile.TemporaryDirectory() as temporary:
            first_dir, second_dir = Path(temporary) / "first", Path(temporary) / "second"
            result = train_baseline(dataset, first_dir, {"backend": "test"})
            train_baseline(dataset_from_rows(changed_rows), second_dir, {"backend": "test"})
            with (first_dir / "baseline.pkl").open("rb") as source:
                first = pickle.load(source)
            with (second_dir / "baseline.pkl").open("rb") as source:
                second = pickle.load(source)
            np.testing.assert_array_equal(first["model"].coef_, second["model"].coef_)
            np.testing.assert_array_equal(first["preprocessing"].named_steps["imputer"].statistics_,
                                          second["preprocessing"].named_steps["imputer"].statistics_)
            np.testing.assert_array_equal(first["preprocessing"].named_steps["scaler"].mean_,
                                          second["preprocessing"].named_steps["scaler"].mean_)
            validation = np.array([membership[sid] == "validation" for sid in dataset.sample_ids])
            scores = first["model"].predict_proba(
                first["preprocessing"].transform(dataset.features[validation]))[:, 1]
            self.assertEqual(evaluate_validation(dataset.labels[validation], scores),
                             result["validation_metrics"])
            self.assertTrue(first["metadata"]["converged"])
            self.assertFalse(first["metadata"]["test_evaluated"])
            self.assertEqual(first["model"].class_weight, "balanced")
            for name in ("split_membership.json", "retained_features.json", "metadata.json",
                         "validation_metrics.json"):
                self.assertTrue((first_dir / name).is_file())


if __name__ == "__main__":
    unittest.main()
