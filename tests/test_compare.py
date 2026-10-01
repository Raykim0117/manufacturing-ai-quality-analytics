"""Guard CPU determinism, persisted split reuse, and validation-only selection."""

from dataclasses import replace
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quality_analytics import compare
from quality_analytics.dataset import dataset_from_rows
from quality_analytics.preprocessing import MissingnessSelector
from quality_analytics.split import load_persisted_split
from quality_analytics.train import evaluate_validation, train_baseline
from test_baseline import sample_rows


class ChallengerTests(unittest.TestCase):
    def test_class_weight_and_cpu_training_are_deterministic(self):
        labels = np.array([0] * 16 + [1] * 4)
        features = np.column_stack((np.arange(20), np.arange(20) % 3)).astype(float)
        first, parameters = compare.fit_candidate(features, labels, compare.CANDIDATE_CONFIGS[0])
        second, repeated = compare.fit_candidate(features.copy(), labels.copy(),
                                                compare.CANDIDATE_CONFIGS[0])
        self.assertEqual(parameters["scale_pos_weight"], 4.0)
        self.assertEqual(parameters["device"], "cpu")
        self.assertEqual(parameters["n_jobs"], 1)
        self.assertEqual(parameters["random_state"], 42)
        self.assertEqual(parameters, repeated)
        np.testing.assert_array_equal(first.predict_proba(features), second.predict_proba(features))
        self.assertEqual(first.get_booster().get_dump(), second.get_booster().get_dump())
        with self.assertRaises(ValueError):
            compare.training_class_weight(np.zeros(20))
        with self.assertRaisesRegex(ValueError, "predefined"):
            compare.fit_candidate(features, labels, {"max_depth": 10, "n_estimators": 1000})

    def test_model_selection_uses_validation_pr_auc_and_lr_wins_exact_ties(self):
        metrics = {
            "xgboost_candidate_1": {"evaluation_split": "validation", "pr_auc": 0.2},
            "logistic_regression": {"evaluation_split": "validation", "pr_auc": 0.2},
            "xgboost_candidate_2": {"evaluation_split": "validation", "pr_auc": 0.1},
        }
        self.assertEqual(compare.select_model(metrics), "logistic_regression")
        metrics["xgboost_candidate_2"]["pr_auc"] = 0.3
        self.assertEqual(compare.select_model(metrics), "xgboost_candidate_2")
        metrics["xgboost_candidate_2"]["evaluation_split"] = "test"
        with self.assertRaisesRegex(ValueError, "validation-only"):
            compare.select_model(metrics)


class ThresholdTests(unittest.TestCase):
    def test_threshold_maximizes_precision_under_recall_constraint(self):
        labels = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        scores = np.array([0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50])
        chosen = compare.choose_threshold(labels, scores)
        self.assertEqual(chosen, 0.65)
        feasible = []
        for threshold in np.unique(scores):
            predicted = scores >= threshold
            recall = recall_score(labels, predicted)
            if recall >= 0.8:
                feasible.append((precision_score(labels, predicted), recall, threshold))
        self.assertEqual(chosen, max(feasible)[2])
        self.assertEqual(recall_score(labels, scores >= chosen), 0.8)

    def test_threshold_ties_favor_higher_recall_then_higher_threshold(self):
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 1])
        scores = np.arange(10, 0, -1, dtype=float) / 10
        # Precision .5 ties at recall .8 and 1.0; choose recall 1.0.
        self.assertEqual(compare.choose_threshold(labels, scores), 0.1)
        with patch.object(compare, "precision_recall_curve", return_value=(
            np.array([0.5, 0.5, 0.5, 1.0]), np.array([0.8, 0.9, 0.9, 0.0]),
            np.array([0.1, 0.2, 0.3]),
        )):
            self.assertEqual(compare.choose_threshold(labels, scores), 0.3)

    def test_threshold_handles_tied_scores_and_rejects_invalid_inputs(self):
        labels = np.array([0, 1, 0, 1])
        scores = np.array([0.0, 0.0, 1.0, 1.0])
        threshold = compare.choose_threshold(labels, scores)
        self.assertEqual(threshold, 0.0)
        self.assertEqual(recall_score(labels, scores >= threshold), 1.0)
        for invalid in (np.array([0.1, 0.2]), np.array([0.1, np.nan, 0.3, 0.4]),
                        np.array([0.1, 0.2, 0.3, 1.1])):
            with self.subTest(scores=invalid), self.assertRaises(ValueError):
                compare.choose_threshold(labels, invalid)


class ComparisonIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.dataset = dataset_from_rows(sample_rows())
        cls.baseline_dir = cls.root / "baseline"
        train_baseline(cls.dataset, cls.baseline_dir, {"backend": "fixture"})
        cls.membership = load_persisted_split(cls.dataset, cls.baseline_dir / "split_membership.json")
        cls.train = np.array([cls.membership[sid] == "train" for sid in cls.dataset.sample_ids])
        cls.validation = np.array([cls.membership[sid] == "validation" for sid in cls.dataset.sample_ids])
        cls.test = np.array([cls.membership[sid] == "test" for sid in cls.dataset.sample_ids])

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_existing_split_is_read_without_regeneration_and_missing_split_fails(self):
        path = self.baseline_dir / "split_membership.json"
        before = path.read_bytes(), path.stat().st_mtime_ns
        with patch("quality_analytics.split.create_membership", side_effect=AssertionError("regenerated")):
            self.assertEqual(load_persisted_split(self.dataset, path), self.membership)
        self.assertEqual(before, (path.read_bytes(), path.stat().st_mtime_ns))
        missing = self.root / "missing.json"
        with self.assertRaises(FileNotFoundError):
            load_persisted_split(self.dataset, missing)
        self.assertFalse(missing.exists())

    def test_train_weight_validation_selection_and_preprocessing_are_isolated(self):
        split_path = self.baseline_dir / "split_membership.json"
        baseline_path = self.baseline_dir / "baseline.pkl"
        before = split_path.read_bytes(), baseline_path.read_bytes(), split_path.stat().st_mtime_ns
        transformed_inputs = []
        original_transform = Pipeline.transform

        def record_transform(pipeline, X):
            transformed_inputs.append(X.copy())
            return original_transform(pipeline, X)

        with patch("quality_analytics.split.create_membership", side_effect=AssertionError("regenerated")), \
                patch.object(Pipeline, "fit", side_effect=AssertionError("refitted pipeline")), \
                patch.object(MissingnessSelector, "fit", side_effect=AssertionError("refitted selection")), \
                patch.object(SimpleImputer, "fit", side_effect=AssertionError("refitted imputer")), \
                patch.object(Pipeline, "transform", new=record_transform), \
                patch.object(compare, "training_class_weight", wraps=compare.training_class_weight) as weight, \
                patch.object(compare, "fit_candidate", wraps=compare.fit_candidate) as fit, \
                patch.object(compare, "choose_threshold", wraps=compare.choose_threshold) as threshold, \
                patch.object(compare, "select_model", wraps=compare.select_model) as selection:
            report = compare.compare_validation(self.dataset, self.baseline_dir,
                                                self.root / "isolation", {"backend": "fixture"})
        for call in weight.call_args_list:
            np.testing.assert_equal(call.args[0], self.dataset.labels[self.train])
        self.assertEqual(fit.call_count, 2)
        with baseline_path.open("rb") as source:
            baseline = pickle.load(source)
        expected_features = baseline["preprocessing"][:-1].transform(self.dataset.features[self.train])
        for call in fit.call_args_list:
            np.testing.assert_equal(call.args[0], expected_features)
            np.testing.assert_equal(call.args[1], self.dataset.labels[self.train])
        threshold.assert_called_once()
        np.testing.assert_equal(threshold.call_args.args[0], self.dataset.labels[self.validation])
        self.assertEqual(len(threshold.call_args.args[1]), 20)
        selection.assert_called_once()
        for metric in selection.call_args.args[0].values():
            self.assertEqual(metric["evaluation_split"], "validation")
            self.assertEqual(metric["rows"], 20)
            self.assertEqual(metric["fail"], 4)
        self.assertEqual(len(transformed_inputs), 4)
        np.testing.assert_equal(transformed_inputs[0], self.dataset.features[self.train])
        for values in transformed_inputs[1:]:
            np.testing.assert_equal(values, self.dataset.features[self.validation])
        self.assertEqual(report["scale_pos_weight"], 48 / 12)
        self.assertFalse(report["test_evaluated"])
        self.assertEqual(before, (split_path.read_bytes(), baseline_path.read_bytes(),
                                  split_path.stat().st_mtime_ns))

    def test_test_measurements_cannot_affect_selection_or_threshold(self):
        features = self.dataset.features.copy()
        features[self.test] = np.inf
        # Poison only test inputs after dataset identity validation; any accidental
        # transform or prediction over those rows would fail on infinity.
        poisoned = replace(self.dataset, features=features)
        first = compare.compare_validation(self.dataset, self.baseline_dir,
                                           self.root / "original", {"backend": "fixture"})
        second = compare.compare_validation(poisoned, self.baseline_dir,
                                            self.root / "poisoned", {"backend": "fixture"})
        self.assertEqual(first, second)

    def test_saved_selected_bundle_reproduces_scores_and_threshold_decisions(self):
        output = self.root / "roundtrip"
        report = compare.compare_validation(self.dataset, self.baseline_dir, output,
                                            {"backend": "fixture"})
        with (output / "selected.pkl").open("rb") as source:
            bundle = pickle.load(source)
        scores = compare.failure_scores(bundle["model"], bundle["preprocessing"].transform(
            self.dataset.features[self.validation]))
        metrics = evaluate_validation(self.dataset.labels[self.validation], scores,
                                      bundle["threshold"], threshold_optimized=True)
        self.assertEqual(metrics, report["selected_validation_metrics"])
        self.assertEqual(bundle["threshold"], report["selected_threshold"])
        self.assertGreaterEqual(metrics["recall"], 0.80)
        self.assertEqual(bundle["preprocessing"].get_feature_names_out().tolist(),
                         bundle["metadata"]["retained_feature_names"])
        self.assertEqual(bundle["metadata"]["model_selection_split"], "validation")
        self.assertEqual(bundle["metadata"]["threshold_selection_split"], "validation")
        self.assertFalse(bundle["metadata"]["preprocessing_refitted"])
        self.assertFalse(bundle["metadata"]["test_evaluated"])
        for name in ("metadata.json", "retained_features.json", "comparison_validation.json",
                     "xgboost_candidate_1.pkl", "xgboost_candidate_2.pkl"):
            self.assertTrue((output / name).is_file())
        before = (output / "selected.pkl").read_bytes()
        with self.assertRaisesRegex(ValueError, "frozen"):
            compare.compare_validation(self.dataset, self.baseline_dir, output, {})
        self.assertEqual(before, (output / "selected.pkl").read_bytes())


if __name__ == "__main__":
    unittest.main()
