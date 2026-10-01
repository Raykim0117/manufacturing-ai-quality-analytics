"""Synthetic TEST evaluation guards and mocked BigQuery batch/report boundaries."""

from datetime import datetime
import hashlib
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from google.api_core.exceptions import NotFound
from google.cloud import bigquery
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quality_analytics.batch import evaluate_test_once, read_cached_batch, test_metrics
from quality_analytics.config import BigQueryConfig
from quality_analytics.dataset import dataset_from_rows
from quality_analytics.preprocessing import make_preprocessing
from quality_analytics.split import split_summary
from quality_analytics.warehouse import (
    batch_summary, persist_test_batch, prediction_schema, quality_summary_sql,
)
from test_baseline import sample_rows


class FrozenFixtureBooster:
    def save_raw(self, raw_format="ubj"):
        return b"synthetic frozen booster"


class FrozenFixtureModel:
    classes_ = np.array([0, 1])

    def fit(self, *args, **kwargs):
        raise AssertionError("Frozen model must not fit")

    def get_booster(self):
        return FrozenFixtureBooster()

    def predict_proba(self, X):
        scores = 1 / (1 + np.exp(-X[:, 0] / 100))
        return np.column_stack((1 - scores, scores))


def write_fixture(root: Path):
    dataset = dataset_from_rows(sample_rows())
    membership = {sid: "train" if i < 60 else "validation" if i < 80 else "test"
                  for i, sid in enumerate(dataset.sample_ids)}
    split = {
        "schema_version": 1, "dataset_version": dataset.dataset_version,
        "dataset_fingerprint": dataset.fingerprint, "random_state": 42,
        "fractions": {"train": 0.6, "validation": 0.2, "test": 0.2},
        "membership": membership, "counts": split_summary(dataset, membership),
    }
    split_path = root / "split.json"
    split_bytes = json.dumps(split).encode("utf-8")
    split_path.write_bytes(split_bytes)
    preprocessing = make_preprocessing(dataset.feature_names)[:-1].fit(dataset.features[:60])
    metadata = {
        "dataset_version": dataset.dataset_version, "dataset_fingerprint": dataset.fingerprint,
        "raw_feature_names": list(dataset.feature_names),
        "retained_feature_names": preprocessing.get_feature_names_out().tolist(),
        "model_version": "selected-model-v1", "selected_model": "xgboost_candidate_1",
        "preprocessing_fit_split": "train", "model_selection_split": "validation",
        "threshold_selection_split": "validation", "test_evaluated": False, "threshold": 0.25,
        "split_membership_sha256": hashlib.sha256(split_bytes).hexdigest(),
    }
    bundle = {"model": FrozenFixtureModel(), "preprocessing": preprocessing,
              "threshold": 0.25, "metadata": metadata}
    bundle_path = root / "selected.pkl"
    bundle_path.write_bytes(pickle.dumps(bundle))
    return dataset, bundle_path, split_path


class FinalEvaluationTests(unittest.TestCase):
    def test_test_scoring_is_once_only_and_freezes_ids_labels_threshold(self):
        original_predict = FrozenFixtureModel.predict_proba
        original_transform = Pipeline.transform
        scoring_inputs, transform_inputs = [], []

        def predict(model, X):
            scoring_inputs.append(X.copy())
            return original_predict(model, X)

        def transform(pipeline, X):
            transform_inputs.append(X.copy())
            return original_transform(pipeline, X)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, bundle, split = write_fixture(root)
            before = bundle.read_bytes(), split.read_bytes()
            with patch.object(FrozenFixtureModel, "predict_proba", new=predict), \
                    patch.object(Pipeline, "transform", new=transform), \
                    patch.object(Pipeline, "fit", side_effect=AssertionError("refit")), \
                    patch.object(Pipeline, "fit_transform", side_effect=AssertionError("refit")), \
                    patch("quality_analytics.split.create_membership", side_effect=AssertionError("regenerated")):
                report = evaluate_test_once(dataset, bundle, split, root / "state")
                with self.assertRaisesRegex(ValueError, "already claimed"):
                    evaluate_test_once(dataset, bundle, split, root / "state")
            self.assertEqual(len(scoring_inputs), 1)
            self.assertEqual(len(transform_inputs), 1)
            np.testing.assert_equal(transform_inputs[0], dataset.features[80:])
            self.assertEqual(report["test_scoring_calls"], 1)
            self.assertEqual(report["threshold"], 0.25)
            self.assertEqual(report["metrics_at_frozen_threshold"]["threshold"], 0.25)
            self.assertEqual(report["metrics_at_0_5_reference"]["threshold"], 0.5)
            self.assertEqual(report["metrics_at_frozen_threshold"]["evaluation_split"], "test")
            with patch("quality_analytics.batch.pickle.loads", side_effect=AssertionError("model loaded")):
                cached, rows = read_cached_batch(root / "state")
            self.assertEqual(cached, report)
            self.assertEqual([row["sample_id"] for row in rows], list(dataset.sample_ids[80:]))
            self.assertEqual([row["actual_failure"] for row in rows], dataset.labels[80:].tolist())
            self.assertTrue(all(row["source"] == "batch" and row["split"] == "test" for row in rows))
            self.assertTrue(all("failure_score" in row and "failure_probability" not in row for row in rows))
            self.assertTrue(all(datetime.fromisoformat(row["scored_at"]).utcoffset().total_seconds() == 0 for row in rows))
            self.assertEqual(before, (bundle.read_bytes(), split.read_bytes()))

    def test_claimed_evaluation_fails_closed_after_scoring_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, bundle, split = write_fixture(root)
            with patch.object(FrozenFixtureModel, "predict_proba", side_effect=RuntimeError("scoring failed")):
                with self.assertRaisesRegex(RuntimeError, "scoring failed"):
                    evaluate_test_once(dataset, bundle, split, root / "state")
            with self.assertRaisesRegex(ValueError, "already claimed"):
                evaluate_test_once(dataset, bundle, split, root / "state")

    def test_metrics_are_trapezoidal_and_cache_checksum_is_verified(self):
        metrics = test_metrics(np.array([0, 1, 0, 1]), np.array([0.1, 0.35, 0.4, 0.8]), 0.5)
        self.assertAlmostEqual(metrics["pr_auc"], 19 / 24)
        self.assertEqual(metrics["confusion_matrix"], [[2, 0], [1, 1]])
        self.assertEqual(metrics["failure_prevalence"], 0.5)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset, bundle, split = write_fixture(root)
            evaluate_test_once(dataset, bundle, split, root / "state")
            payload = root / "state/test_predictions.jsonl"
            payload.write_bytes(payload.read_bytes().replace(b'"failure_score":', b'"changed_score":'))
            with self.assertRaisesRegex(ValueError, "checksum"):
                read_cached_batch(root / "state")


class BatchWarehouseTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        dataset, bundle, split = write_fixture(root)
        evaluate_test_once(dataset, bundle, split, root / "state")
        _, self.rows = read_cached_batch(root / "state")
        self.config = BigQueryConfig("example-project", "secom_quality", "asia-northeast3", "secom_raw")
        self.client = Mock(spec=bigquery.Client)
        self.client.get_dataset.return_value.location = self.config.location
        self.client.create_table.side_effect = lambda table, exists_ok: table
        self.client.get_job.side_effect = NotFound("no existing job")
        self.job = self.client.load_table_from_json.return_value
        self.job.output_rows = len(self.rows)
        self.persisted = []
        self.job.result.side_effect = lambda: self.persisted.extend(self.rows)
        view = Mock()
        view.table_type = "VIEW"
        view.view_query = quality_summary_sql(self.config, self.rows[0]["run_id"])
        self.client.get_table.return_value = view

        def query(sql, **kwargs):
            job = Mock()
            if "WHERE run_id = @run_id" in sql:
                job.result.return_value = self.persisted.copy()
            elif sql.startswith("SELECT *"):
                job.result.return_value = [batch_summary(self.rows)]
            else:
                job.result.return_value = []
            return job

        self.client.query.side_effect = query

    def test_prediction_schema_matches_required_fields_and_sql(self):
        fields = prediction_schema()
        expected = ["prediction_id", "sample_id", "run_id", "model_version", "dataset_version",
                    "source", "split", "scored_at", "failure_score", "predicted_failure",
                    "threshold", "actual_failure"]
        self.assertEqual([field.name for field in fields], expected)
        self.assertEqual(fields[7].field_type, "TIMESTAMP")
        self.assertEqual(fields[8].field_type, "FLOAT64")
        self.assertEqual(fields[9].field_type, "BOOL")
        self.assertEqual(fields[11].field_type, "INT64")
        self.assertEqual(fields[11].mode, "NULLABLE")
        self.assertEqual(fields[6].mode, "NULLABLE")
        sql = (ROOT / "sql/create_predictions_table.sql").read_text(encoding="utf-8")
        for name in expected:
            self.assertIn(name, sql)

    def test_append_preserves_rows_and_retries_do_not_duplicate_or_score(self):
        first = persist_test_batch(self.client, self.config, self.rows)
        self.assertEqual(first["rows_written"], 20)
        self.assertEqual(first["rows_verified"], 20)
        call = self.client.load_table_from_json.call_args
        self.assertEqual(call.args, (self.rows, self.config.predictions_table_id))
        self.assertEqual(call.kwargs["job_config"].write_disposition, "WRITE_APPEND")
        self.assertEqual(call.kwargs["job_config"].create_disposition, "CREATE_NEVER")
        self.assertFalse(call.kwargs["job_config"].autodetect)
        self.assertEqual(call.kwargs["job_id"], first["load_job_id"])
        self.assertEqual(first["summary"], batch_summary(self.rows))
        self.client.create_dataset.assert_not_called()
        with patch.object(FrozenFixtureModel, "predict_proba", side_effect=AssertionError("rescored")):
            repeated = persist_test_batch(self.client, self.config, self.rows)
        self.assertEqual(repeated["rows_written"], 0)
        self.assertEqual(repeated["rows_verified"], 20)
        self.client.load_table_from_json.assert_called_once()

    def test_reporting_view_filters_exact_run_source_and_split(self):
        sql = quality_summary_sql(self.config, self.rows[0]["run_id"])
        self.assertIn(f"WHERE run_id = '{self.rows[0]['run_id']}'", sql)
        self.assertIn("source = 'batch'", sql)
        self.assertIn("split = 'test'", sql)
        self.assertIn("actual_failure IS NOT NULL", sql)
        self.assertIn("GROUP BY run_id", sql)
        self.assertNotIn("MAX(scored_at)", sql)
        self.assertNotIn("yield", sql.split("SELECT", 1)[1])
        for name in batch_summary(self.rows):
            self.assertIn(name, sql)
        with self.assertRaises(ValueError):
            quality_summary_sql(self.config, "unsafe'; DROP TABLE x")

    def test_validation_rows_and_raw_destination_cannot_be_persisted(self):
        wrong = [dict(row) for row in self.rows]
        wrong[0]["split"] = "validation"
        with self.assertRaisesRegex(ValueError, "split=test"):
            persist_test_batch(self.client, self.config, wrong)
        collision = BigQueryConfig("example-project", "secom_quality", "asia-northeast3",
                                   "secom_raw", predictions_table="secom_raw")
        with self.assertRaisesRegex(ValueError, "raw table"):
            persist_test_batch(self.client, collision, self.rows)
        self.client.create_table.assert_not_called()
        self.client.load_table_from_json.assert_not_called()

    def test_persisted_label_mismatch_and_load_failure_are_reported(self):
        self.persisted = [dict(row) for row in self.rows]
        self.persisted[0]["actual_failure"] = 1 - self.persisted[0]["actual_failure"]
        with self.assertRaisesRegex(RuntimeError, "actual_failure"):
            persist_test_batch(self.client, self.config, self.rows)
        self.persisted = []
        self.job.result.side_effect = RuntimeError("load failed")
        with self.assertRaisesRegex(RuntimeError, "load failed"):
            persist_test_batch(self.client, self.config, self.rows)


if __name__ == "__main__":
    unittest.main()
