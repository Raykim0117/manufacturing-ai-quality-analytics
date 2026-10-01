"""API checks use synthetic requests and the saved bundle, never dataset rows."""

from datetime import datetime
from contextlib import ExitStack
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from uuid import UUID

import numpy as np
from fastapi.testclient import TestClient
from google.cloud import bigquery
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quality_analytics.api import create_app
from quality_analytics.config import BigQueryConfig, InferenceConfig
from quality_analytics.predict import FrozenPredictor, RAW_FEATURE_NAMES
from quality_analytics.warehouse import OnlinePredictionStore, prediction_schema

BUNDLE_PATH = ROOT / "artifacts/secom-01b3bed261fc1223/selected-model-v1/selected.pkl"
CLOUD_ENV = {
    "GCP_PROJECT_ID": "example-project", "BQ_DATASET": "secom_quality",
    "BQ_LOCATION": "asia-northeast3", "BQ_RAW_TABLE": "secom_raw",
    "BQ_PREDICTIONS_TABLE": "quality_predictions",
}


class InferenceApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original_bundle_bytes = BUNDLE_PATH.read_bytes()
        cls.bundle = pickle.loads(cls.original_bundle_bytes)
        cls.threshold = cls.bundle["threshold"]

    def setUp(self):
        self.app = create_app(InferenceConfig(BUNDLE_PATH))
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.client = stack.enter_context(TestClient(self.app))

    def request(self, features=None, **changes):
        payload = {"sample_id": "demo-001", "features": [0.0] * 590 if features is None else features}
        payload.update(changes)
        return self.client.post("/predict", json=payload)

    def test_health_and_model_info_do_not_use_bigquery(self):
        with patch("quality_analytics.warehouse.bigquery.Client", side_effect=AssertionError("credentials")):
            health = self.client.get("/health")
            info = self.client.get("/model-info")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json(), {"status": "ok", "model_version": "selected-model-v1"})
        self.assertEqual(info.status_code, 200)
        self.assertEqual(info.json()["raw_feature_count"], 590)
        self.assertEqual(info.json()["raw_feature_names"], list(RAW_FEATURE_NAMES))
        self.assertEqual(info.json()["positive_class"]["label"], 1)
        self.assertIn("FAIL", info.json()["positive_class"]["meaning"])
        self.assertEqual(info.json()["threshold"], self.threshold)

    def test_valid_prediction_and_unique_ids(self):
        response = self.request(features=list(range(590)))
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(set(result), {"prediction_id", "sample_id", "failure_score",
                                     "predicted_failure", "threshold", "model_version", "stored"})
        UUID(result["prediction_id"])
        self.assertEqual(result["sample_id"], "demo-001")
        self.assertTrue(0 <= result["failure_score"] <= 1)
        self.assertEqual(result["predicted_failure"], result["failure_score"] >= self.threshold)
        self.assertEqual(result["threshold"], self.threshold)
        self.assertFalse(result["stored"])
        self.assertNotEqual(result["prediction_id"], self.request().json()["prediction_id"])

    def test_wrong_length_is_rejected_before_scoring(self):
        with patch.object(FrozenPredictor, "score", side_effect=AssertionError("invalid scored")):
            for length in (0, 589, 591):
                with self.subTest(length=length):
                    self.assertEqual(self.request(features=[0] * length).status_code, 422)

    def test_nonfinite_and_nonnumeric_values_are_rejected(self):
        with patch.object(FrozenPredictor, "score", side_effect=AssertionError("invalid scored")):
            for value in (float("nan"), float("inf"), -float("inf"), "NaN", "1", True, {}):
                with self.subTest(value=value):
                    features = [0.0] * 590
                    features[47] = value
                    response = self.client.post(
                        "/predict", content=json.dumps({"sample_id": "demo", "features": features}),
                        headers={"Content-Type": "application/json"},
                    )
                    self.assertEqual(response.status_code, 422)

    def test_empty_ids_extra_fields_and_missing_fields_are_rejected(self):
        for sample_id in ("", "  \t", None, 123):
            with self.subTest(sample_id=sample_id):
                self.assertEqual(self.request(sample_id=sample_id).status_code, 422)
        for field in ("threshold", "actual_failure", "label", "unexpected"):
            with self.subTest(field=field):
                self.assertEqual(self.request(**{field: 1}).status_code, 422)
        self.assertEqual(self.client.post("/predict", json={"sample_id": "demo"}).status_code, 422)
        self.assertEqual(self.client.post("/predict", json={"features": [0] * 590}).status_code, 422)

    def test_nulls_use_frozen_medians_and_saved_bundle_scores_match(self):
        preprocessing = self.bundle["preprocessing"]
        for features in ([None] * 590, [float(index) / 10 for index in range(590)]):
            with self.subTest(all_null=features[0] is None):
                raw = np.asarray(features, dtype=float).reshape(1, -1)
                transformed = preprocessing.transform(raw)
                self.assertTrue(np.isfinite(transformed).all())
                if features[0] is None:
                    medians = preprocessing.named_steps["imputer"].statistics_.reshape(1, -1)
                    np.testing.assert_equal(transformed, preprocessing.named_steps["constants"].transform(medians))
                expected = float(self.bundle["model"].predict_proba(transformed)[0, 1])
                response = self.request(features=features)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["failure_score"], expected)

    def test_frozen_threshold_includes_equality(self):
        with patch.object(FrozenPredictor, "score", return_value=self.threshold):
            result = self.request().json()
        self.assertTrue(result["predicted_failure"])
        self.assertEqual(result["threshold"], self.threshold)
        self.assertEqual(self.request(threshold=0.5).status_code, 422)

    def test_no_retraining_refitting_or_artifact_modification(self):
        predictor = self.app.state.predictor
        booster_before = bytes(predictor.model.get_booster().save_raw())
        preprocessing_before = pickle.dumps(predictor.preprocessing)
        with patch.object(XGBClassifier, "fit", side_effect=AssertionError("retrain")), \
                patch.object(Pipeline, "fit", side_effect=AssertionError("refit")), \
                patch.object(Pipeline, "fit_transform", side_effect=AssertionError("refit")), \
                patch("quality_analytics.dataset.read_local_dataset",
                      side_effect=AssertionError("dataset read")), \
                patch("quality_analytics.warehouse.read_raw_table",
                      side_effect=AssertionError("dataset query")):
            self.assertEqual(self.request(features=[None] * 590).status_code, 200)
        self.assertEqual(booster_before, bytes(predictor.model.get_booster().save_raw()))
        self.assertEqual(preprocessing_before, pickle.dumps(predictor.preprocessing))
        self.assertEqual(BUNDLE_PATH.read_bytes(), self.original_bundle_bytes)

    def test_feature_order_is_preserved_and_mismatch_fails_startup(self):
        features = list(range(590))
        original_transform = Pipeline.transform
        seen = []

        def transform(pipeline, raw):
            seen.append(raw.copy())
            return original_transform(pipeline, raw)

        with patch.object(Pipeline, "transform", new=transform):
            self.assertEqual(self.request(features=features).status_code, 200)
        np.testing.assert_equal(seen[0], np.asarray(features).reshape(1, -1))
        for key in ("raw_feature_names", "retained_feature_names"):
            broken = pickle.loads(self.original_bundle_bytes)
            broken["metadata"][key].reverse()
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "selected.pkl"
                path.write_bytes(pickle.dumps(broken))
                with self.assertRaisesRegex(ValueError, "schema|order"):
                    with TestClient(create_app(InferenceConfig(path))):
                        pass

    def test_local_defaults_require_no_credentials_and_load_once(self):
        unused_writer = Mock(side_effect=AssertionError("local persistence"))
        with patch.dict(os.environ, {}, clear=True), \
                patch("quality_analytics.warehouse.bigquery.Client", side_effect=AssertionError("credentials")), \
                patch.object(FrozenPredictor, "load", wraps=FrozenPredictor.load) as load:
            with TestClient(create_app(prediction_writer=unused_writer)) as client:
                client.get("/health")
                client.get("/model-info")
                response = client.post("/predict", json={"sample_id": "local", "features": [None] * 590})
                self.assertEqual(response.status_code, 200)
                self.assertFalse(response.json()["stored"])
            load.assert_called_once()
        unused_writer.assert_not_called()

    def test_persistence_success_preserves_api_schema_and_labels_are_null(self):
        client = Mock(spec=bigquery.Client)
        client.insert_rows_json.return_value = []
        with patch.dict(os.environ, CLOUD_ENV, clear=True), \
                patch("quality_analytics.warehouse.bigquery.Client", return_value=client) as construct:
            with TestClient(create_app(InferenceConfig(BUNDLE_PATH, True))) as api:
                api.get("/health")
                api.get("/model-info")
                construct.assert_not_called()
                response = api.post("/predict", json={"sample_id": "online-001", "features": [None] * 590})
                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.json()["stored"])
                self.assertEqual(api.post("/predict", json={"sample_id": "bad", "features": []}).status_code, 422)
            construct.assert_called_once_with(project="example-project")
        client.insert_rows_json.assert_called_once()
        call = client.insert_rows_json.call_args
        self.assertEqual(call.args[0], "example-project.secom_quality.quality_predictions")
        row = call.args[1][0]
        self.assertEqual(set(row), {field.name for field in prediction_schema()})
        self.assertEqual(row["source"], "api")
        self.assertIsNone(row["split"])
        self.assertIsNone(row["actual_failure"])
        self.assertTrue(row["run_id"].startswith("api-"))
        self.assertEqual(row["dataset_version"], self.bundle["metadata"]["dataset_version"])
        for key in ("prediction_id", "sample_id", "model_version", "failure_score", "predicted_failure", "threshold"):
            self.assertEqual(row[key], response.json()[key])
        self.assertEqual(datetime.fromisoformat(row["scored_at"]).utcoffset().total_seconds(), 0)
        self.assertEqual(call.kwargs["row_ids"], [row["prediction_id"]])
        client.query.assert_not_called()
        client.create_table.assert_not_called()
        client.close.assert_called_once()

    def test_persistence_exceptions_and_rejected_rows_return_503(self):
        for rejected_rows in (False, True):
            with self.subTest(rejected_rows=rejected_rows):
                client = Mock(spec=bigquery.Client)
                if rejected_rows:
                    client.insert_rows_json.return_value = [{"index": 0, "errors": [{"message": "private detail"}]}]
                else:
                    client.insert_rows_json.side_effect = RuntimeError("private detail")
                with patch.dict(os.environ, CLOUD_ENV, clear=True), \
                        patch("quality_analytics.warehouse.bigquery.Client", return_value=client), \
                        self.assertLogs("quality_analytics.api", level="ERROR") as logs:
                    with TestClient(create_app(InferenceConfig(BUNDLE_PATH, True))) as api:
                        response = api.post("/predict", json={"sample_id": "demo", "features": [0] * 590})
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.json(), {"detail": "Prediction persistence unavailable"})
                self.assertNotIn("private detail", " ".join(logs.output))

    def test_credential_discovery_failure_returns_503_but_health_succeeds(self):
        with patch.dict(os.environ, CLOUD_ENV, clear=True), \
                patch("quality_analytics.warehouse.bigquery.Client", side_effect=RuntimeError("no ADC")), \
                self.assertLogs("quality_analytics.api", level="ERROR"):
            with TestClient(create_app(InferenceConfig(BUNDLE_PATH, True))) as api:
                self.assertEqual(api.get("/health").status_code, 200)
                response = api.post("/predict", json={"sample_id": "demo", "features": [0] * 590})
                self.assertEqual(response.status_code, 503)

    def test_config_rejects_ambiguous_flag_and_store_rejects_raw_destination(self):
        with patch.dict(os.environ, {"PERSIST_PREDICTIONS": "maybe"}, clear=True):
            with self.assertRaisesRegex(ValueError, "true or false"):
                InferenceConfig.from_env()
        with self.assertRaisesRegex(ValueError, "raw table"):
            OnlinePredictionStore(BigQueryConfig("example-project", "secom_quality", "US", "secom_raw", "secom_raw"))


if __name__ == "__main__":
    unittest.main()
