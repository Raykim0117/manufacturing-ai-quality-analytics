"""Verify configuration and BigQuery load boundaries without contacting GCP."""

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from google.cloud import bigquery
from quality_analytics.config import BigQueryConfig
from quality_analytics.warehouse import load_raw_table, raw_schema, read_raw_table


class ConfigurationTests(unittest.TestCase):
    def test_configuration_comes_from_environment(self):
        env = {"GCP_PROJECT_ID": "example-project", "BQ_DATASET": "example_data",
               "BQ_LOCATION": "US", "BQ_RAW_TABLE": "example_raw"}
        with patch.dict(os.environ, env, clear=True):
            config = BigQueryConfig.from_env()
        self.assertEqual(config.table_id, "example-project.example_data.example_raw")
        self.assertEqual(config.location, "US")

    def test_missing_configuration_fails(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(
            ValueError, "GCP_PROJECT_ID, BQ_DATASET, BQ_LOCATION, BQ_RAW_TABLE"
        ):
            BigQueryConfig.from_env()

    def test_destination_identifiers_cannot_include_sql_or_extra_qualification(self):
        env = {"GCP_PROJECT_ID": "example-project", "BQ_DATASET": "other.dataset",
               "BQ_LOCATION": "US", "BQ_RAW_TABLE": "example_raw"}
        with patch.dict(os.environ, env, clear=True), self.assertRaisesRegex(
            ValueError, "BQ_DATASET"
        ):
            BigQueryConfig.from_env()


class WarehouseTests(unittest.TestCase):
    def setUp(self):
        self.config = BigQueryConfig("example-project", "example_data", "US", "example_raw")
        self.client = Mock(spec=bigquery.Client)
        dataset = bigquery.Dataset(self.config.dataset_id)
        dataset.location = "US"
        self.client.create_dataset.return_value = dataset
        self.rows = [{
            "dataset_version": "test-version", "sample_id": "test-version:0001",
            "test_time": "2008-07-19 11:55:00", "failure_label": 0,
            **{f"feature_{index:03d}": None for index in range(590)},
        }]
        self.job = self.client.load_table_from_json.return_value
        self.job.output_rows = len(self.rows)

    def test_explicit_schema_has_590_nullable_float_features(self):
        schema = raw_schema(590)
        self.assertEqual(len(schema), 594)
        self.assertEqual(schema[2].field_type, "DATETIME")
        self.assertEqual(schema[3].field_type, "INT64")
        self.assertTrue(all(field.mode == "REQUIRED" for field in schema[:4]))
        self.assertTrue(all(field.field_type == "FLOAT64" and field.mode == "NULLABLE"
                            for field in schema[4:]))
        self.assertEqual(schema[-1].name, "feature_589")

    def test_dataset_is_created_if_needed_and_raw_table_is_explicitly_replaced(self):
        self.assertIs(load_raw_table(self.client, self.config, self.rows, 590), self.job)
        dataset_args = self.client.create_dataset.call_args
        self.assertEqual(dataset_args.args[0].project, self.config.project_id)
        self.assertEqual(dataset_args.args[0].dataset_id, self.config.dataset)
        self.assertEqual(dataset_args.args[0].location, "US")
        self.assertTrue(dataset_args.kwargs["exists_ok"])
        load_args = self.client.load_table_from_json.call_args
        self.assertEqual(load_args.args, (self.rows, self.config.table_id))
        self.assertEqual(load_args.kwargs["location"], "US")
        job_config = load_args.kwargs["job_config"]
        self.assertEqual(job_config.write_disposition, "WRITE_TRUNCATE")
        self.assertEqual(job_config.create_disposition, "CREATE_IF_NEEDED")
        self.assertFalse(job_config.autodetect)
        self.assertFalse(job_config.ignore_unknown_values)
        self.assertEqual(job_config.max_bad_records, 0)
        self.assertEqual(len(job_config.schema), 594)
        self.job.result.assert_called_once_with()

    def test_empty_or_malformed_rows_cannot_replace_a_table(self):
        for rows in ([], [{"sample_id": "incomplete"}]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                load_raw_table(self.client, self.config, rows, 590)
        self.client.create_dataset.assert_not_called()
        self.client.load_table_from_json.assert_not_called()

    def test_location_mismatch_prevents_table_upload(self):
        self.client.create_dataset.return_value.location = "EU"
        with self.assertRaisesRegex(ValueError, "Dataset location"):
            load_raw_table(self.client, self.config, self.rows, 590)
        self.client.load_table_from_json.assert_not_called()

    def test_load_job_failures_are_propagated(self):
        self.job.result.side_effect = RuntimeError("Load job failed")
        with self.assertRaisesRegex(RuntimeError, "Load job failed"):
            load_raw_table(self.client, self.config, self.rows, 590)

    def test_loaded_row_count_must_match_prepared_rows(self):
        self.job.output_rows = 0
        with self.assertRaisesRegex(RuntimeError, "expected 1"):
            load_raw_table(self.client, self.config, self.rows, 590)

    def test_training_read_is_ordered_bounded_and_does_not_write(self):
        self.client.query.return_value.result.return_value = self.rows
        self.assertEqual(read_raw_table(self.client, self.config), self.rows)
        call = self.client.query.call_args
        self.assertTrue(call.args[0].startswith("SELECT dataset_version, sample_id, test_time, failure_label"))
        self.assertIn("feature_589 FROM `example-project.example_data.example_raw` ORDER BY sample_id",
                      call.args[0])
        self.assertEqual(call.kwargs["job_config"].maximum_bytes_billed, 100_000_000)
        self.assertEqual(call.kwargs["location"], "US")
        self.client.create_dataset.assert_not_called()
        self.client.load_table_from_json.assert_not_called()

    def test_training_read_rejects_unsafe_identifiers(self):
        config = BigQueryConfig("example-project", "dataset`; DROP TABLE x", "US", "raw")
        with self.assertRaises(ValueError):
            read_raw_table(self.client, config)
        self.client.query.assert_not_called()


if __name__ == "__main__":
    unittest.main()
