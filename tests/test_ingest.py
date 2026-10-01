"""Critical ingestion checks against the repository's validated SECOM source."""

from collections import Counter
from copy import deepcopy
from datetime import datetime
import io
import json
import math
from pathlib import Path
import re
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quality_analytics import ingest
from quality_analytics.validate_raw import file_hash, read_labels, read_measurements


class IngestionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw_dir = ROOT / "data/raw"
        cls.report_path = ROOT / "reports/raw_data_validation.json"
        cls.report_hash = file_hash(cls.report_path)
        cls.report = ingest.load_validation_report(cls.report_path)
        cls.before = {name: file_hash(cls.raw_dir / name) for name in ingest.RAW_FILES}
        cls.rows = ingest.prepare_raw_rows(cls.raw_dir, cls.report)

    def test_1567_rows_in_original_order(self):
        self.assertEqual(len(self.rows), 1567)
        measurements = read_measurements(self.raw_dir / "secom.data")
        for actual, expected in zip(self.rows, measurements):
            if math.isnan(expected[0]):
                self.assertIsNone(actual["feature_000"])
            else:
                self.assertEqual(actual["feature_000"], expected[0])
        self.assertEqual(self.rows[0]["feature_000"], 3030.93)

    def test_exactly_590_features(self):
        expected = [f"feature_{index:03d}" for index in range(590)]
        self.assertEqual(self.report["feature_count"], 590)
        for row in self.rows:
            self.assertEqual([name for name in row if name.startswith("feature_")], expected)
            self.assertEqual(len(row), 594)
            self.assertNotIn("feature_590", row)

    def test_pass_fail_mapping_preserves_alignment(self):
        labels = read_labels(self.raw_dir / "secom_labels.data")
        for row, (label, _) in zip(self.rows, labels):
            self.assertEqual(row["failure_label"], 0 if label == -1 else 1)
        self.assertEqual(Counter(row["failure_label"] for row in self.rows), {0: 1463, 1: 104})
        self.assertEqual(self.rows[2]["failure_label"], 1)

    def test_sample_ids_are_stable_unique_and_follow_source_position(self):
        repeated = ingest.prepare_raw_rows(self.raw_dir, self.report)
        self.assertEqual(
            [row["sample_id"] for row in self.rows],
            [row["sample_id"] for row in repeated],
        )
        self.assertEqual(len({row["sample_id"] for row in self.rows}), 1567)
        self.assertEqual(len({row["dataset_version"] for row in self.rows}), 1)
        self.assertTrue(self.rows[0]["sample_id"].endswith(":0001"))
        self.assertTrue(self.rows[-1]["sample_id"].endswith(":1567"))

    def test_timestamps_parse_without_timezone_or_reordering(self):
        labels = read_labels(self.raw_dir / "secom_labels.data")
        for row, (_, source_timestamp) in zip(self.rows, labels):
            actual = datetime.fromisoformat(row["test_time"])
            self.assertIsNone(actual.tzinfo)
            self.assertEqual(actual.strftime("%d/%m/%Y %H:%M:%S"), source_timestamp)
        self.assertEqual(self.rows[0]["test_time"], "2008-07-19 11:55:00")
        self.assertEqual(self.rows[-1]["test_time"], "2008-10-17 06:07:00")

    def test_nan_is_null_for_strict_json_without_imputation(self):
        missing = sum(value is None for row in self.rows for value in row.values())
        self.assertEqual(missing, 41951)
        self.assertEqual(missing, self.report["missing_value_count"])
        self.assertIsNone(self.rows[0]["feature_072"])
        json.dumps(self.rows, allow_nan=False)

    def test_raw_files_and_validation_report_are_unchanged(self):
        for name in ingest.RAW_FILES:
            self.assertEqual(file_hash(self.raw_dir / name), self.before[name])
            self.assertEqual(self.before[name], self.report["source_files"][name]["sha256_after"])
        self.assertEqual(file_hash(self.report_path), self.report_hash)

    def test_metadata_warning_is_preserved(self):
        self.assertEqual(self.report["metadata_declared_attribute_count"], 591)
        self.assertTrue(any("591" in warning and "590" in warning
                            for warning in self.report["warnings"]))

    def test_dimension_mismatch_stops_before_cloud_configuration(self):
        for field, invalid in (("sample_count", 1568), ("feature_count", 591),
                               ("label_row_count", 1568)):
            report = deepcopy(self.report)
            report[field] = invalid
            with self.subTest(field=field), patch.object(
                ingest, "load_validation_report", return_value=report
            ), patch.object(ingest.BigQueryConfig, "from_env") as cloud_config, patch(
                "sys.stderr", new_callable=io.StringIO
            ):
                self.assertEqual(ingest.main(["--replace"]), 1)
                cloud_config.assert_not_called()

    def test_source_checksum_mismatch_is_rejected(self):
        report = deepcopy(self.report)
        report["source_files"]["secom.data"]["sha256_after"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "checksum differs"):
            ingest.prepare_raw_rows(self.raw_dir, report)

    def test_unknown_label_and_invalid_timestamp_are_rejected(self):
        original = read_labels(self.raw_dir / "secom_labels.data")
        for replacement in ((2, original[0][1]), (-1, "31/02/2008 11:55:00")):
            labels = [replacement, *original[1:]]
            with self.subTest(replacement=replacement), patch.object(
                ingest, "read_labels", return_value=labels
            ), self.assertRaises(ValueError):
                ingest.prepare_raw_rows(self.raw_dir, self.report)

    def test_sql_documents_exactly_the_measured_features(self):
        sql = (ROOT / "sql/create_raw_table.sql").read_text(encoding="utf-8")
        fields = re.findall(r"^\s*(feature_\d+) FLOAT64", sql, flags=re.MULTILINE)
        self.assertEqual(fields, [f"feature_{index:03d}" for index in range(590)])
        self.assertIn("test_time DATETIME NOT NULL", sql)
        self.assertIn("failure_label INT64 NOT NULL", sql)


if __name__ == "__main__":
    unittest.main()
