"""Validate SECOM against its report and explicitly replace a BigQuery raw table."""

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

from .config import BigQueryConfig
from .validate_raw import TIMESTAMP_FORMAT, file_hash, read_labels, read_measurements


RAW_FILES = ("secom.data", "secom_labels.data", "secom.names")


def load_validation_report(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("validation_passed") is not True:
        raise ValueError("Source validation report must have validation_passed=true")
    for name in ("sample_count", "label_row_count", "feature_count"):
        if type(report.get(name)) is not int or report[name] <= 0:
            raise ValueError(f"Validation report needs a positive integer {name}")
    if report["sample_count"] != report["label_row_count"]:
        raise ValueError("Validation report measurement and label row counts disagree")
    if report.get("label_mapping") != {"-1": "PASS", "1": "FAIL"}:
        raise ValueError("Validation report must confirm -1=PASS and 1=FAIL")
    for name in RAW_FILES:
        expected = report.get("source_files", {}).get(name, {}).get("sha256_after", "")
        if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
            raise ValueError(f"Validation report needs a SHA-256 checksum for {name}")
    return report


def make_sample_id(dataset_version: str, source_row: int) -> str:
    """Identify a one-based source row; never depend on upload time or sorting."""
    if source_row < 1:
        raise ValueError("Source row number must be positive")
    return f"{dataset_version}:{source_row:04d}"


def prepare_raw_rows(raw_dir: Path, report: dict) -> list[dict]:
    """Read only; convert NaN to JSON null without changing measurements."""
    hashes_before = {name: file_hash(raw_dir / name) for name in RAW_FILES}
    measurements = read_measurements(raw_dir / "secom.data")
    labels = read_labels(raw_dir / "secom_labels.data")
    if len(measurements) != report["sample_count"]:
        raise ValueError(
            f"Sample count differs from validation report: "
            f"expected {report['sample_count']}, got {len(measurements)}"
        )
    if len(labels) != report["label_row_count"] or len(labels) != len(measurements):
        raise ValueError("Label row count differs from validation report or measurements")
    if any(len(row) != report["feature_count"] for row in measurements):
        raise ValueError(
            f"Feature count differs from validation report: expected {report['feature_count']}"
        )
    for name in RAW_FILES:
        if hashes_before[name] != report["source_files"][name]["sha256_after"]:
            raise ValueError(f"{name} checksum differs from validated source")

    content_id = hashlib.sha256(
        f"{hashes_before['secom.data']}:{hashes_before['secom_labels.data']}".encode("ascii")
    ).hexdigest()
    dataset_version = f"secom-{content_id[:16]}"
    feature_names = [f"feature_{index:03d}" for index in range(report["feature_count"])]
    result = []
    for source_row, (values, (label, timestamp)) in enumerate(
        zip(measurements, labels), start=1
    ):
        if label not in (-1, 1):
            raise ValueError(f"Source row {source_row}: unknown label {label}")
        if any(math.isinf(value) for value in values):
            raise ValueError(f"Source row {source_row}: infinite measurement")
        # strptime returns a naive datetime. DATETIME serialization assigns no timezone.
        test_time = datetime.strptime(timestamp, TIMESTAMP_FORMAT)
        record = {
            "dataset_version": dataset_version,
            "sample_id": make_sample_id(dataset_version, source_row),
            "test_time": test_time.isoformat(sep=" "),
            "failure_label": {-1: 0, 1: 1}[label],
        }
        record.update(
            (name, None if math.isnan(value) else value)
            for name, value in zip(feature_names, values)
        )
        result.append(record)
    if hashes_before != {name: file_hash(raw_dir / name) for name in RAW_FILES}:
        raise ValueError("Raw files changed while preparing ingestion")
    return result


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=root / "data/raw")
    parser.add_argument(
        "--validation-report", type=Path, default=root / "reports/raw_data_validation.json"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Validate locally; no cloud access")
    mode.add_argument(
        "--replace", action="store_true", help="Upload and replace the configured raw table"
    )
    args = parser.parse_args(argv)
    try:
        report = load_validation_report(args.validation_report)
        for warning in report.get("warnings", []):
            print(f"Source warning: {warning}", file=sys.stderr)
        rows = prepare_raw_rows(args.raw_dir, report)
        summary = {
            "sample_count": len(rows),
            "feature_count": report["feature_count"],
            "dataset_version": rows[0]["dataset_version"],
            "mode": "dry-run" if args.dry_run else "replace",
        }
        if args.replace:
            config = BigQueryConfig.from_env()
            # Import lazily: local validation requires neither the SDK nor credentials.
            from google.cloud import bigquery
            from .warehouse import load_raw_table

            with bigquery.Client(project=config.project_id, location=config.location) as client:
                job = load_raw_table(client, config, rows, report["feature_count"])
            summary.update({"table_id": config.table_id, "job_id": job.job_id})
        print(json.dumps(summary, indent=2))
    except (OSError, ValueError, ImportError) as error:
        print(f"Ingestion failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
