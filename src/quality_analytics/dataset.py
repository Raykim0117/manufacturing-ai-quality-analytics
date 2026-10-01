"""One normalized raw schema for local files and BigQuery training reads."""

from dataclasses import dataclass
from datetime import datetime
import argparse
import hashlib
import json
from numbers import Real
from pathlib import Path
import re

import numpy as np


@dataclass(frozen=True)
class TrainingDataset:
    sample_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    features: np.ndarray
    labels: np.ndarray
    dataset_version: str
    fingerprint: str


def dataset_from_rows(rows: list[dict]) -> TrainingDataset:
    """Validate raw records and extract only ordered measurements as predictors."""
    if not rows:
        raise ValueError("Training dataset is empty")
    names = tuple(sorted(name for name in rows[0] if re.fullmatch(r"feature_\d{3}", name)))
    if not names or names != tuple(f"feature_{i:03d}" for i in range(len(names))):
        raise ValueError("Raw features must be contiguous and ordered from feature_000")
    expected = set(names) | {"dataset_version", "sample_id", "test_time", "failure_label"}
    normalized = []
    for row in rows:
        if set(row) != expected:
            raise ValueError("Raw row fields do not match the normalized schema")
        if any(not isinstance(row[key], str) or not row[key].strip()
               for key in ("sample_id", "dataset_version")):
            raise ValueError("Sample ID and dataset version must be nonempty strings")
        label = row["failure_label"]
        if type(label) is not int or label not in (0, 1):
            raise ValueError("failure_label must be 0=PASS or 1=FAIL")
        timestamp = row["test_time"]
        if not isinstance(timestamp, datetime):
            timestamp = datetime.fromisoformat(timestamp)
        if timestamp.tzinfo is not None:
            raise ValueError("Raw timestamps must be timezone-naive")
        record = {key: row[key] for key in ("sample_id", "dataset_version", "failure_label")}
        record["test_time"] = timestamp.isoformat(sep=" ")
        for name in names:
            value = row[name]
            if value is not None and (isinstance(value, bool) or not isinstance(value, Real)
                                      or not np.isfinite(value)):
                raise ValueError(f"{name} must contain finite numbers or null")
            record[name] = None if value is None else float(value)
        normalized.append(record)
    normalized.sort(key=lambda row: row["sample_id"])
    ids = tuple(row["sample_id"] for row in normalized)
    versions = {row["dataset_version"] for row in normalized}
    if len(set(ids)) != len(ids) or len(versions) != 1:
        raise ValueError("Sample IDs must be unique within one dataset version")
    labels = np.array([row["failure_label"] for row in normalized], dtype=np.int64)
    if set(labels) != {0, 1}:
        raise ValueError("Training dataset must contain both PASS and FAIL")
    fingerprint = hashlib.sha256(json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()
    return TrainingDataset(
        ids, names,
        np.array([[np.nan if row[name] is None else row[name] for name in names]
                  for row in normalized], dtype=np.float64),
        labels, next(iter(versions)), fingerprint,
    )


def read_local_dataset(path: Path) -> TrainingDataset:
    """Read normalized JSON lines; local mode needs no cloud configuration."""
    with path.open(encoding="utf-8") as source:
        rows = [json.loads(line) for line in source if line.strip()]
    return dataset_from_rows(rows)


def main() -> int:
    """Export an ignored local cache using the existing hash-validated parser."""
    from .ingest import load_validation_report, prepare_raw_rows

    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--raw-dir", type=Path, default=root / "data/raw")
    parser.add_argument("--validation-report", type=Path,
                        default=root / "reports/raw_data_validation.json")
    parser.add_argument("--output", type=Path, default=root / "data/normalized/secom.jsonl")
    args = parser.parse_args()
    output = args.output.resolve()
    raw_dir = args.raw_dir.resolve()
    if output == args.validation_report.resolve() or output == raw_dir or raw_dir in output.parents:
        parser.error("Normalized output must be outside raw sources and the validation report")
    rows = prepare_raw_rows(raw_dir, load_validation_report(args.validation_report))
    dataset_from_rows(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as target:
        for row in rows:
            target.write(json.dumps(row, allow_nan=False) + "\n")
    print(f"Exported {len(rows)} validated raw rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
