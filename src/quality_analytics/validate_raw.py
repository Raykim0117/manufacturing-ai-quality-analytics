"""Validate untouched SECOM source files using only the Python standard library."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys


TIMESTAMP_FORMAT = "%d/%m/%Y %H:%M:%S"
LABEL_PATTERN = re.compile(r'(-?\d+)\s+"([^"\r\n]+)"')
LABEL_MEANINGS = {-1: "PASS", 1: "FAIL"}
SOURCE_URL = "https://archive.ics.uci.edu/dataset/179/secom"


def file_hash(path: Path) -> str:
    """Hash source bytes without changing the file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_measurements(path: Path) -> list[list[float]]:
    """Preserve row/column order, accepting whitespace and numeric NaN tokens."""
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            tokens = line.split()
            if not tokens:
                raise ValueError(f"{path.name}:{line_number}: empty measurement row")
            try:
                row = [float(token) for token in tokens]
            except ValueError as error:
                raise ValueError(
                    f"{path.name}:{line_number}: nonnumeric measurement"
                ) from error
            if rows and len(row) != len(rows[0]):
                raise ValueError(f"{path.name}:{line_number}: inconsistent feature count")
            rows.append(row)
    if not rows:
        raise ValueError(f"{path.name}: no measurement rows")
    return rows


def read_labels(path: Path) -> list[tuple[int, str]]:
    """Keep original labels and timestamp text without sorting or remapping."""
    rows = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            match = LABEL_PATTERN.fullmatch(line.strip())
            if match is None:
                raise ValueError(f"{path.name}:{line_number}: invalid label/timestamp row")
            rows.append((int(match.group(1)), match.group(2)))
    if not rows:
        raise ValueError(f"{path.name}: no label rows")
    return rows


def validate_raw(raw_dir: Path) -> dict:
    paths = {name: raw_dir / name for name in (
        "secom.data", "secom_labels.data", "secom.names"
    )}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ValueError(f"Missing required source files: {', '.join(missing)}")
    before = {name: file_hash(path) for name, path in paths.items()}
    measurements = read_measurements(paths["secom.data"])
    labels = read_labels(paths["secom_labels.data"])
    names = paths["secom.names"].read_text(encoding="utf-8")

    sample_count, feature_count = len(measurements), len(measurements[0])
    missing_count = sum(math.isnan(value) for row in measurements for value in row)
    infinite_count = sum(math.isinf(value) for row in measurements for value in row)
    distribution = Counter(label for label, _ in labels)
    parsed_times, invalid_times = [], []
    for line_number, (_, timestamp) in enumerate(labels, start=1):
        try:
            parsed_times.append(datetime.strptime(timestamp, TIMESTAMP_FORMAT))
        except ValueError:
            invalid_times.append({"line": line_number, "timestamp": timestamp})

    warnings = []
    declared_width = re.search(r"Number of Attributes:\s*(\d+)", names)
    declared_width = int(declared_width.group(1)) if declared_width else None
    if declared_width is not None and declared_width != feature_count:
        warnings.append(
            f"secom.names declares {declared_width} attributes; "
            f"secom.data contains {feature_count} numeric columns. "
            "Use the measured width; do not add a feature to match the metadata."
        )
    if ".1 corresponds to a pass" in names:
        warnings.append(
            "secom.names renders the PASS label as '.1'; the official UCI page "
            "confirms -1=PASS and 1=FAIL. Raw documentation was not corrected."
        )

    after = {name: file_hash(path) for name, path in paths.items()}
    checks = {
        "required_files_present": True,
        "row_counts_match": sample_count == len(labels),
        "consistent_feature_width": True,
        "labels_valid": set(distribution).issubset(LABEL_MEANINGS),
        "all_timestamps_parseable": not invalid_times,
        "no_infinite_measurements": infinite_count == 0,
        "raw_files_unchanged": before == after,
    }
    return {
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_passed": all(checks.values()),
        "checks": checks,
        "source_files": {
            name: {"exists": True, "size_bytes": path.stat().st_size,
                   "sha256_before": before[name], "sha256_after": after[name]}
            for name, path in paths.items()
        },
        "sample_count": sample_count,
        "label_row_count": len(labels),
        "feature_count": feature_count,
        "metadata_declared_attribute_count": declared_width,
        "missing_value_count": missing_count,
        "missing_value_percentage": 100 * missing_count / (sample_count * feature_count),
        "infinite_value_count": infinite_count,
        "label_distribution": {
            str(label): {"meaning": LABEL_MEANINGS.get(label, "UNKNOWN"), "count": count}
            for label, count in sorted(distribution.items())
        },
        "label_mapping": {str(label): meaning for label, meaning in LABEL_MEANINGS.items()},
        "label_mapping_source": SOURCE_URL,
        "timestamps": {
            "format": TIMESTAMP_FORMAT,
            "parseable_count": len(parsed_times),
            "unparseable_count": len(invalid_times),
            "invalid_examples": invalid_times[:5],
            "first_original": labels[0][1],
            "last_original": labels[-1][1],
            "minimum": min(parsed_times).isoformat() if parsed_times else None,
            "maximum": max(parsed_times).isoformat() if parsed_times else None,
            "timezone": "Unspecified in source; no timezone assigned",
        },
        "warnings": warnings,
        "limitations": [
            "Rows are paired by original position; matching counts cannot prove semantic alignment.",
            "Validation only: no values imputed, labels remapped, or features transformed.",
        ],
    }


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=repo_root / "data/raw")
    parser.add_argument(
        "--output", type=Path, default=repo_root / "reports/raw_data_validation.json"
    )
    args = parser.parse_args()
    try:
        raw_dir = args.raw_dir.resolve()
        output = args.output.resolve()
        if output == raw_dir or raw_dir in output.parents:
            raise ValueError("The validation report must be written outside the raw directory")
        report = validate_raw(raw_dir)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError) as error:
        print(f"Validation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["validation_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
