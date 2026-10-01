"""One-time frozen TEST evaluation and persistence of its cached predictions."""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import pickle
import sys
from uuid import NAMESPACE_URL, uuid4, uuid5

import numpy as np
from sklearn.metrics import (
    auc, confusion_matrix, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score,
)

from .config import BigQueryConfig
from .dataset import TrainingDataset, read_local_dataset
from .split import load_persisted_split
from .train import write_json


def test_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    """Evaluate cached scores at a fixed threshold; perform no optimization."""
    precision, recall, _ = precision_recall_curve(labels, scores, pos_label=1)
    predicted = scores >= threshold
    return {
        "evaluation_split": "test", "threshold": threshold, "threshold_optimized": False,
        "positive_class": "manufacturing failure (1)",
        "pr_auc": float(auc(recall, precision)),
        "pr_auc_definition": "precision_recall_curve + trapezoidal auc(recall, precision)",
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
        "confusion_matrix_order": "rows=actual, columns=predicted; [PASS=0, FAIL=1]",
        "rows": len(labels), "fail": int(labels.sum()), "failure_prevalence": float(labels.mean()),
        "zero_division": 0, "pr_curve": {"recall": recall.tolist(), "precision": precision.tolist()},
    }


def evaluate_test_once(
    dataset: TrainingDataset,
    bundle_path: Path,
    split_path: Path,
    state_dir: Path,
) -> dict:
    """Claim a permanent local guard before scoring; future attempts fail closed."""
    guard = state_dir / "evaluation_guard.json"
    if guard.exists():
        raise ValueError("Final TEST evaluation already claimed; use --persist with cached scores")
    bundle_bytes, split_bytes = bundle_path.read_bytes(), split_path.read_bytes()
    bundle = pickle.loads(bundle_bytes)
    metadata = bundle["metadata"]
    if (metadata["dataset_fingerprint"] != dataset.fingerprint
            or metadata["dataset_version"] != dataset.dataset_version
            or metadata["raw_feature_names"] != list(dataset.feature_names)
            or metadata["selected_model"] != "xgboost_candidate_1"
            or metadata["preprocessing_fit_split"] != "train"
            or metadata["model_selection_split"] != "validation"
            or metadata["threshold_selection_split"] != "validation"
            or metadata["test_evaluated"] is not False
            or bundle["threshold"] != metadata["threshold"]):
        raise ValueError("Frozen selected bundle identity or threshold does not match")
    if hashlib.sha256(split_bytes).hexdigest() != metadata["split_membership_sha256"]:
        raise ValueError("Persisted membership checksum differs from frozen selection")
    membership = load_persisted_split(dataset, split_path)
    preprocessing, model = bundle["preprocessing"], bundle["model"]
    names = preprocessing.get_feature_names_out().tolist()
    if names != metadata["retained_feature_names"]:
        raise ValueError("Feature order differs from frozen selection")
    test = np.array([membership[sid] == "test" for sid in dataset.sample_ids])
    labels = dataset.labels[test]
    if set(labels) != {0, 1}:
        raise ValueError("Final TEST membership must contain both classes")
    model_before = bytes(model.get_booster().save_raw(raw_format="ubj"))
    preprocessing_before = pickle.dumps(preprocessing)
    scored_at = datetime.now(timezone.utc).isoformat()
    run_id = f"test-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    identity = {
        "run_id": run_id, "dataset_version": dataset.dataset_version,
        "dataset_fingerprint": dataset.fingerprint, "model_version": metadata["model_version"],
        "selected_model": metadata["selected_model"], "threshold": bundle["threshold"],
        "selected_bundle_sha256": hashlib.sha256(bundle_bytes).hexdigest(),
        "split_membership_sha256": hashlib.sha256(split_bytes).hexdigest(),
        "test_sample_ids": [sid for sid, keep in zip(dataset.sample_ids, test) if keep],
    }
    state_dir.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents two processes from claiming the same evaluation.
    with guard.open("x", encoding="utf-8") as target:
        json.dump({**identity, "state": "started"}, target, indent=2)
    features = preprocessing.transform(dataset.features[test])
    scores = model.predict_proba(features)[:, list(model.classes_).index(1)]
    if scores.shape != labels.shape or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Final TEST failure scores have invalid dimensions or values")
    rows = [{
        "prediction_id": str(uuid5(NAMESPACE_URL, f"{run_id}:{sample_id}")),
        "sample_id": sample_id, "run_id": run_id,
        "model_version": metadata["model_version"], "dataset_version": dataset.dataset_version,
        "source": "batch", "split": "test", "scored_at": scored_at,
        "failure_score": float(score), "predicted_failure": bool(score >= bundle["threshold"]),
        "threshold": bundle["threshold"], "actual_failure": int(label),
    } for sample_id, score, label in zip(identity["test_sample_ids"], scores, labels)]
    # Save scores before evaluation/reporting so a later error cannot require rescoring.
    prediction_bytes = ("\n".join(json.dumps(row, allow_nan=False) for row in rows) + "\n").encode("utf-8")
    with (state_dir / "test_predictions.jsonl").open("xb") as target:
        target.write(prediction_bytes)
    report = {
        **identity, "state": "scored", "test_scoring_calls": 1,
        "prediction_sha256": hashlib.sha256(prediction_bytes).hexdigest(),
        "rows_prepared": len(rows), "scored_at": scored_at,
        "metrics_at_frozen_threshold": test_metrics(labels, scores, bundle["threshold"]),
        "metrics_at_0_5_reference": test_metrics(labels, scores, 0.5),
        "reference_statement": "Threshold 0.5 is a reference only; selected threshold is unchanged.",
        "selection_uses_test": False, "retrained": False, "preprocessing_refitted": False,
        "dependencies": {name: version(name) for name in ("numpy", "scikit-learn", "xgboost", "google-cloud-bigquery")},
        "bigquery": {"status": "pending", "rows_written": 0},
    }
    if (bundle_bytes != bundle_path.read_bytes() or split_bytes != split_path.read_bytes()
            or model_before != bytes(model.get_booster().save_raw(raw_format="ubj"))
            or preprocessing_before != pickle.dumps(preprocessing)):
        raise RuntimeError("Frozen artifacts changed during final TEST evaluation")
    write_json(state_dir / "evaluation.json", report)
    write_json(guard, {**identity, "state": "scored", "prediction_sha256": report["prediction_sha256"]})
    return report


def read_cached_batch(state_dir: Path) -> tuple[dict, list[dict]]:
    """Persistence retries read scores only; no dataset or model is loaded."""
    report = json.loads((state_dir / "evaluation.json").read_text(encoding="utf-8"))
    guard = json.loads((state_dir / "evaluation_guard.json").read_text(encoding="utf-8"))
    payload = (state_dir / "test_predictions.jsonl").read_bytes()
    if (report["state"] != "scored" or guard["state"] != "scored"
            or report["run_id"] != guard["run_id"]
            or hashlib.sha256(payload).hexdigest() != report["prediction_sha256"]
            or report["prediction_sha256"] != guard["prediction_sha256"]):
        raise ValueError("Cached TEST batch identity or checksum is invalid")
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
    if len(rows) != report["rows_prepared"] or [row["sample_id"] for row in rows] != report["test_sample_ids"]:
        raise ValueError("Cached batch does not preserve TEST membership")
    if any(row["run_id"] != report["run_id"] or row["threshold"] != report["threshold"]
           or row["model_version"] != report["model_version"]
           or row["dataset_version"] != report["dataset_version"] for row in rows):
        raise ValueError("Cached batch differs from frozen evaluation metadata")
    return report, rows


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--evaluate-once", action="store_true")
    mode.add_argument("--persist", action="store_true", help="Upload saved scores without model evaluation")
    parser.add_argument("--local-data", type=Path, default=root / "data/normalized/secom.jsonl")
    parser.add_argument("--report", type=Path, default=root / "reports/final_test_evaluation.json")
    args = parser.parse_args(argv)
    try:
        if args.evaluate_once:
            dataset = read_local_dataset(args.local_data)
            if len(dataset.sample_ids) != 1567 or len(dataset.feature_names) != 590:
                raise ValueError("SECOM final evaluation requires 1567 rows and 590 measured features")
            artifact_root = root / "artifacts" / dataset.dataset_version
            report = evaluate_test_once(
                dataset, artifact_root / "selected-model-v1/selected.pkl",
                artifact_root / "logistic-regression-v1/split_membership.json",
                artifact_root / "final-test-v1",
            )
        else:
            # Locate the one saved evaluation without reading inputs or loading any model.
            states = list((root / "artifacts").glob("*/final-test-v1/evaluation.json"))
            if len(states) != 1:
                raise ValueError("Persistence requires exactly one completed cached TEST evaluation")
            state_dir = states[0].parent
            report, rows = read_cached_batch(state_dir)
            config = BigQueryConfig.from_env()
            from google.cloud import bigquery
            from .warehouse import persist_test_batch

            with bigquery.Client(project=config.project_id, location=config.location) as client:
                receipt = persist_test_batch(client, config, rows)
            report["bigquery"] = {"status": "verified", **receipt}
            write_json(state_dir / "persistence.json", report["bigquery"])
            write_json(state_dir / "evaluation.json", report)
        write_json(args.report, report)
        print(json.dumps({
            "run_id": report["run_id"], "rows_prepared": report["rows_prepared"],
            "metrics_at_frozen_threshold": {key: value for key, value in
                report["metrics_at_frozen_threshold"].items() if key != "pr_curve"},
            "bigquery": report["bigquery"],
        }, indent=2))
    except (OSError, ValueError, ImportError) as error:
        print(f"Final TEST batch failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
