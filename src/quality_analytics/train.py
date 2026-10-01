"""Fit the Logistic Regression baseline on train; evaluate validation only."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import pickle
import platform
import sys
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc, confusion_matrix, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score,
)

from .config import BigQueryConfig
from .dataset import TrainingDataset, dataset_from_rows, read_local_dataset
from .preprocessing import make_preprocessing
from .split import RANDOM_STATE, load_or_create_split, split_summary


THRESHOLD = 0.5
MODEL_VERSION = "logistic-regression-v1"


def evaluate_validation(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float = THRESHOLD,
    threshold_optimized: bool = False,
) -> dict:
    """Failure is positive; PR-AUC uses trapezoids, not average precision."""
    precision, recall, _ = precision_recall_curve(labels, scores, pos_label=1)
    predicted = (scores >= threshold).astype(int)
    return {
        "evaluation_split": "validation",
        "threshold": threshold,
        "threshold_optimized": threshold_optimized,
        "positive_class": "manufacturing failure (1)",
        "pr_auc": float(auc(recall, precision)),
        "pr_auc_definition": "precision_recall_curve + trapezoidal auc(recall, precision)",
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "confusion_matrix": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
        "confusion_matrix_order": "rows=actual, columns=predicted; [PASS=0, FAIL=1]",
        "zero_division": 0,
        "rows": len(labels),
        "fail": int(labels.sum()),
        "failure_prevalence": float(labels.mean()),
        "pr_curve": {"recall": recall.tolist(), "precision": precision.tolist()},
    }


def write_json(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def train_baseline(
    dataset: TrainingDataset,
    output_dir: Path,
    provenance: dict,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    membership = load_or_create_split(dataset, output_dir / "split_membership.json")
    train = np.array([membership[sid] == "train" for sid in dataset.sample_ids])
    validation = np.array([membership[sid] == "validation" for sid in dataset.sample_ids])
    preprocessing = make_preprocessing(dataset.feature_names)
    X_train = preprocessing.fit_transform(dataset.features[train])
    model = LogisticRegression(
        class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE,
        solver="lbfgs", C=1.0, tol=1e-4,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(X_train, dataset.labels[train])
    X_validation = preprocessing.transform(dataset.features[validation])
    scores = model.predict_proba(X_validation)[:, list(model.classes_).index(1)]
    metrics = evaluate_validation(dataset.labels[validation], scores)
    retained = preprocessing.get_feature_names_out().tolist()
    metadata = {
        "model_version": MODEL_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_version": dataset.dataset_version,
        "dataset_fingerprint": dataset.fingerprint,
        "source": provenance,
        "random_state": RANDOM_STATE,
        "threshold": THRESHOLD,
        "split_counts": split_summary(dataset, membership),
        "raw_feature_names": list(dataset.feature_names),
        "raw_feature_count": len(dataset.feature_names),
        "retained_feature_names": retained,
        "retained_feature_count": len(retained),
        "preprocessing_fit_split": "train",
        "preprocessing_fit_rows": int(train.sum()),
        "missingness_threshold": 0.5,
        "dropped_for_missingness": [name for name, keep in zip(
            dataset.feature_names, preprocessing.named_steps["missingness"].support_
        ) if not keep],
        "dropped_constants": [name for name, keep in zip(
            preprocessing.named_steps["missingness"].get_feature_names_out(),
            preprocessing.named_steps["constants"].get_support(),
        ) if not keep],
        "model_parameters": model.get_params(),
        "converged": True,
        "iterations": model.n_iter_.tolist(),
        "test_evaluated": False,
        "dependencies": {name: version(name) for name in (
            "numpy", "scikit-learn", "scipy", "joblib", "threadpoolctl", "google-cloud-bigquery"
        )},
        "python_version": platform.python_version(),
        "limitations": [
            "Random stratification does not establish future production performance.",
            "Balanced class weights do not establish score calibration.",
            "Dataset fingerprint validates identity; it fits no transformations.",
        ],
    }
    bundle = {"preprocessing": preprocessing, "model": model,
              "metadata": metadata, "validation_metrics": metrics}
    bundle_path = output_dir / "baseline.pkl"
    with bundle_path.open("wb") as output:
        pickle.dump(bundle, output, protocol=pickle.HIGHEST_PROTOCOL)
    with bundle_path.open("rb") as source:
        restored = pickle.load(source)
    restored_scores = restored["model"].predict_proba(
        restored["preprocessing"].transform(dataset.features[validation])
    )[:, 1]
    np.testing.assert_allclose(scores, restored_scores, rtol=1e-12, atol=1e-12)
    write_json(output_dir / "retained_features.json", retained)
    write_json(output_dir / "validation_metrics.json", metrics)
    write_json(output_dir / "metadata.json", metadata)
    return {"metadata": metadata, "validation_metrics": metrics}


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("bigquery", "local"),
                        default=os.environ.get("DATA_BACKEND", "bigquery"))
    parser.add_argument("--local-data", type=Path, default=root / "data/normalized/secom.jsonl")
    parser.add_argument("--artifact-root", type=Path, default=root / "artifacts")
    parser.add_argument("--report", type=Path, default=root / "reports/baseline_validation.json")
    args = parser.parse_args(argv)
    try:
        if args.backend == "local":
            dataset = read_local_dataset(args.local_data)
            provenance = {"backend": "local", "normalized_file": str(args.local_data)}
        else:
            from google.cloud import bigquery
            from .warehouse import read_raw_table

            config = BigQueryConfig.from_env()
            with bigquery.Client(project=config.project_id, location=config.location) as client:
                dataset = dataset_from_rows(read_raw_table(client, config))
            provenance = {"backend": "bigquery", "table_id": config.table_id,
                          "location": config.location, "read_only": True}
        if len(dataset.sample_ids) != 1567 or len(dataset.feature_names) != 590:
            raise ValueError("SECOM baseline requires 1567 rows and 590 measured features")
        output_dir = args.artifact_root / dataset.dataset_version / MODEL_VERSION
        result = train_baseline(dataset, output_dir, provenance)
        write_json(args.report, result)
        summary = {"artifact_directory": str(output_dir), **result["metadata"]["split_counts"],
                   "raw_features": result["metadata"]["raw_feature_count"],
                   "retained_features": result["metadata"]["retained_feature_count"],
                   "validation_metrics": {key: value for key, value in
                                          result["validation_metrics"].items() if key != "pr_curve"}}
        print(json.dumps(summary, indent=2))
    except (OSError, ValueError, ImportError, ConvergenceWarning) as error:
        print(f"Baseline training failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
