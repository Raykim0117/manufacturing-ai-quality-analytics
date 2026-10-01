"""Train bounded CPU challengers and freeze validation-only model selection."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import pickle
import sys

import numpy as np
from sklearn.metrics import precision_recall_curve
from xgboost import XGBClassifier

from .config import BigQueryConfig
from .dataset import TrainingDataset, dataset_from_rows, read_local_dataset
from .split import RANDOM_STATE, load_persisted_split, split_summary
from .train import MODEL_VERSION, evaluate_validation, write_json


SELECTED_VERSION = "selected-model-v1"
CANDIDATE_CONFIGS = (
    {"max_depth": 2, "n_estimators": 120},
    {"max_depth": 3, "n_estimators": 180},
)
RECALL_TARGET = 0.80


def training_class_weight(train_labels: np.ndarray) -> float:
    """Use only the labels supplied to training: PASS count divided by FAIL."""
    labels = np.asarray(train_labels)
    if labels.ndim != 1 or set(labels) != {0, 1}:
        raise ValueError("Training labels must contain both PASS=0 and FAIL=1")
    return float(np.sum(labels == 0) / np.sum(labels == 1))


def fit_candidate(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    config: dict,
) -> tuple[XGBClassifier, dict]:
    parameters = {
        **config, "objective": "binary:logistic", "eval_metric": "logloss",
        "device": "cpu", "tree_method": "hist", "n_jobs": 1,
        "random_state": RANDOM_STATE, "learning_rate": 0.05,
        "subsample": 0.8, "colsample_bytree": 0.8,
        "min_child_weight": 3, "reg_lambda": 2.0,
        "scale_pos_weight": training_class_weight(train_labels),
    }
    if config not in CANDIDATE_CONFIGS:
        raise ValueError("Only the two predefined bounded configurations are allowed")
    model = XGBClassifier(**parameters)
    model.fit(train_features, train_labels)
    return model, parameters


def select_model(validation_metrics: dict[str, dict]) -> str:
    """Compare validation PR-AUC only; exact ties keep Logistic Regression."""
    if "logistic_regression" not in validation_metrics or len(validation_metrics) > 3:
        raise ValueError("Comparison requires Logistic Regression and at most two challengers")
    for metrics in validation_metrics.values():
        if metrics.get("evaluation_split") != "validation" or not np.isfinite(metrics["pr_auc"]):
            raise ValueError("Model selection requires finite validation-only PR-AUC")
    selected = "logistic_regression"
    for name, metrics in validation_metrics.items():
        if metrics["pr_auc"] > validation_metrics[selected]["pr_auc"]:
            selected = name
    return selected


def choose_threshold(validation_labels: np.ndarray, validation_scores: np.ndarray) -> float:
    """Maximize precision at recall >=0.80, then recall, then threshold."""
    labels, scores = np.asarray(validation_labels), np.asarray(validation_scores)
    if labels.ndim != 1 or scores.shape != labels.shape or set(labels) != {0, 1}:
        raise ValueError("Validation labels/scores must align and contain both classes")
    if not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Validation failure scores must be finite and in [0, 1]")
    precision, recall, thresholds = precision_recall_curve(labels, scores, pos_label=1)
    # The last PR point has no threshold and predicts no failures; exclude it.
    eligible = np.flatnonzero(recall[:-1] >= RECALL_TARGET)
    if not len(eligible):
        raise ValueError("No validation threshold satisfies the recall constraint")
    best = max(eligible, key=lambda i: (precision[i], recall[i], thresholds[i]))
    return float(thresholds[best])


def failure_scores(model, features: np.ndarray) -> np.ndarray:
    index = list(model.classes_).index(1)
    return model.predict_proba(features)[:, index]


def compare_validation(
    dataset: TrainingDataset,
    baseline_dir: Path,
    output_dir: Path,
    provenance: dict,
) -> dict:
    """Reuse the saved baseline and memberships; never transform test rows."""
    if output_dir.exists():
        raise ValueError("Selected artifact directory already exists; frozen bundles cannot be replaced")
    split_path = baseline_dir / "split_membership.json"
    membership = load_persisted_split(dataset, split_path)
    baseline_path = baseline_dir / "baseline.pkl"
    baseline_bytes = baseline_path.read_bytes()
    baseline = pickle.loads(baseline_bytes)
    metadata = baseline["metadata"]
    if (metadata["dataset_fingerprint"] != dataset.fingerprint
            or metadata["dataset_version"] != dataset.dataset_version
            or metadata["raw_feature_names"] != list(dataset.feature_names)
            or metadata["split_counts"] != split_summary(dataset, membership)
            or metadata["preprocessing_fit_split"] != "train"
            or metadata["test_evaluated"] is not False):
        raise ValueError("Baseline metadata does not match the persisted dataset and split")
    preprocessing = baseline["preprocessing"]
    if list(preprocessing.named_steps) != ["missingness", "imputer", "constants", "scaler"]:
        raise ValueError("Baseline preprocessing steps do not match the expected pipeline")
    # Slicing and copying fitted steps preserves their training-only state.
    unscaled = deepcopy(preprocessing[:-1])
    retained = unscaled.get_feature_names_out().tolist()
    if retained != metadata["retained_feature_names"]:
        raise ValueError("Retained feature ordering does not match baseline metadata")
    train = np.array([membership[sid] == "train" for sid in dataset.sample_ids])
    validation = np.array([membership[sid] == "validation" for sid in dataset.sample_ids])
    train_labels = dataset.labels[train]
    validation_labels = dataset.labels[validation]
    train_features = unscaled.transform(dataset.features[train])
    validation_features = unscaled.transform(dataset.features[validation])
    models = {"logistic_regression": baseline["model"]}
    scores = {"logistic_regression": failure_scores(
        baseline["model"], preprocessing.transform(dataset.features[validation])
    )}
    metrics = {"logistic_regression": evaluate_validation(validation_labels, scores["logistic_regression"])}
    if not np.isclose(metrics["logistic_regression"]["pr_auc"],
                      baseline["validation_metrics"]["pr_auc"], rtol=0, atol=1e-12):
        raise ValueError("Recomputed Logistic Regression validation PR-AUC differs from baseline")
    parameters = {"logistic_regression": metadata["model_parameters"]}
    for index, config in enumerate(CANDIDATE_CONFIGS, start=1):
        name = f"xgboost_candidate_{index}"
        models[name], parameters[name] = fit_candidate(train_features, train_labels, config)
        scores[name] = failure_scores(models[name], validation_features)
        metrics[name] = evaluate_validation(validation_labels, scores[name])
    selected = select_model(metrics)
    threshold = choose_threshold(validation_labels, scores[selected])
    selected_metrics = evaluate_validation(
        validation_labels, scores[selected], threshold=threshold, threshold_optimized=True
    )
    selected_preprocessing = preprocessing if selected == "logistic_regression" else unscaled
    frozen_metadata = {
        **deepcopy(metadata), "model_version": SELECTED_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": provenance, "selected_model": selected,
        "model_parameters": parameters[selected], "threshold": threshold,
        "threshold_selection_split": "validation", "model_selection_split": "validation",
        "threshold_rule": "max precision with recall >=0.80; ties: higher recall, higher threshold",
        "recall_target": RECALL_TARGET,
        "scaling": selected == "logistic_regression",
        "dependencies": {**metadata["dependencies"], "xgboost": version("xgboost")},
        "baseline_sha256": hashlib.sha256(baseline_bytes).hexdigest(),
        "split_membership_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "train_class_counts": {"pass": int(np.sum(train_labels == 0)),
                               "fail": int(np.sum(train_labels == 1))},
        "preprocessing_refitted": False, "test_evaluated": False,
    }
    # Baseline convergence fields describe LR only, so name their provenance.
    frozen_metadata["baseline_converged"] = frozen_metadata.pop("converged")
    frozen_metadata["baseline_iterations"] = frozen_metadata.pop("iterations")
    report = {
        "dataset_version": dataset.dataset_version, "source": provenance,
        "split_counts": metadata["split_counts"],
        "validation_class_prevalence": float(validation_labels.mean()),
        "raw_feature_count": len(dataset.feature_names), "retained_feature_count": len(retained),
        "scale_pos_weight": training_class_weight(train_labels),
        "candidate_parameters": parameters, "validation_metrics_at_0_5": metrics,
        "selection_metric": "validation precision_recall_curve + trapezoidal auc(recall, precision)",
        "model_tie_break": "Exact PR-AUC ties favor Logistic Regression",
        "selected_model": selected, "selected_threshold": threshold,
        "threshold_rule": frozen_metadata["threshold_rule"],
        "selected_validation_metrics": selected_metrics,
        "selected_validation_metrics_at_0_5": metrics[selected],
        "test_evaluated": False,
        "test_statement": "TEST remains untouched by preprocessing fitting, model fitting, "
                          "prediction, evaluation, model selection, and threshold selection. "
                          "Snapshot identity and persisted membership were validated only.",
        "refit_after_selection": False,
    }
    bundle = {"model": models[selected], "preprocessing": selected_preprocessing,
              "threshold": threshold, "metadata": frozen_metadata,
              "validation_metrics": selected_metrics, "comparison_report": report}
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "selected.pkl").open("wb") as target:
        pickle.dump(bundle, target, protocol=pickle.HIGHEST_PROTOCOL)
    with (output_dir / "selected.pkl").open("rb") as source:
        restored = pickle.load(source)
    restored_scores = failure_scores(restored["model"], restored["preprocessing"].transform(
        dataset.features[validation]
    ))
    np.testing.assert_allclose(scores[selected], restored_scores, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(scores[selected] >= threshold, restored_scores >= restored["threshold"])
    for name in models:
        if name != "logistic_regression":
            with (output_dir / f"{name}.pkl").open("wb") as target:
                pickle.dump({"model": models[name], "preprocessing": unscaled,
                             "parameters": parameters[name], "validation_metrics": metrics[name]},
                            target, protocol=pickle.HIGHEST_PROTOCOL)
    write_json(output_dir / "metadata.json", frozen_metadata)
    write_json(output_dir / "retained_features.json", retained)
    write_json(output_dir / "comparison_validation.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("local", "bigquery"),
                        default=os.environ.get("DATA_BACKEND", "bigquery"))
    parser.add_argument("--local-data", type=Path, default=root / "data/normalized/secom.jsonl")
    parser.add_argument("--baseline-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report", type=Path, default=root / "reports/comparison_validation.json")
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
            provenance = {"backend": "bigquery", "table_id": config.table_id, "read_only": True}
        if len(dataset.sample_ids) != 1567 or len(dataset.feature_names) != 590:
            raise ValueError("SECOM comparison requires 1567 rows and 590 measured features")
        artifact_root = root / "artifacts" / dataset.dataset_version
        baseline_dir = args.baseline_dir or artifact_root / MODEL_VERSION
        output_dir = args.output_dir or artifact_root / SELECTED_VERSION
        report = compare_validation(dataset, baseline_dir, output_dir, provenance)
        write_json(args.report, report)
        print(json.dumps({
            "validation_pr_auc": {name: metrics["pr_auc"] for name, metrics in
                                  report["validation_metrics_at_0_5"].items()},
            "selected_model": report["selected_model"],
            "selected_threshold": report["selected_threshold"],
            "selected_metrics": {key: value for key, value in
                                 report["selected_validation_metrics"].items() if key != "pr_curve"},
            "test_evaluated": False, "artifact_directory": str(output_dir),
        }, indent=2))
    except (OSError, ValueError, ImportError) as error:
        print(f"Validation comparison failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
