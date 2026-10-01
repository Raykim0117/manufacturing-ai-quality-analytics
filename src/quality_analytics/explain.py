"""Offline validation SHAP associations for the frozen XGBoost selection."""

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import pickle
import sys

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap
from xgboost import XGBClassifier

from .dataset import TrainingDataset, read_local_dataset
from .split import RANDOM_STATE, load_persisted_split
from .train import write_json


VALIDATION_CAP = 100
BACKGROUND_CAP = 64
ASSOCIATION_STATEMENT = (
    "SHAP explains model associations, not causal effects or physical root causes. "
    "Anonymized SECOM feature IDs prevent physical process interpretation. "
    "Process-engineer review would be required before operational action."
)


def fixed_indices(
    dataset: TrainingDataset,
    membership: dict[str, str],
    split: str,
    cap: int,
) -> np.ndarray:
    """Sample stable ID positions without consulting labels or held-out scores."""
    maximum = {"train": BACKGROUND_CAP, "validation": VALIDATION_CAP}
    if split not in maximum or type(cap) is not int or not 1 <= cap <= maximum[split]:
        raise ValueError("SHAP sampling permits capped train background or validation rows only")
    positions = np.array([i for i, sid in enumerate(dataset.sample_ids)
                          if membership[sid] == split], dtype=int)
    if not len(positions):
        raise ValueError(f"No {split} samples available")
    selected = np.random.default_rng(RANDOM_STATE).choice(
        positions, size=min(cap, len(positions)), replace=False
    )
    return np.sort(selected)


def save_summary_plot(explanation: shap.Explanation, path: Path) -> None:
    """Render the top 20 associations with reproducible jitter and original IDs."""
    state = np.random.get_state()
    try:
        np.random.seed(RANDOM_STATE)
        shap.summary_plot(
            explanation.values, explanation.data, feature_names=explanation.feature_names,
            max_display=20, plot_type="dot", plot_size=(10, 9), show=False,
        )
        plt.title(f"Validation SHAP model associations ({len(explanation.values)} samples)")
        plt.xlabel("SHAP value (failure log-odds)")
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
    finally:
        plt.close("all")
        np.random.set_state(state)


def explain_validation(
    dataset: TrainingDataset,
    bundle_path: Path,
    split_path: Path,
    report_dir: Path,
    sample_size: int = VALIDATION_CAP,
    background_size: int = BACKGROUND_CAP,
) -> dict:
    """Transform existing fitted pipelines only; test rows receive no predictions."""
    bundle_bytes, split_bytes = bundle_path.read_bytes(), split_path.read_bytes()
    bundle = pickle.loads(bundle_bytes)
    metadata = bundle["metadata"]
    if (metadata["selected_model"] != "xgboost_candidate_1"
            or not isinstance(bundle["model"], XGBClassifier)
            or metadata["dataset_version"] != dataset.dataset_version
            or metadata["dataset_fingerprint"] != dataset.fingerprint
            or metadata["raw_feature_names"] != list(dataset.feature_names)
            or metadata["test_evaluated"] is not False
            or metadata["preprocessing_fit_split"] != "train"
            or metadata["scaling"] is not False
            or bundle["threshold"] != metadata["threshold"]):
        raise ValueError("Frozen selected bundle does not match the expected XGBoost dataset and schema")
    if hashlib.sha256(split_bytes).hexdigest() != metadata["split_membership_sha256"]:
        raise ValueError("Split membership checksum differs from frozen selection")
    membership = load_persisted_split(dataset, split_path)
    preprocessing, model = bundle["preprocessing"], bundle["model"]
    feature_names = preprocessing.get_feature_names_out().tolist()
    if feature_names != metadata["retained_feature_names"]:
        raise ValueError("Feature order does not match the frozen selected bundle")
    background_indices = fixed_indices(dataset, membership, "train", background_size)
    validation_indices = fixed_indices(dataset, membership, "validation", sample_size)
    preprocessing_before = pickle.dumps(preprocessing)
    model_before = bytes(model.get_booster().save_raw(raw_format="ubj"))
    background = preprocessing.transform(dataset.features[background_indices])
    features = preprocessing.transform(dataset.features[validation_indices])
    if features.shape[1] != len(feature_names) or background.shape[1] != len(feature_names):
        raise ValueError("Transformed feature dimensions do not match the frozen feature order")
    explainer = shap.TreeExplainer(
        model, data=background, model_output="raw", feature_perturbation="interventional",
        feature_names=feature_names,
    )
    explanation = explainer(features, check_additivity=True)
    values = np.asarray(explanation.values)
    if values.shape != features.shape or not np.isfinite(values).all():
        raise ValueError("SHAP dimensions or values do not match the explained validation sample")
    if list(explanation.feature_names) != feature_names:
        raise ValueError("SHAP output feature order differs from the frozen bundle")
    base_values = np.asarray(explanation.base_values)
    if base_values.shape != (len(features),) or not np.isfinite(base_values).all():
        raise ValueError("SHAP expected values must match the explained sample count")
    margins = model.predict(features, output_margin=True)
    reconstructed = base_values + values.sum(axis=1)
    np.testing.assert_allclose(reconstructed, margins, rtol=1e-5, atol=1e-5)
    scores = model.predict_proba(features)[:, list(model.classes_).index(1)]
    np.testing.assert_allclose(1 / (1 + np.exp(-reconstructed)), scores, rtol=1e-5, atol=1e-6)
    local_index = int(np.argmax(scores))
    if scores[local_index] < bundle["threshold"]:
        raise ValueError("Fixed validation sample has no prediction above the frozen risk threshold")
    mean_absolute = np.abs(values).mean(axis=0)
    order = sorted(range(len(feature_names)), key=lambda i: (-mean_absolute[i], feature_names[i]))
    importance = [{"rank": rank, "feature_id": feature_names[i],
                   "mean_abs_shap": float(mean_absolute[i])}
                  for rank, i in enumerate(order, start=1)]
    common = {
        "dataset_version": dataset.dataset_version, "model_version": metadata["model_version"],
        "selected_model": metadata["selected_model"], "threshold": bundle["threshold"],
        "selected_bundle_sha256": hashlib.sha256(bundle_bytes).hexdigest(),
        "split_membership_sha256": hashlib.sha256(split_bytes).hexdigest(),
        "explainer": "TreeExplainer", "feature_perturbation": "interventional",
        "model_output": "raw", "units": "failure log-odds (raw XGBoost margin)",
        "analysis_split": "validation", "background_split": "train", "random_state": RANDOM_STATE,
        "explained_sample_count": len(validation_indices), "background_sample_count": len(background_indices),
        "test_evaluated": False, "interpretation": ASSOCIATION_STATEMENT,
    }
    global_report = {
        **common, "feature_names": feature_names, "shap_shape": list(values.shape),
        "explained_sample_ids": [dataset.sample_ids[i] for i in validation_indices],
        "background_sample_ids": [dataset.sample_ids[i] for i in background_indices],
        "dependencies": {name: version(name) for name in (
            "shap", "matplotlib", "numpy", "xgboost", "scikit-learn", "numba", "pandas"
        )},
        "importance_definition": "mean absolute SHAP over the fixed validation sample",
        "feature_importance": importance, "top_20_features": importance[:20],
        "maximum_additivity_error": float(np.max(np.abs(reconstructed - margins))),
        "test_statement": "TEST was not transformed, predicted, explained, or evaluated. "
                          "Snapshot identity and saved membership were validated only.",
    }
    row_index = int(validation_indices[local_index])
    local_order = sorted(range(len(feature_names)), key=lambda i: (-abs(values[local_index, i]), feature_names[i]))
    raw_positions = {name: i for i, name in enumerate(dataset.feature_names)}
    contributions = []
    for i, name in enumerate(feature_names):
        raw_value = dataset.features[row_index, raw_positions[name]]
        contributions.append({
            "feature_id": name, "raw_value": None if np.isnan(raw_value) else float(raw_value),
            "transformed_value": float(features[local_index, i]),
            "training_median_imputed": bool(np.isnan(raw_value)),
            "shap_value": float(values[local_index, i]),
        })
    local_report = {
        **common, "sample_id": dataset.sample_ids[row_index], "split": "validation",
        "selection_rule": "highest failure score within the fixed explained validation sample",
        "failure_score": float(scores[local_index]), "predicted_high_risk": True,
        "score_interpretation": "uncalibrated model score; not measured manufacturing yield",
        "base_value": float(base_values[local_index]), "model_raw_margin": float(margins[local_index]),
        "shap_value_sum": float(values[local_index].sum()),
        "reconstructed_raw_margin": float(reconstructed[local_index]),
        "feature_order": feature_names, "feature_contributions": contributions,
        "top_20_associations": [contributions[i] for i in local_order[:20]],
    }
    if (model_before != bytes(model.get_booster().save_raw(raw_format="ubj"))
            or preprocessing_before != pickle.dumps(preprocessing)
            or bundle_bytes != bundle_path.read_bytes() or split_bytes != split_path.read_bytes()):
        raise RuntimeError("Frozen model, preprocessing, bundle, or membership changed during SHAP analysis")
    report_dir.mkdir(parents=True, exist_ok=True)
    save_summary_plot(explanation, report_dir / "shap_summary.png")
    write_json(report_dir / "shap_feature_importance.json", global_report)
    write_json(report_dir / "shap_local_example.json", local_report)
    return {"global": global_report, "local": local_report}


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-data", type=Path, default=root / "data/normalized/secom.jsonl")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--split-membership", type=Path)
    parser.add_argument("--report-dir", type=Path, default=root / "reports")
    parser.add_argument("--sample-size", type=int, default=VALIDATION_CAP)
    parser.add_argument("--background-size", type=int, default=BACKGROUND_CAP)
    args = parser.parse_args(argv)
    try:
        dataset = read_local_dataset(args.local_data)
        artifact_root = root / "artifacts" / dataset.dataset_version
        bundle_path = args.bundle or artifact_root / "selected-model-v1/selected.pkl"
        split_path = args.split_membership or artifact_root / "logistic-regression-v1/split_membership.json"
        result = explain_validation(dataset, bundle_path, split_path, args.report_dir,
                                    args.sample_size, args.background_size)
        print(json.dumps({
            "explained_validation_samples": result["global"]["explained_sample_count"],
            "background_training_samples": result["global"]["background_sample_count"],
            "top_10_features": result["global"]["feature_importance"][:10],
            "local_sample_id": result["local"]["sample_id"],
            "local_failure_score": result["local"]["failure_score"],
            "test_evaluated": False,
        }, indent=2))
    except (OSError, ValueError, ImportError) as error:
        print(f"SHAP analysis failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
