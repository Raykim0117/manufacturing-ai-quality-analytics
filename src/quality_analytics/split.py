"""Persist deterministic stratified memberships keyed by stable sample ID."""

import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from .dataset import TrainingDataset


RANDOM_STATE = 42
SPLIT_NAMES = ("train", "validation", "test")


def create_membership(dataset: TrainingDataset) -> dict[str, str]:
    """Canonical ID order makes assignment independent of source read order."""
    indices = np.arange(len(dataset.sample_ids))
    train, held_out = train_test_split(
        indices, test_size=0.4, stratify=dataset.labels, random_state=RANDOM_STATE
    )
    validation, test = train_test_split(
        held_out, test_size=0.5, stratify=dataset.labels[held_out], random_state=RANDOM_STATE
    )
    return {dataset.sample_ids[int(index)]: name
            for name, group in zip(SPLIT_NAMES, (train, validation, test)) for index in group}


def split_summary(dataset: TrainingDataset, membership: dict[str, str]) -> dict:
    return {
        name: {"rows": int(np.sum(mask)), "fail": int(dataset.labels[mask].sum()),
               "pass": int(np.sum(mask) - dataset.labels[mask].sum())}
        for name in SPLIT_NAMES
        for mask in [np.array([membership[sample_id] == name for sample_id in dataset.sample_ids])]
    }


def load_or_create_split(dataset: TrainingDataset, path: Path) -> dict[str, str]:
    expected = {
        "schema_version": 1,
        "dataset_version": dataset.dataset_version,
        "dataset_fingerprint": dataset.fingerprint,
        "random_state": RANDOM_STATE,
        "fractions": {"train": 0.6, "validation": 0.2, "test": 0.2},
        "method": "sorted sample_id; stratified 60/40 then stratified 50/50 held-out",
        "membership": create_membership(dataset),
    }
    expected["counts"] = split_summary(dataset, expected["membership"])
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != expected:
            raise ValueError("Persisted split differs from dataset or split configuration")
        return saved["membership"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return expected["membership"]


def load_persisted_split(dataset: TrainingDataset, path: Path) -> dict[str, str]:
    """Read existing membership without generating or writing assignments."""
    saved = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1, "dataset_version": dataset.dataset_version,
        "dataset_fingerprint": dataset.fingerprint, "random_state": RANDOM_STATE,
        "fractions": {"train": 0.6, "validation": 0.2, "test": 0.2},
    }
    if any(saved.get(key) != value for key, value in expected.items()):
        raise ValueError("Persisted split identity or configuration does not match")
    membership = saved.get("membership", {})
    if set(membership) != set(dataset.sample_ids) or set(membership.values()) != set(SPLIT_NAMES):
        raise ValueError("Persisted split must assign every sample to one valid split")
    if saved.get("counts") != split_summary(dataset, membership):
        raise ValueError("Persisted split counts do not match the dataset")
    return membership
