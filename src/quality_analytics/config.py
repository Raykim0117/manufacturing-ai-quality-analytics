"""Environment-only configuration for raw and prediction BigQuery resources."""

from dataclasses import dataclass
import os
from pathlib import Path
import re


@dataclass(frozen=True)
class BigQueryConfig:
    project_id: str
    dataset: str
    location: str
    raw_table: str
    predictions_table: str = "quality_predictions"

    @classmethod
    def from_env(cls) -> "BigQueryConfig":
        names = ("GCP_PROJECT_ID", "BQ_DATASET", "BQ_LOCATION", "BQ_RAW_TABLE")
        values = {name: os.environ.get(name, "").strip() for name in names}
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise ValueError(f"Missing environment variables: {', '.join(missing)}")
        if not re.fullmatch(r"[a-z][a-z0-9-]{4,28}[a-z0-9]", values["GCP_PROJECT_ID"]):
            raise ValueError("GCP_PROJECT_ID must be a valid Google Cloud project ID")
        values["BQ_PREDICTIONS_TABLE"] = os.environ.get(
            "BQ_PREDICTIONS_TABLE", "quality_predictions"
        ).strip()
        for name in ("BQ_DATASET", "BQ_RAW_TABLE", "BQ_PREDICTIONS_TABLE"):
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,1023}", values[name]):
                raise ValueError(f"{name} must be a simple BigQuery identifier")
        if not re.fullmatch(r"[A-Za-z0-9-]+", values["BQ_LOCATION"]):
            raise ValueError("BQ_LOCATION must be a BigQuery region or multi-region")
        return cls(*(values[name] for name in names), predictions_table=values["BQ_PREDICTIONS_TABLE"])

    @property
    def dataset_id(self) -> str:
        return f"{self.project_id}.{self.dataset}"

    @property
    def table_id(self) -> str:
        return f"{self.dataset_id}.{self.raw_table}"

    @property
    def predictions_table_id(self) -> str:
        return f"{self.dataset_id}.{self.predictions_table}"

    @property
    def summary_view_id(self) -> str:
        return f"{self.dataset_id}.quality_summary"


@dataclass(frozen=True)
class InferenceConfig:
    model_path: Path
    persist_predictions: bool = False

    @classmethod
    def from_env(cls) -> "InferenceConfig":
        persistence = os.environ.get("PERSIST_PREDICTIONS", "false").strip().lower()
        if persistence not in ("true", "false"):
            raise ValueError("PERSIST_PREDICTIONS must be true or false")
        configured_path = os.environ.get("MODEL_PATH", "").strip()
        if configured_path:
            model_path = Path(configured_path)
        else:
            root = Path(__file__).resolve().parents[2]
            candidates = list((root / "artifacts").glob("*/selected-model-v1/selected.pkl"))
            if len(candidates) != 1:
                raise ValueError("Set MODEL_PATH to the trusted frozen selected-model bundle")
            model_path = candidates[0]
        return cls(model_path, persistence == "true")
