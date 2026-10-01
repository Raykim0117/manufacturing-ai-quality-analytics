"""Environment-only configuration for the BigQuery ingestion stage."""

from dataclasses import dataclass
import os
import re


@dataclass(frozen=True)
class BigQueryConfig:
    project_id: str
    dataset: str
    location: str
    raw_table: str

    @classmethod
    def from_env(cls) -> "BigQueryConfig":
        names = ("GCP_PROJECT_ID", "BQ_DATASET", "BQ_LOCATION", "BQ_RAW_TABLE")
        values = {name: os.environ.get(name, "").strip() for name in names}
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise ValueError(f"Missing environment variables: {', '.join(missing)}")
        if not re.fullmatch(r"[a-z][a-z0-9-]{4,28}[a-z0-9]", values["GCP_PROJECT_ID"]):
            raise ValueError("GCP_PROJECT_ID must be a valid Google Cloud project ID")
        for name in ("BQ_DATASET", "BQ_RAW_TABLE"):
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,1023}", values[name]):
                raise ValueError(f"{name} must be a simple BigQuery identifier")
        if not re.fullmatch(r"[A-Za-z0-9-]+", values["BQ_LOCATION"]):
            raise ValueError("BQ_LOCATION must be a BigQuery region or multi-region")
        return cls(*(values[name] for name in names))

    @property
    def dataset_id(self) -> str:
        return f"{self.project_id}.{self.dataset}"

    @property
    def table_id(self) -> str:
        return f"{self.dataset_id}.{self.raw_table}"
