"""Small BigQuery helpers for raw ingestion and read-only training queries."""

import re
from datetime import datetime, timedelta
import hashlib
import math
from pathlib import Path
from string import Template

from google.api_core.exceptions import Conflict, NotFound
from google.cloud import bigquery

from .config import BigQueryConfig


def read_raw_table(
    client: bigquery.Client,
    config: BigQueryConfig,
    feature_count: int = 590,
) -> list[dict]:
    """Read an ordered snapshot without creating or changing any table."""
    if feature_count <= 0:
        raise ValueError("Feature count must be positive")
    # Revalidate even when the caller constructed the configuration directly.
    if not re.fullmatch(r"[a-z][a-z0-9-]{4,28}[a-z0-9]", config.project_id):
        raise ValueError("Invalid project ID")
    if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,1023}", value)
           for value in (config.dataset, config.raw_table)):
        raise ValueError("Invalid BigQuery identifier")
    fields = ", ".join(field.name for field in raw_schema(feature_count))
    query = f"SELECT {fields} FROM `{config.table_id}` ORDER BY sample_id"
    job_config = bigquery.QueryJobConfig(maximum_bytes_billed=100_000_000)
    job = client.query(query, job_config=job_config, location=config.location)
    return [dict(row) for row in job.result()]


def raw_schema(feature_count: int) -> list[bigquery.SchemaField]:
    """Use the measured width supplied by the validation report, never metadata."""
    return [
        bigquery.SchemaField("dataset_version", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("sample_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("test_time", "DATETIME", mode="REQUIRED"),
        bigquery.SchemaField("failure_label", "INT64", mode="REQUIRED"),
        *[
            bigquery.SchemaField(f"feature_{index:03d}", "FLOAT64", mode="NULLABLE")
            for index in range(feature_count)
        ],
    ]


def ensure_dataset(client: bigquery.Client, config: BigQueryConfig) -> None:
    dataset = bigquery.Dataset(config.dataset_id)
    dataset.location = config.location
    existing = client.create_dataset(dataset, exists_ok=True)
    if existing.location.casefold() != config.location.casefold():
        raise ValueError(
            f"Dataset location is {existing.location}; configured BQ_LOCATION is {config.location}"
        )


def load_raw_table(
    client: bigquery.Client,
    config: BigQueryConfig,
    rows: list[dict],
    feature_count: int,
) -> bigquery.LoadJob:
    """Explicitly replace one raw table, waiting for the load job to finish."""
    if not rows or feature_count <= 0:
        raise ValueError("Refusing to replace a raw table with an empty dataset")
    schema = raw_schema(feature_count)
    expected_fields = {field.name for field in schema}
    if any(set(row) != expected_fields for row in rows):
        raise ValueError("Prepared row fields do not match the explicit raw schema")
    ensure_dataset(client, config)
    job_config = bigquery.LoadJobConfig(
        schema=schema,
        autodetect=False,
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        create_disposition=bigquery.CreateDisposition.CREATE_IF_NEEDED,
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        ignore_unknown_values=False,
        max_bad_records=0,
    )
    job = client.load_table_from_json(
        rows, config.table_id, job_config=job_config, location=config.location
    )
    job.result()
    if job.output_rows != len(rows):
        raise RuntimeError(f"BigQuery loaded {job.output_rows} rows; expected {len(rows)}")
    return job


def prediction_schema() -> list[bigquery.SchemaField]:
    """Explicit schema shared by offline labeled batches and future unlabeled rows."""
    fields = (
        ("prediction_id", "STRING", "REQUIRED"), ("sample_id", "STRING", "REQUIRED"),
        ("run_id", "STRING", "REQUIRED"), ("model_version", "STRING", "REQUIRED"),
        ("dataset_version", "STRING", "REQUIRED"), ("source", "STRING", "REQUIRED"),
        ("split", "STRING", "NULLABLE"), ("scored_at", "TIMESTAMP", "REQUIRED"),
        ("failure_score", "FLOAT64", "REQUIRED"), ("predicted_failure", "BOOL", "REQUIRED"),
        ("threshold", "FLOAT64", "REQUIRED"), ("actual_failure", "INT64", "NULLABLE"),
    )
    return [bigquery.SchemaField(name, kind, mode=mode) for name, kind, mode in fields]


def validate_test_predictions(rows: list[dict]) -> None:
    """Reject malformed or mixed runs before performing any cloud mutation."""
    fields = {field.name for field in prediction_schema()}
    if not rows or any(set(row) != fields for row in rows):
        raise ValueError("TEST batch fields do not match the explicit prediction schema")
    if len({row["sample_id"] for row in rows}) != len(rows) or len({row["prediction_id"] for row in rows}) != len(rows):
        raise ValueError("TEST batch sample and prediction IDs must be unique")
    for name in ("run_id", "model_version", "dataset_version", "scored_at", "threshold"):
        if any(row[name] != rows[0][name] for row in rows):
            raise ValueError("TEST batch must describe one explicit run and frozen threshold")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", rows[0]["run_id"]):
        raise ValueError("Batch run_id must be a simple nonempty identifier")
    for row in rows:
        if any(not isinstance(row[name], str) or not row[name].strip() for name in (
            "prediction_id", "sample_id", "run_id", "model_version", "dataset_version"
        )):
            raise ValueError("TEST batch identifiers must be nonempty strings")
        if row["source"] != "batch" or row["split"] != "test":
            raise ValueError("Only source=batch and split=test may be persisted by this command")
        if type(row["actual_failure"]) is not int or row["actual_failure"] not in (0, 1):
            raise ValueError("Offline TEST labels must be PASS=0 or FAIL=1")
        for name in ("failure_score", "threshold"):
            if (type(row[name]) not in (int, float) or not math.isfinite(row[name])
                    or not 0 <= row[name] <= 1):
                raise ValueError(f"{name} must be a finite model score in [0, 1]")
        if (type(row["predicted_failure"]) is not bool
                or row["predicted_failure"] != (row["failure_score"] >= row["threshold"])):
            raise ValueError("Predicted class differs from the frozen threshold")
        timestamp = datetime.fromisoformat(row["scored_at"])
        if timestamp.tzinfo is None or timestamp.utcoffset() != timedelta(0):
            raise ValueError("Batch scored_at must be UTC")


def quality_summary_sql(config: BigQueryConfig, run_id: str) -> str:
    """Render a logical view pinned to exactly one explicitly published batch run."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", run_id):
        raise ValueError("Invalid explicit batch run ID")
    if not re.fullmatch(r"[a-z][a-z0-9-]{4,28}[a-z0-9]", config.project_id):
        raise ValueError("Invalid project ID")
    if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,1023}", value)
           for value in (config.dataset, config.predictions_table)):
        raise ValueError("Invalid reporting resource identifier")
    if len({config.table_id, config.predictions_table_id, config.summary_view_id}) != 3:
        raise ValueError("Reporting resources must not replace the raw table or each other")
    path = Path(__file__).resolve().parents[2] / "sql/create_quality_summary.sql"
    return Template(path.read_text(encoding="utf-8")).substitute(
        GCP_PROJECT_ID=config.project_id, BQ_DATASET=config.dataset,
        BQ_PREDICTIONS_TABLE=config.predictions_table, BATCH_RUN_ID=run_id,
    )


def batch_summary(rows: list[dict]) -> dict:
    """Expected report values from saved predictions; never query or score a model."""
    failures = sum(row["actual_failure"] for row in rows)
    flagged = sum(row["predicted_failure"] for row in rows)
    true_positive = sum(row["predicted_failure"] and row["actual_failure"] == 1 for row in rows)
    return {
        "run_id": rows[0]["run_id"], "sample_count": len(rows), "observed_fail_count": failures,
        "observed_fail_rate": failures / len(rows), "flagged_count": flagged,
        "mean_failure_score": sum(row["failure_score"] for row in rows) / len(rows),
        "recall": true_positive / failures if failures else 0.0,
        "precision": true_positive / flagged if flagged else 0.0,
    }


def _read_run(client: bigquery.Client, config: BigQueryConfig, run_id: str) -> list[dict]:
    fields = ", ".join(field.name for field in prediction_schema())
    query = f"SELECT {fields} FROM `{config.predictions_table_id}` WHERE run_id = @run_id ORDER BY sample_id"
    job_config = bigquery.QueryJobConfig(
        maximum_bytes_billed=100_000_000,
        query_parameters=[bigquery.ScalarQueryParameter("run_id", "STRING", run_id)],
    )
    return [dict(row) for row in client.query(query, job_config=job_config, location=config.location).result()]


def _verify_run(expected: list[dict], actual: list[dict]) -> None:
    if len(actual) != len(expected):
        raise RuntimeError(f"BigQuery run has {len(actual)} rows; expected {len(expected)}")
    for left, right in zip(sorted(expected, key=lambda row: row["sample_id"]),
                           sorted(actual, key=lambda row: row["sample_id"])):
        for field, value in left.items():
            saved = right[field]
            if field == "scored_at":
                if not isinstance(saved, datetime):
                    saved = datetime.fromisoformat(saved)
                matches = datetime.fromisoformat(value) == saved
            elif field in ("failure_score", "threshold"):
                matches = math.isclose(value, saved, rel_tol=0, abs_tol=1e-12)
            else:
                matches = value == saved
            if not matches:
                raise RuntimeError(f"Persisted {field} differs for sample {left['sample_id']}")


def persist_test_batch(client: bigquery.Client, config: BigQueryConfig, rows: list[dict]) -> dict:
    """Append cached TEST rows and verify the explicit-run table and reporting view."""
    validate_test_predictions(rows)
    run_id = rows[0]["run_id"]
    sql = quality_summary_sql(config, run_id)
    dataset = client.get_dataset(config.dataset_id)
    if dataset.location.casefold() != config.location.casefold():
        raise ValueError("Reporting dataset location differs from configured BQ_LOCATION")
    table = client.create_table(bigquery.Table(config.predictions_table_id, schema=prediction_schema()), exists_ok=True)
    aliases = {"INT64": "INTEGER", "FLOAT64": "FLOAT", "BOOL": "BOOLEAN"}

    def signature(schema):
        return [(field.name, aliases.get(field.field_type, field.field_type), field.mode) for field in schema]

    if signature(table.schema) != signature(prediction_schema()):
        raise ValueError("Existing quality_predictions schema differs from the explicit schema")
    existing = _read_run(client, config, run_id)
    rows_written = 0
    job_id = f"test_batch_{hashlib.sha256(f'{config.predictions_table_id}:{run_id}'.encode()).hexdigest()[:32]}"
    if not existing:
        try:
            job = client.get_job(job_id, location=config.location)
        except NotFound:
            job_config = bigquery.LoadJobConfig(
                schema=prediction_schema(), autodetect=False,
                source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
                create_disposition=bigquery.CreateDisposition.CREATE_NEVER,
                write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
                ignore_unknown_values=False, max_bad_records=0,
            )
            try:
                job = client.load_table_from_json(
                    rows, config.predictions_table_id, job_id=job_id,
                    job_config=job_config, location=config.location,
                )
            except Conflict:
                job = client.get_job(job_id, location=config.location)
        job.result()
        if job.output_rows != len(rows):
            raise RuntimeError(f"Load job wrote {job.output_rows} rows; expected {len(rows)}")
        rows_written = len(rows)
        existing = _read_run(client, config, run_id)
    _verify_run(rows, existing)
    client.query(sql, job_config=bigquery.QueryJobConfig(maximum_bytes_billed=100_000_000),
                 location=config.location).result()
    view = client.get_table(config.summary_view_id)
    if view.table_type != "VIEW" or not view.view_query:
        raise RuntimeError("quality_summary is not a verified logical view")
    expected_where = f"run_id = '{run_id}'"
    if expected_where not in view.view_query or "source = 'batch'" not in view.view_query or "split = 'test'" not in view.view_query:
        raise RuntimeError("Reporting view is not pinned to the explicit TEST batch run")
    summary_rows = list(client.query(
        f"SELECT * FROM `{config.summary_view_id}`",
        job_config=bigquery.QueryJobConfig(maximum_bytes_billed=100_000_000), location=config.location,
    ).result())
    if len(summary_rows) != 1:
        raise RuntimeError("Reporting view must return one explicit run")
    summary = dict(summary_rows[0])
    expected = batch_summary(rows)
    for key, value in expected.items():
        actual = summary.get(key)
        if isinstance(value, float):
            matches = actual is not None and math.isclose(value, actual, rel_tol=0, abs_tol=1e-12)
        else:
            matches = value == actual
        if not matches:
            raise RuntimeError(f"Reporting view {key} differs from cached batch")
    return {"run_id": run_id, "rows_written": rows_written, "rows_verified": len(existing),
            "load_job_id": job_id, "predictions_table": config.predictions_table_id,
            "summary_view": config.summary_view_id, "location": config.location, "summary": summary}
