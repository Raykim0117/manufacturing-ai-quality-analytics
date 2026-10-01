"""Only the BigQuery operations required to load the SECOM raw snapshot."""

from google.cloud import bigquery

from .config import BigQueryConfig


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
