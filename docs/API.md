# Frozen-model inference service

The API loads the existing trusted `selected.pkl` once during startup. It serves
the frozen XGBoost candidate 1, its train-fitted preprocessing, raw/retained
feature ordering, and operating threshold. Startup checks schema and retained
feature order against the bundle. No dataset is loaded, scored, or evaluated by
this service; only submitted samples are scored. No split is regenerated and no
training, fitting, threshold selection, or SHAP computation runs in the API.

Current bundle: `selected-model-v1`, dataset version `secom-01b3bed261fc1223`,
590 ordered raw features, 450 retained features, frozen threshold
`0.08485201001167297`. Failure is the positive class (`FAIL=1`, `PASS=0`).
Classify as failure when `failure_score >= threshold`. The score is uncalibrated
model output; it is not measured manufacturing yield.

## Local execution

Run from the repository root in PowerShell. The repository does not auto-load
`.env` or `.env.example`.

```powershell
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = "$PWD\src"
$env:APP_ENV = "local"
$env:MODEL_PATH = "$PWD\artifacts\secom-01b3bed261fc1223\selected-model-v1\selected.pkl"
$env:PERSIST_PREDICTIONS = "false"
$env:PORT = "8080"
& .\.venv\Scripts\python.exe -m uvicorn quality_analytics.api:create_app --factory --host 127.0.0.1 --port $env:PORT
```

Stop the local server with Ctrl+C. Local mode requires no Google credentials or
cloud configuration. `PERSIST_PREDICTIONS` defaults to `false` and accepts only
`true` or `false` (case-insensitive). If `MODEL_PATH` is omitted, exactly one
`artifacts/*/selected-model-v1/selected.pkl` must exist; otherwise set it explicitly.
Only operator-configured, trusted local pickle bundles may be loaded.

## Endpoints

| Endpoint | Result |
| --- | --- |
| `GET /health` | `status=ok` and `model_version` after startup; no BigQuery query or credential discovery |
| `GET /model-info` | `model_version`, `raw_feature_count`, ordered `raw_feature_names`, `positive_class`, frozen `threshold` |
| `POST /predict` | One inference, with optional synchronous persistence; returns the fields below |

Requests contain only `sample_id` and `features`. `sample_id` must be a nonempty
string containing a non-whitespace character. `features` must contain exactly
590 finite JSON numbers or `null`, ordered from `feature_000` through
`feature_589`. Nulls use the saved training medians. Wrong length, NaN/infinity,
strings, booleans, missing fields, empty IDs, and extra fields return `422` before
scoring or persistence. Labels and request-supplied thresholds are extra fields
and are rejected. Validation error responses omit submitted values, including
nonfinite numbers that cannot safely be serialized into JSON.

In another PowerShell terminal, this creates a complete synthetic all-null
request with 590 entries (a contract demonstration, not a dataset evaluation):

```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8080/health"
Invoke-RestMethod -Uri "http://127.0.0.1:8080/model-info"
$body = @{ sample_id = "demo-001"; features = [object[]]::new(590) } | ConvertTo-Json -Depth 3 -Compress
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8080/predict" -ContentType "application/json" -Body $body
```

Verified local response for that synthetic input; the UUID changes each request:

```json
{
  "prediction_id": "340e1d12-76a7-4a05-aac3-33e528e611e3",
  "sample_id": "demo-001",
  "failure_score": 0.46548160910606384,
  "predicted_failure": true,
  "threshold": 0.08485201001167297,
  "model_version": "selected-model-v1",
  "stored": false
}
```

## Optional persistence

Set `PERSIST_PREDICTIONS=true` explicitly to enable persistence. Only in that
mode must `GCP_PROJECT_ID`, `BQ_DATASET`, `BQ_LOCATION`, and `BQ_RAW_TABLE` be
configured; `BQ_PREDICTIONS_TABLE` defaults to `quality_predictions`. Use the
existing prediction table, never the raw table. Authentication uses Application
Default Credentials and requires write permission on the prediction table.
The service creates no tables, views, or query jobs. Credentials are discovered
lazily on the first prediction write, so health and model-info do not contact
BigQuery even in persistence mode.

Each successful request inserts one row with the existing 12-field schema:
`prediction_id`, `sample_id`, `run_id`, `model_version`, `dataset_version`,
`source`, `split`, `scored_at`, `failure_score`, `predicted_failure`, `threshold`,
`actual_failure`. The service uses a UUID prediction ID, an `api-<uuid>` run ID
per application lifetime, UTC scoring time, `source="api"`, `split=NULL`, and
`actual_failure=NULL`. The existing explicit TEST batch reporting view excludes
these API rows.

The API returns `stored=true` only after BigQuery accepts the row. SDK exceptions,
credential failures, or per-row insert errors return `503` with
`{"detail":"Prediction persistence unavailable"}`. Logs contain a sanitized
message and omit submitted data and provider exception details. UUID insert IDs
provide traceability; exactly-once delivery and idempotent client retries remain
outside the MVP scope. Online persistence was verified using mocked BigQuery
boundaries only; this stage performed no cloud writes.

## Verification

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

All 70 tests passed, including 14 API tests. Checks cover endpoint responses,
strict validation, null imputation, saved-bundle score parity, raw/retained order,
threshold equality, no fitting or model/bundle mutation, one startup load,
credential-free local mode, online schema, and persistence failure handling.
API tests submit synthetic measurements only. Existing evaluation tests use
synthetic fixtures; the real held-out TEST evaluation was not repeated.
A real local Uvicorn HTTP smoke check passed for health, model-info, and the
synthetic all-null prediction. `pip check` found no broken dependencies.

Implementation references: [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/),
[Pydantic strict validation](https://docs.pydantic.dev/latest/concepts/strict_mode/),
and [BigQuery JSON row inserts](https://docs.cloud.google.com/python/docs/reference/bigquery/latest/google.cloud.bigquery.client.Client#google_cloud_bigquery_client_Client_insert_rows_json).
