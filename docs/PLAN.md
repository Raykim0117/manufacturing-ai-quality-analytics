# Semiconductor Manufacturing AI Quality Analytics — MVP Plan

## 1. Objective

Build a reproducible, end-to-end portfolio demonstration for a DB HiTek IT position focused on AI automation and manufacturing data analytics. Use semiconductor process measurements to rank samples by failure risk, explain model associations, serve predictions, and expose results to business users through BigQuery Connected Sheets.

Limit implementation to **10 hours**, including setup, verification, deployment, and documentation. Favor readable Python modules, manual execution, and one deployed service. Success means demonstrating the full data-to-business workflow and reporting results honestly; it does not require a particular model score or prove production readiness.

Repository inspection: only `AGENTS.md`, Git metadata, and a Python 3.10 `.venv/` currently exist. There are no commits, application modules, tests, dependency manifests, or cloud configuration. Everything below is proposed work; this document creates no application or infrastructure.

## 2. Architecture

```text
UCI SECOM files
  → BigQuery raw measurements
  → Python: split, preprocess, train Logistic Regression and XGBoost
  → held-out evaluation + offline SHAP explanations
  → BigQuery batch predictions → reporting view → Connected Sheets

Saved preprocessing + selected model + schema + threshold
  → FastAPI → Docker → Artifact Registry → Cloud Run
  → online predictions stored in BigQuery → reporting view
```

Training and batch scoring run manually on the developer machine using CPU. Package the evaluated model artifacts inside the API image; load them once at startup. Cloud Run performs inference and persistence only. SHAP runs offline to keep request latency and deployment complexity small.

## 3. Repository Structure

Proposed files and responsibilities; create them during implementation only:

```text
AGENTS.md                     contributor guidelines
README.md                     setup, execution, results, demo, cleanup
requirements.txt              compatible, pinned runtime dependencies
.env.example                  non-secret configuration placeholders
.gitignore / .dockerignore    exclude credentials, caches, local data
Dockerfile                    API image with explicitly copied model artifacts
src/quality_analytics/
  config.py                   environment configuration
  ingest.py                   SECOM parsing and raw BigQuery load
  warehouse.py                small BigQuery read/write helpers
  preprocessing.py            fitted, reusable transformations
  train.py                    both models, evaluation, artifacts, offline SHAP
  predict.py                  shared inference and batch scoring
  api.py                      FastAPI schemas and endpoints
sql/                          table definitions and reporting view
tests/                        critical-path unittest tests
data/                         ignored source files and local normalized cache
artifacts/                    ignored model bundles and schema metadata
reports/                      metrics, PR curve, SHAP figures
docs/PLAN.md                  this plan
```

Use one Python package and ordinary CLI module entry points. Add generated reports selectively after review; do not commit raw data or model binaries. Avoid extra abstraction layers, notebooks required for execution, or separate services.

## 4. Data Flow

1. Download `secom.data` and `secom_labels.data` from UCI. Preserve row alignment; record source URL, checksum, parser version, row counts, class counts, and parsed feature width. Cite the dataset and its CC BY 4.0 license. UCI reports 1,567 samples, 104 failures, missing values, and pass/fail labels `-1`/`1`. Verify actual file dimensions rather than hard-coding the catalog's feature count. [UCI SECOM](https://archive.ics.uci.edu/dataset/179/secom)
2. Assign stable `sample_id` values from dataset version and source row number. Map pass to `0` and fail to `1`. Parse timestamps explicitly; retain source time as timezone-unspecified `DATETIME`, without inventing a timezone.
3. Load `secom_raw` with `dataset_version`, `sample_id`, `test_time`, `failure_label`, and ordered nullable `FLOAT64` columns named `feature_000`, `feature_001`, etc. Use an explicit generated schema. For this single-dataset MVP, replace the raw snapshot only through an explicit ingestion command.
4. Query raw rows from BigQuery into Python, ordered by stable ID. Local mode reads the same normalized schema from disk. Save split membership and exclude labels, IDs, and timestamps from predictors.
5. Save preprocessing, model, raw/retained feature names, threshold, dependency versions, dataset version, seed, and evaluation metadata as one versioned bundle. Batch-score the held-out test samples for the reporting demo; mark these rows `split=test` and `source=batch`.
6. Append batch and API results to `quality_predictions`: `prediction_id`, `sample_id`, `run_id`, `model_version`, `dataset_version`, `source`, nullable `split`, UTC `scored_at`, `failure_score`, `predicted_failure`, `threshold`, and nullable `actual_failure`. Labels accompany offline evaluation only; the API accepts no label.
7. Create `quality_summary` for the latest demo batch run: sample count, flagged count, observed failure count/rate, mean risk score, recall, and precision. Separate API traffic from labeled evaluation. Connected Sheets displays summary KPIs, a risk-ranked table, and one simple chart. Predicted risk is not measured yield; omit invented tool, lot, or sensor names.

## 5. ML Methodology

- **Positive class:** manufacturing failure. Its reported prevalence is about 6.6%, so accuracy is not the primary metric and should not headline the portfolio.
- **Split:** fixed seed `42`, stratified 60% training / 20% validation / 20% test. Persist IDs and class counts. Do not use the test set to select features, hyperparameters, models, or thresholds. This random split is a portfolio assumption, not evidence of performance on future production periods.
- **Preprocessing:** on training rows only, drop columns with more than 50% missing values and all-empty columns; median-impute remaining columns and remove constants. Standardize for Logistic Regression; XGBoost uses the same retained/imputed features without scaling. Persist each fitted pipeline and preserve feature order at inference. Never refit transformations on validation, test, or API data.
- **Baseline:** regularized Logistic Regression with `class_weight=balanced` and a sufficient iteration limit. Verify convergence.
- **Challenger:** CPU XGBoost with shallow trees, a bounded tree count, fixed seed, and `scale_pos_weight` calculated from training class counts. Allow at most two small configurations; no broad search, synthetic oversampling, or repeated tuning against the holdout.
- **Primary metric:** PR-AUC from failure scores using `precision_recall_curve` followed by trapezoidal `auc(recall, precision)`. Record this definition explicitly. Average precision uses a different integration rule and must not be silently substituted or mixed into comparisons. Show the PR curve and split prevalence as a reference. [scikit-learn metric definitions](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html)
- **Selection:** choose the model with higher validation PR-AUC; exact ties favor Logistic Regression. Train and report both even if XGBoost does not outperform the baseline. Freeze the selected fitted bundle before final test evaluation; avoid a train-plus-validation refit that invalidates the chosen threshold.
- **Threshold:** on validation only, choose maximum precision subject to recall at least `0.80`; resolve ties by higher recall, then higher threshold. This is a demo operating assumption, not a promised test result or manufacturing-approved requirement. Report the tradeoff against threshold `0.5`.
- **Secondary metrics:** failure-class recall, precision, F1, and ROC-AUC on the same test set; include confusion-matrix counts and class support. Report zero-division handling explicitly. Do not invent score targets or hide weak results.
- **SHAP:** explain XGBoost using TreeExplainer; if Logistic Regression is selected for serving, also provide its linear SHAP explanation. Use a small training-only background where required and a fixed, capped held-out sample. Save global mean-absolute importance and one local example with original feature IDs and documented output units. SHAP values describe model associations, not causes, root causes, or actionable process settings.

## 6. GCP Resources Required

| Resource | MVP purpose |
| --- | --- |
| One billing-enabled GCP project | Own resources and run queries |
| BigQuery dataset `secom_quality` | Raw table, prediction table, reporting view |
| Artifact Registry Docker repository `quality-api` | Store manually built API image |
| Cloud Run service `quality-api` | Serve the packaged model |
| Dedicated runtime service account | Write prediction rows without downloaded keys |
| Google Sheet with Connected Sheets access | Business-user reporting demo |

Enable BigQuery, Artifact Registry, and Cloud Run APIs. Choose one supported region for dataset, registry, and service where practical. Build locally with Docker and push manually; no Cloud Build pipeline, storage bucket, or hosted training is required.

Use local Application Default Credentials (`gcloud auth application-default login`) and Cloud Run's attached service identity. Never download or commit service account keys. Grant the runtime identity BigQuery Data Editor on the prediction table only; direct row inserts need no query-job role. The developer needs query/load access, registry upload access, deployment access, and permission to attach the runtime identity. Keep provisioning authority separate from runtime permissions. [Cloud Run service identity](https://docs.cloud.google.com/run/docs/configuring/services/service-identity)

Require authenticated Cloud Run invocation using IAM and an identity token. Grant report users BigQuery Data Viewer on reporting resources and BigQuery Job User on the billing project as needed. Check the account's Connected Sheets availability and refresh permissions during the first hour. [Connected Sheets requirements](https://support.google.com/docs/answer/9702507)

Configuration variables: `APP_ENV`, `DATA_BACKEND=local|bigquery`, `GCP_PROJECT_ID`, `BQ_DATASET`, `BQ_LOCATION`, `BQ_RAW_TABLE`, `BQ_PREDICTIONS_TABLE`, `GCP_REGION`, `MODEL_PATH`, `PERSIST_PREDICTIONS`, and `PORT`. Validate required cloud settings only in cloud mode. Local inference defaults to persistence disabled and must not require GCP credentials. Keep `.env` and ADC files outside version control and Docker context.

Bind the container to `0.0.0.0` on the injected `PORT`; start with zero minimum instances and a low maximum instance count. Set a billing alert, limit query bytes, and document resource cleanup; alerts are not spending caps. [Cloud Run container contract](https://docs.cloud.google.com/run/docs/container-contract)

## 7. API Contract

| Endpoint | Contract |
| --- | --- |
| `GET /health` | `200` after the bundle loads; returns status and model version; performs no BigQuery query |
| `GET /model-info` | Returns raw feature names/order/count, positive-class meaning, threshold, and model version |
| `POST /predict` | Scores one sample and persists it when configured; returns `200` on success |

Prediction request example (shortened for documentation):

```json
{"sample_id": "demo-001", "features": [0.12, null, -1.4]}
```

The actual array must contain exactly the bundle's raw feature count in published order. Accept finite JSON numbers or `null`; impute nulls through the saved pipeline. Reject wrong lengths, nonnumeric values, nonfinite values, empty IDs, and extra fields with `422`. Do not accept already transformed features, labels, or user-supplied thresholds.

Example response (illustrative values, not measured results):

```json
{
  "prediction_id": "generated-uuid",
  "sample_id": "demo-001",
  "failure_score": 0.23,
  "predicted_failure": true,
  "threshold": 0.20,
  "model_version": "selected-model-v1",
  "stored": true
}
```

Classify as failure when the score is greater than or equal to the stored threshold. Treat `failure_score` as an uncalibrated model score, not a certified probability. Return `stored=false` in local mode. When persistence is enabled, acknowledge success only after BigQuery accepts the row; return `503` for warehouse failures and log a sanitized error. UUIDs provide traceability; exactly-once delivery and idempotent client retries are outside MVP scope. IAM rejects unauthorized cloud calls before application handling.

## 8. Definition of Done

- [ ] Reproducible SECOM ingestion into BigQuery with alignment, schema, provenance, and label checks.
- [ ] Both models trained without preprocessing leakage; frozen split and versioned bundle saved.
- [ ] Comparison report leads with PR-AUC and includes all four secondary metrics, threshold rationale, PR curve, confusion matrix, and limitations.
- [ ] Global and local SHAP artifacts use association language and original feature IDs.
- [ ] Held-out batch predictions are queryable in BigQuery, with model version, threshold, and labels distinguished from online scores.
- [ ] Local API and Docker container return matching scores for a fixed sample; invalid input is rejected.
- [ ] Image pushed to Artifact Registry; authenticated Cloud Run request succeeds and its prediction appears in BigQuery.
- [ ] Connected Sheets refreshes from the reporting view and shows KPIs plus a risk-ranked table/chart.
- [ ] Critical tests pass: source/label alignment and mapping; training-only preprocessing and feature order; artifact round-trip score parity; API validation and persistence-failure behavior. Use `unittest` and mocked cloud boundaries; do not add broad coverage targets.
- [ ] README contains exact commands, environment settings, dataset attribution, measured results, architecture, demo evidence, assumptions, and cleanup instructions. Review repository and image context for secrets.

Do not claim full completion if GCP deployment or Connected Sheets is blocked. Record the specific unmet item and available local evidence.

## 9. 10-Hour Execution Checklist

| Elapsed time | Budget | Work and exit evidence |
| --- | --- | --- |
| 0:00–1:00 | 1.00 h | Check billing/IAM/Sheets access and Docker; scaffold package, environment examples, ignore rules, compatible dependencies; parse and validate SECOM |
| 1:00–2:00 | 1.00 h | Create warehouse schema, load raw data, verify row/class counts and local/cloud schema parity |
| 2:00–3:00 | 1.00 h | Persist split; implement fitted preprocessing and Logistic Regression; save validation metrics |
| 3:00–4:00 | 1.00 h | Train bounded XGBoost candidates; select model and threshold; freeze bundle and evaluate holdout |
| 4:00–4:45 | 0.75 h | Produce capped offline SHAP analysis, PR curve, and comparison report |
| 4:45–5:30 | 0.75 h | Batch-score heldout, store predictions, create and verify reporting view |
| 5:30–6:30 | 1.00 h | Implement shared inference and FastAPI contract; verify local requests and optional persistence |
| 6:30–7:45 | 1.25 h | Build/test Docker, push image, deploy authenticated Cloud Run, confirm API-to-BigQuery write |
| 7:45–8:30 | 0.75 h | Connect Sheets, build minimal KPIs/table/chart, verify refresh and permissions |
| 8:30–9:15 | 0.75 h | Complete critical tests and end-to-end smoke checks; fix blocking failures |
| 9:15–10:00 | 0.75 h | Finish README, screenshots and demo narrative; inspect secrets, document costs/cleanup and unresolved limits |
| **Total** | **10.00 h** | Stop at the time limit and report actual completion |

Create critical tests alongside the relevant path; the verification block completes the final checks. At each time boundary, stop optional refinement. Reduce tuning, SHAP sample size, and visual polish before sacrificing integration or evaluation validity. Escalate access blockers early; an exported spreadsheet is fallback evidence, not a completed Connected Sheets integration. Do not expand scope to compensate for blocked cloud access.

## 10. Risks and Scope Exclusions

Assumptions and limitations:

- Dataset access, local Docker, GCP billing/permissions, and Sheets access are available or resolvable in the first hour; account setup can consume the budget.
- SECOM is small, old, imbalanced, and anonymized. Few held-out failures make metrics unstable; random splitting may overlook temporal drift or correlated production samples. No confidence-interval study, factory generalization claim, or verified leakage-free measurement timing is promised.
- Unknown sensor meanings and timestamp timezone limit operational interpretation. SHAP associations do not identify physical causes. Decisions require process-engineer review; there is no automatic production intervention.
- Class weighting can distort calibration. Scores support ranking and a documented demo threshold; false-negative and false-positive costs are assumed, not validated business costs.
- Dependency compatibility, model serialization, container size, and cold starts can block deployment. Pin compatible versions early and load only trusted, locally generated artifacts.
- Manual batch reruns may create duplicate prediction rows; reporting must filter one explicit run. API retries can also duplicate writes. No transactional workflow or exactly-once guarantee.
- Cloud billing, IAM errors, and Connected Sheets access/refresh constraints can prevent a full demo. Document blockers, bound usage, and provide cleanup steps.

Explicitly excluded: Kubernetes, Airflow, Terraform, CI/CD, Streamlit, React, LLMs, deep learning, GPUs, Vertex AI, automated retraining, live factory feeds, streaming infrastructure, custom authentication, online SHAP, elaborate dashboards, extensive hyperparameter optimization, causal discovery, and production monitoring systems. Use a single API service, manual CLI execution, minimal SQL, and critical-path tests.
