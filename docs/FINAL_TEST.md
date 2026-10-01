# Final TEST evaluation and BigQuery reporting batch

The selected XGBoost candidate 1, preprocessing, 450 retained feature IDs/order,
and operating threshold 0.08485201001167297 were frozen before this evaluation.
The existing persisted TEST membership was loaded without regeneration. No
model or preprocessing was retrained, refitted, selected, or tuned on TEST.

The first and only final scoring call transformed and predicted all 314 TEST
rows once. The same cached scores produce both frozen-threshold metrics and the
threshold-0.5 reference. Only the frozen-threshold classifications were uploaded.
Original selected/baseline artifact files retained their SHA-256 checksums.

## Final metrics

Failure is positive. PR-AUC uses precision_recall_curve followed by trapezoidal
auc(recall, precision); average precision is not substituted. Recall, precision,
and F1 use zero_division=0. There are 21 failures and 293 passes, with observed
failure prevalence 0.06687898089171974.

| Metric | Frozen threshold | Threshold 0.5 reference |
| --- | ---: | ---: |
| PR-AUC | 0.277825 | 0.277825 |
| Recall | 0.952381 | 0.285714 |
| Precision | 0.076336 | 0.300000 |
| F1 | 0.141343 | 0.292683 |
| ROC-AUC | 0.768243 | 0.768243 |

Confusion matrices have actual rows and predicted columns, both [PASS, FAIL]:

- Frozen threshold: [[51, 242], [1, 20]]; TN=51, FP=242, FN=1, TP=20.
- Reference threshold: [[279, 14], [15, 6]]; TN=279, FP=14, FN=15, TP=6.

Threshold 0.5 is a reference only. The selected threshold remains unchanged.
The frozen operating point flags 262 rows, including 242 false positives.
High recall here is not a production guarantee. This random held-out split does
not establish future manufacturing performance. Scores are uncalibrated model
scores and are never described as measured manufacturing yield.

## One-time evaluation and retry commands

Run commands from the repository root in PowerShell. The evaluation below has
already been completed and must not be rerun:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
& .\.venv\Scripts\python.exe -m quality_analytics.batch --evaluate-once
```

The CLI requires the verified data/normalized/secom.jsonl cache and existing
selected-model-v1/selected.pkl and logistic-regression-v1/split_membership.json
under artifacts/<dataset_version>/. Missing or changed inputs fail validation.
An exclusive evaluation_guard.json claim is created before scoring. Repeating
--evaluate-once fails before loading/scoring the model. A failure after the
claim remains guarded; there is no force/re-evaluation option. If a process
fails before saving scores, investigate the guarded state rather than rescoring.

The ignored artifacts/secom-01b3bed261fc1223/final-test-v1/ directory contains:

- evaluation_guard.json: permanent evaluation claim and frozen identity.
- test_predictions.jsonl: 314 saved labeled TEST predictions with stable UUID5
  prediction IDs, sample IDs, one run ID, UTC scoring time, scores, and threshold.
- evaluation.json: frozen-threshold and reference metrics, PR coordinates,
  dependencies, snapshot/bundle/split/cache hashes, and persistence status.
- persistence.json: verified BigQuery destination, load job, row counts, and
  reporting summary.

Reports/final_test_evaluation.json is the reviewed final evaluation and cloud
verification report. --report selects another report path. Persistence retries
use only the saved JSONL and evaluation metadata; they load no model, fit no
pipeline, and produce no new test scores:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
$env:GCP_PROJECT_ID = "mfg-ai-quality-2026"
$env:BQ_DATASET = "secom_quality"
$env:BQ_LOCATION = "asia-northeast3"
$env:BQ_RAW_TABLE = "secom_raw"
$env:BQ_PREDICTIONS_TABLE = "quality_predictions"
& .\.venv\Scripts\python.exe -m quality_analytics.batch --persist
& .\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

BigQuery uses existing ADC and process environment values. No credentials are
stored in the repository. No additional dependencies were needed for this stage.

## Verified BigQuery resources

Project mfg-ai-quality-2026 was discovered in the existing gcloud and ADC
configuration; the existing secom_quality dataset location was verified as
asia-northeast3 before writing. The raw secom_raw table is not written to.

- Table: mfg-ai-quality-2026.secom_quality.quality_predictions.
- Logical view: mfg-ai-quality-2026.secom_quality.quality_summary.
- Explicit run: test-20261001T141318Z-e1b441ac.
- Load job: test_batch_2bdaef07865619875aa256aff3feefff.
- Rows written and individually verified: 314.

The table uses the explicit schema in sql/create_predictions_table.sql:
prediction_id, sample_id, run_id, model_version, dataset_version, source, split,
scored_at, failure_score, predicted_failure, threshold, and actual_failure.
Scores are FLOAT64, flags BOOL, scoring times UTC TIMESTAMP, and offline labels
INT64. split and actual_failure are nullable to preserve the planned schema.
This batch command nevertheless requires every row to have source=batch,
split=test, and an actual PASS=0/FAIL=1 label. It refuses raw/report resource
collisions and incompatible existing prediction schemas.

Loads append with strict schema, no autodetection, and zero bad records. A stable
run-specific load-job ID supports retries. Existing rows for this run are read
and verified before any new append, preventing ordinary retries from duplicating
this cached batch. This is an MVP retry guard, not a general exactly-once
distributed delivery guarantee. Persisted sample IDs, labels, scores, flags,
thresholds, timestamps, and prediction/run/model/dataset IDs are checked against
every saved row.

The view in sql/create_quality_summary.sql filters the exact explicit run ID,
source=batch, split=test, and non-null labels. Publishing this run sets the view
to the current explicit batch; it does not choose a run with MAX(scored_at) or
combine batches and future API traffic. Its verified single row contains:

| Field | Value |
| --- | ---: |
| sample_count | 314 |
| observed_fail_count | 21 |
| observed_fail_rate | 0.06687898089171974 |
| flagged_count | 262 |
| mean_failure_score | 0.2219596259046798 |
| recall | 0.9523809523809523 |
| precision | 0.07633587786259542 |

Observed failure fields use actual labels. Flagged counts and mean failure
scores are model outputs. No predicted failure rate is named or interpreted as
manufacturing yield. The view is ready for a later Connected Sheets step; no
Sheets UI, service, or deployment was implemented here.

## Verification

All 56 unittest tests passed, including eight new batch-stage tests. Synthetic
fixtures test one-call scoring, permanent evaluation guards, frozen threshold,
no fitting or split regeneration, exact TEST membership and label preservation,
cached-score integrity, prediction schema/naming, append/retry behavior, explicit
run filtering, and persistence failure/mismatch handling. Existing real TEST
samples are not evaluated by these tests. The existing full suite uses real
source and validation artifacts only where previously established.

Dependency consistency and git diff whitespace checks passed. Live BigQuery
verification confirmed the prediction schema, all 314 persisted rows, logical
view type and explicit-run filter, and each summary value against the cached
batch. No further model selection or threshold work follows this final report.
