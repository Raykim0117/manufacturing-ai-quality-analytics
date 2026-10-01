# Logistic Regression baseline

This stage reads a normalized SECOM snapshot, persists a stratified split, fits
preprocessing and Logistic Regression on training rows, and evaluates validation
at threshold 0.5. It does not evaluate the test set or tune the threshold.
Training reads BigQuery with a SELECT ordered by sample_id; it never creates,
replaces, or updates the raw table. Queries have a 100 MB billing limit.

Run from the repository root in PowerShell:

```powershell
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
$env:GCP_PROJECT_ID = "your-gcp-project-id"
$env:BQ_DATASET = "secom_quality"
$env:BQ_LOCATION = "your-bigquery-region"
$env:BQ_RAW_TABLE = "secom_raw"
& .\.venv\Scripts\python.exe -m quality_analytics.train --backend bigquery
```

BigQuery uses existing Application Default Credentials. Configuration is read
from the process environment; .env.example is not loaded automatically. BigQuery
is the default backend, with an optional DATA_BACKEND environment override.
There is no automatic fallback from BigQuery to local data.

Local verification needs neither credentials nor cloud settings. Export the
same normalized raw schema through the existing hash/dimension-validated parser:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
& .\.venv\Scripts\python.exe -m quality_analytics.dataset
& .\.venv\Scripts\python.exe -m quality_analytics.train --backend local
& .\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

The ignored data/normalized/secom.jsonl cache contains raw measurements with
missing values represented as null. Its creation does not transform features or
modify raw files or their validation report. --local-data selects another JSONL
file with the same schema. The training CLI requires 1,567 rows and 590 features;
unit tests can exercise the training functions using smaller synthetic datasets.

The split sorts stable sample IDs and uses random_state=42 for a stratified
60/40 split followed by a stratified half split of the held-out rows. Integer
rounding produces 940 train, 313 validation, and 314 test rows. Membership is
saved by sample_id and reused on reruns. A saved membership is rejected if its
dataset fingerprint, configuration, counts, or assignments change. The complete
normalized snapshot fingerprint detects changes in labels, timestamps, and
measurements; computing it fits no preprocessing and evaluates no test scores.

Preprocessing drops columns with more than 50% training missingness, including
all-empty columns. Exactly 50% remains eligible. It then uses training medians,
removes columns constant after training imputation, and uses training means and
standard deviations for scaling. It preserves original feature names and order.
Identifiers, dataset versions, timestamps, and labels are excluded from inputs.
Logistic Regression uses balanced class weights, lbfgs, C=1, max_iter=5000,
tol=0.0001, and seed 42. A convergence warning aborts training rather than being
reported as a successful baseline.

Artifacts live in ignored artifacts/<dataset_version>/logistic-regression-v1/:

- split_membership.json: ID assignments, counts, seed, and dataset fingerprint.
- baseline.pkl: fitted preprocessing pipeline, fitted model, feature/schema and
  dependency metadata, threshold, and validation metrics in one versioned bundle.
- retained_features.json: ordered original feature names.
- metadata.json: provenance, dataset identity, raw/retained feature lists, removed
  features, training parameters, iterations, split counts, and package versions.
- validation_metrics.json: threshold-0.5 metrics and PR curve coordinates.

Only load trusted, locally generated pickle artifacts. Prediction parity is
checked on validation rows after the bundle is saved and reloaded. --artifact-root
changes the artifact directory. reports/baseline_validation.json is the reviewed
validation report; --report selects another output path. No model binaries or
normalized datasets belong in Git.

PR-AUC is computed using precision_recall_curve followed by trapezoidal
auc(recall, precision), as defined by the scikit-learn
[curve](https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.precision_recall_curve.html)
and [integration](https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.auc.html)
APIs. Average precision is not substituted. Failure is the positive class.
Recall, precision, and F1 use zero_division=0. ROC-AUC uses failure scores.
Confusion-matrix rows are actual classes and columns are predicted classes,
both ordered PASS=0, FAIL=1. Threshold 0.5 is fixed and has not been optimized.

## Verified local results

No BigQuery configuration was supplied for this verification run. These results
come from the locally normalized, hash-validated SECOM source, not a live cloud
read. The BigQuery boundary is covered with mocked read-only query tests.

Dataset version: secom-01b3bed261fc1223.

| Split | Rows | FAIL | PASS |
| --- | ---: | ---: | ---: |
| Train | 940 | 62 | 878 |
| Validation | 313 | 21 | 292 |
| Test | 314 | 21 | 293 |

There are 590 raw features and 450 retained training-selected features.
Validation at threshold 0.5 produced PR-AUC 0.108350, recall 0.333333,
precision 0.134615, F1 0.191781, and ROC-AUC 0.663568. Validation failure
prevalence is 0.067093. Confusion counts are TN=247, FP=45, FN=14, TP=7.
These are baseline results, not a model selection or a production-quality claim.
Random splitting does not demonstrate future manufacturing performance, and
balanced class weights do not establish calibrated probabilities. Test scores
remain unevaluated for the later model comparison stage.

SECOM is attributed to the [UCI dataset](https://archive.ics.uci.edu/dataset/179/secom)
under CC BY 4.0; reports/raw_data_validation.json records source checksums and
the measured width of 590 despite the raw documentation's 591-attribute claim.
