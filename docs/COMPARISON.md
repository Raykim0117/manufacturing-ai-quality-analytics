# Validation comparison and frozen selection

This stage continues the verified Logistic Regression baseline. It reads the
existing split_membership.json and baseline.pkl, checks dataset identity, and
does not regenerate assignments or refit preprocessing. The original baseline
files and persisted split were unchanged after the verified run.

Run from the repository root in PowerShell using the existing local cache:

```powershell
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
& .\.venv\Scripts\python.exe -m quality_analytics.compare --backend local
& .\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

The default backend is BigQuery, optionally overridden by DATA_BACKEND.
--backend bigquery uses the existing read-only warehouse query and configured
GCP_PROJECT_ID, BQ_DATASET, BQ_LOCATION, and BQ_RAW_TABLE. The snapshot must match
the baseline fingerprint. --local-data selects a normalized JSONL file;
--baseline-dir selects the existing baseline artifact directory. A missing or
incompatible baseline or split is an error; neither is generated automatically.

Two fixed CPU XGBoost candidates are compared:

| Candidate | Maximum depth | Trees |
| --- | ---: | ---: |
| xgboost_candidate_1 | 2 | 120 |
| xgboost_candidate_2 | 3 | 180 |

Both use seed 42, one CPU thread, hist trees, learning_rate=0.05, subsample=0.8,
colsample_bytree=0.8, min_child_weight=3, and reg_lambda=2. XGBoost 2.1.4 is
pinned. scale_pos_weight uses only TRAIN labels: 878 PASS / 62 FAIL =
14.161290322580646, following the documented
[XGBoost parameter](https://xgboost.readthedocs.io/en/release_2.1.0/parameter.html).
No early stopping or broader search is performed. Both models train on the
existing 940 training rows only.

XGBoost receives a copy of the baseline pipeline's fitted missingness filter,
median imputer, and constant filter, with the scaler omitted. It uses the same
450 retained measurements in original feature order. Logistic Regression uses
the saved scaled pipeline and model; it is not retrained. Only train and
validation features are transformed. The complete snapshot fingerprint and
membership counts are checked for identity; test scores are never calculated.

Model selection uses only validation PR-AUC from precision_recall_curve and
trapezoidal auc(recall, precision). Exact ties favor Logistic Regression; exact
ties between the two challengers retain the first candidate. Scores for the
selected model alone are used to choose a validation threshold maximizing
precision subject to recall >=0.80. Ties favor higher recall, then higher
threshold. The terminal PR point has no threshold and is excluded. Predictions
use score >= threshold. Secondary metrics use failure as positive, and precision,
recall, and F1 use zero_division=0.

## Verified local validation results

These results use the existing locally normalized, hash-validated SECOM snapshot
secom-01b3bed261fc1223. There are 313 validation samples with 21 failures, giving
failure prevalence 0.0670926517571885. No live BigQuery read was needed for this
verification.

| Model at threshold 0.5 | PR-AUC | Recall | Precision | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.108350 | 0.333333 | 0.134615 | 0.191781 | 0.663568 |
| XGBoost candidate 1 | 0.173602 | 0.285714 | 0.250000 | 0.266667 | 0.667482 |
| XGBoost candidate 2 | 0.160332 | 0.047619 | 0.200000 | 0.076923 | 0.644651 |

Confusion matrices at threshold 0.5, ordered [PASS, FAIL] with actual rows and
predicted columns, are [[247, 45], [14, 7]], [[274, 18], [15, 6]], and
[[288, 4], [20, 1]], respectively.

Selected model: xgboost_candidate_1. Selected validation threshold:
0.08485201001167297. At this threshold, recall=0.9047619047619048,
precision=0.07251908396946564, F1=0.13427561837455831, and
ROC-AUC=0.6674820613176777. Its confusion matrix is [[49, 243], [2, 19]].
The high-recall operating point produces many false positives. This is a demo
constraint selected on validation, not a production-approved operating point or
a guarantee of test recall. Scores are not measured manufacturing yield.

TEST remains untouched by preprocessing fitting, model fitting, prediction,
evaluation, model selection, and threshold selection. There is no train-plus-
validation refit after selection. The frozen training state and threshold will
be used unchanged for a later, separately authorized test evaluation.

## Saved files and verification

The ignored artifacts/secom-01b3bed261fc1223/selected-model-v1/ directory contains:

- selected.pkl: selected fitted model and preprocessing, threshold, ordered
  features, dependency/dataset metadata, metrics, and comparison report.
- metadata.json and retained_features.json: readable frozen configuration,
  feature order, training class counts, and baseline/split SHA-256 checksums.
- comparison_validation.json: all three validation metric sets at threshold
  0.5, selected threshold metrics, class prevalence, parameters, PR curve
  coordinates, and the explicit test-set statement.
- xgboost_candidate_1.pkl and xgboost_candidate_2.pkl: fitted candidate snapshots
  with their shared unscaled preprocessing, parameters, and validation metrics.

reports/comparison_validation.json is the reviewed comparison report.
--report changes that path. --output-dir selects a new artifact directory. An
existing selected directory is never overwritten; subsequent runs must use a
different directory if explicitly needed. Only load trusted local pickle files.

All 43 unittest tests passed, including 9 challenger-stage tests covering CPU
determinism, training-only class weights, persisted split reuse, validation-only
model and threshold selection, threshold tie rules, absence of preprocessing
refitting, isolation from poisoned synthetic test measurements, and restored
bundle prediction/threshold parity. Dependency consistency and git diff
whitespace checks also passed. The real baseline files and split membership
retained their original SHA-256 checksums.
