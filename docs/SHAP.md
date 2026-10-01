# Offline SHAP model associations

**SHAP explains model associations. SHAP does NOT establish causality.**
Anonymized SECOM feature IDs prevent physical process interpretation. Features
such as feature_103 and feature_021 cannot be assigned invented sensor names,
process settings, causal effects, or physical root causes. Process-engineer
review would be required before operational action.

This stage explains the frozen XGBoost candidate 1 in selected-model-v1.
The model, train-fitted preprocessing, retained feature order, selected threshold
0.08485201001167297, and persisted membership are unchanged. It does not fit
anything, regenerate splits, select models or thresholds, or evaluate test data.

## Reproduce the artifacts

Run from the repository root in PowerShell with the already verified normalized
cache, selected bundle, and persisted split:

```powershell
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
& .\.venv\Scripts\python.exe -m quality_analytics.explain
& .\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
```

The implementation uses SHAP 0.46.0, Matplotlib 3.9.4, XGBoost 2.1.4, and NumPy
1.26.4 on CPU. Only SHAP and Matplotlib were added to the dependency manifest for
this stage. Their installed dependencies are recorded in the importance report.
Matplotlib uses the headless Agg backend and an ignored .cache/matplotlib/
directory for its font/configuration cache unless MPLCONFIGDIR is already set.

The command reads data/normalized/secom.jsonl,
artifacts/<dataset_version>/selected-model-v1/selected.pkl, and the original
logistic-regression-v1/split_membership.json. --local-data, --bundle,
--split-membership, and --report-dir override those paths. Missing artifacts
fail instead of being recreated. Only load trusted local pickle bundles.
SHAP tests require these existing local artifacts and normalized data.

## Sampling and explanation method

Stable sample ID order and NumPy default_rng(seed=42) select 100 validation rows
without replacement. The validation cap is 100. A separate seed-42 draw selects
64 TRAIN rows as the background, capped at 64. Smaller --sample-size and
--background-size values are supported; larger values are rejected. Selected
sample IDs are recorded so the exact draws can be audited. Labels and scores
are not used to choose either sample. The local example is the highest failure
score within the fixed explained validation subset and must meet the existing
selected threshold.

The saved missingness filter, training median imputer, and constant-feature
filter transform only those background and validation rows. There is no scaling,
refitting, feature removal, or change of order in this stage. The 450 retained
features keep their original feature_000-style IDs.

[TreeExplainer](https://shap.readthedocs.io/en/stable/generated/shap.TreeExplainer.html)
uses model_output="raw", feature_perturbation="interventional", and the sampled
training background. The algorithm's interventional option specifies a
feature-dependence convention; it does not establish causal manufacturing
effects. The background choice and correlated measurements can change how
associations are distributed among features.

For this binary XGBoost model, raw output is the failure log-odds margin. For
each explained validation row:

```text
expected raw output + sum(SHAP values) = model raw margin
sigmoid(model raw margin) = model failure score
```

Additivity and score reconstruction are checked numerically. A positive signed
SHAP value contributes toward a higher model failure log-odds score relative to
the background; a negative value contributes toward a lower model score. These
are contributions within the fitted model, not effects of physically changing a
sensor. Global mean absolute SHAP summarizes magnitude and removes direction.
It is an association ranking for these 100 validation samples, not a population
estimate or a manufacturing yield measurement.

## Saved artifacts

- reports/shap_feature_importance.json: mean absolute SHAP for all 450 retained
  features, sorted ranking, top 20, exact sample/background IDs, dimensions,
  units, dependencies, additivity error, and frozen bundle/split checksums.
- reports/shap_summary.png: top-20 summary plot for the fixed validation sample,
  with original IDs, signed log-odds contributions, and feature-value colors.
  Colors use retained/imputed measurements; imputed values are not new
  observations. Plot jitter is seeded for reproducibility.
- reports/shap_local_example.json: validation sample
  secom-01b3bed261fc1223:1273, failure score 0.7022242546081543, frozen threshold,
  expected value, raw margin, all 450 ordered signed contributions, and top-20
  associations. Original missing values appear as null; transformed values and
  training-median imputation flags are recorded separately. No actual label or
  classification-performance metric is used for this illustrative selection.

## Verified associations and safeguards

| Rank | Feature ID | Mean absolute SHAP, log-odds |
| --- | --- | ---: |
| 1 | feature_103 | 0.228930 |
| 2 | feature_021 | 0.193981 |
| 3 | feature_033 | 0.184623 |
| 4 | feature_317 | 0.164193 |
| 5 | feature_129 | 0.157962 |
| 6 | feature_200 | 0.139761 |
| 7 | feature_488 | 0.128111 |
| 8 | feature_344 | 0.114566 |
| 9 | feature_577 | 0.102970 |
| 10 | feature_031 | 0.096215 |

The verified output has SHAP dimensions [100, 450] and a 64-row training
background. The summary plot was inspected for legible original IDs and units.
Frozen artifact files retained their SHA-256 checksums. In-memory booster bytes
and fitted preprocessing state are also checked before/after explanations.

All 48 unittest tests passed, including five SHAP-stage tests covering frozen
state without fitting or split regeneration, deterministic capped sampling,
original feature order, rejection of incorrect SHAP dimensions, additivity and
PNG output, and isolation from poisoned synthetic test measurements. Dependency
consistency and git diff whitespace checks passed.

**TEST remains untouched:** test rows are not transformed, predicted, explained,
or evaluated. Whole-snapshot identity and saved membership are checked only to
confirm the input belongs to the frozen dataset. No test metrics are reported,
and no service, warehouse predictions, or deployment stage is implemented here.
