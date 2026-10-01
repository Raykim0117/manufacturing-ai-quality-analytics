-- Publish the latest explicitly chosen batch run; never mix runs or API traffic.
-- Observed failure rate uses actual labels; model scores are not manufacturing yield.
CREATE OR REPLACE VIEW `${GCP_PROJECT_ID}.${BQ_DATASET}.quality_summary` AS
SELECT
  run_id,
  COUNT(*) AS sample_count,
  COUNTIF(actual_failure = 1) AS observed_fail_count,
  SAFE_DIVIDE(COUNTIF(actual_failure = 1), COUNT(*)) AS observed_fail_rate,
  COUNTIF(predicted_failure) AS flagged_count,
  AVG(failure_score) AS mean_failure_score,
  COALESCE(SAFE_DIVIDE(COUNTIF(predicted_failure AND actual_failure = 1),
                       COUNTIF(actual_failure = 1)), 0.0) AS recall,
  COALESCE(SAFE_DIVIDE(COUNTIF(predicted_failure AND actual_failure = 1),
                       COUNTIF(predicted_failure)), 0.0) AS precision
FROM `${GCP_PROJECT_ID}.${BQ_DATASET}.${BQ_PREDICTIONS_TABLE}`
WHERE run_id = '${BATCH_RUN_ID}'
  AND source = 'batch'
  AND split = 'test'
  AND actual_failure IS NOT NULL
GROUP BY run_id;
