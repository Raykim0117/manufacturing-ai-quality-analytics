-- Offline labels are nullable for future unlabeled predictions.
-- Python batch persistence creates this explicit schema and appends only TEST rows.
CREATE TABLE IF NOT EXISTS `${GCP_PROJECT_ID}.${BQ_DATASET}.${BQ_PREDICTIONS_TABLE}` (
  prediction_id STRING NOT NULL,
  sample_id STRING NOT NULL,
  run_id STRING NOT NULL,
  model_version STRING NOT NULL,
  dataset_version STRING NOT NULL,
  source STRING NOT NULL,
  split STRING,
  scored_at TIMESTAMP NOT NULL,
  failure_score FLOAT64 NOT NULL,
  predicted_failure BOOL NOT NULL,
  threshold FLOAT64 NOT NULL,
  actual_failure INT64
);
