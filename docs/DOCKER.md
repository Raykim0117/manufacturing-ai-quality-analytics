# Local inference container

This stage packages the already verified FastAPI service and frozen selected
model. It does not train, fit, tune, evaluate a dataset, or write to the cloud.
Local Docker execution requires a running Docker engine using Linux containers.

## Image contents

The Dockerfile uses `python:3.10-slim-bookworm`, installs the existing
`requirements.txt` unchanged, and installs `libgomp1` for CPU XGBoost's OpenMP
runtime. The application runs as UID/GID `10001`, with Uvicorn as the main
process so it receives termination signals.

Only these repository files are copied into the image:

- `requirements.txt`
- `src/quality_analytics/api.py`
- `src/quality_analytics/config.py`
- `src/quality_analytics/predict.py`
- `src/quality_analytics/preprocessing.py`
- `src/quality_analytics/warehouse.py`
- `artifacts/secom-01b3bed261fc1223/selected-model-v1/selected.pkl`

The frozen pickle contains the model, fitted preprocessing, original feature
ordering, threshold, and metadata. Its SHA-256 before containerization is
`66e00f2be70812c15c3d94dba81ce62116f8df3089d548dd4f5fa01a64559d8c`.
It is copied to `/app/artifacts/selected-model-v1/selected.pkl` without changing
its contents. The bundle remains ignored by Git and must exist locally to build.

The `.dockerignore` starts with a deny-all rule and permits only the files above
plus the build instructions. It excludes all other artifacts, raw/normalized
data, training/evaluation/SHAP modules, SQL, notebooks, tests, reports, docs,
Git metadata, the virtual environment, caches, local env files, and credentials.
The full existing dependency manifest is installed, including its offline/test
dependencies; no alternate manifest or model version is introduced.

## Build and run

Run from the repository root in PowerShell after starting Docker Desktop in
Linux-container mode:

```powershell
docker version
docker build --tag quality-api:local .
docker run --detach --name quality-api-local --publish 127.0.0.1:8080:8080 --env PERSIST_PREDICTIONS=false quality-api:local
```

The image defaults to `PERSIST_PREDICTIONS=false`, `PORT=8080`, and an explicit
`MODEL_PATH`. No credentials, env-file loading, host mounts, or cloud settings
are required. The container starts:

```sh
uvicorn quality_analytics.api:create_app --factory --host 0.0.0.0 --port ${PORT:-8080}
```

The shell expands the port and uses `exec` to start Uvicorn. If overriding `PORT`,
publish that same container port. The image's health check uses the configured
port, allows 30 seconds for startup, and requests `/health` every 30 seconds.

## HTTP smoke checks

Once startup completes, run:

```powershell
$health = Invoke-RestMethod -Uri "http://127.0.0.1:8080/health"
$health | ConvertTo-Json
$info = Invoke-RestMethod -Uri "http://127.0.0.1:8080/model-info"
$info | ConvertTo-Json -Depth 3
$body = @{ sample_id = "docker-demo-001"; features = [object[]]::new(590) } | ConvertTo-Json -Depth 3 -Compress
$prediction = Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8080/predict" -ContentType "application/json" -Body $body
$prediction | ConvertTo-Json
docker inspect --format '{{.State.Health.Status}}' quality-api-local
docker exec quality-api-local id -u
docker image inspect quality-api:local --format '{{.Size}}'
```

`/health` should return `{"status":"ok","model_version":"selected-model-v1"}`.
Model info must show 590 raw features ordered `feature_000` through
`feature_589`, failure as positive class, and frozen threshold
`0.08485201001167297`. The UID should be `10001`; image size is reported in bytes.
The API's health endpoint performs no BigQuery query.

The request above is a synthetic all-null sample, not a SECOM dataset row.
The previously verified host FastAPI response for the same measurements had
`failure_score=0.46548160910606384`, `predicted_failure=true`, the frozen
threshold, and `stored=false`. Prediction IDs differ per call. A model score is
not measured manufacturing yield.

For a direct container-versus-host API comparison, leave the container running
and execute this from the repository root:

```powershell
$env:PYTHONPATH = "$PWD\src"
$env:PERSIST_PREDICTIONS = "false"
@'
import json
from pathlib import Path
from urllib.request import Request, urlopen
from fastapi.testclient import TestClient
from quality_analytics.api import create_app
from quality_analytics.config import InferenceConfig

payload = {"sample_id": "docker-demo-001", "features": [None] * 590}
bundle = Path("artifacts/secom-01b3bed261fc1223/selected-model-v1/selected.pkl")
with TestClient(create_app(InferenceConfig(bundle))) as local:
    local_result = local.post("/predict", json=payload)
    local_result.raise_for_status()
    expected = local_result.json()
    expected_info = local.get("/model-info").json()
request = Request("http://127.0.0.1:8080/predict",
                  data=json.dumps(payload).encode(),
                  headers={"Content-Type": "application/json"})
with urlopen(request, timeout=10) as response:
    actual = json.load(response)
with urlopen("http://127.0.0.1:8080/model-info", timeout=10) as response:
    assert json.load(response) == expected_info
for field in ("sample_id", "failure_score", "predicted_failure", "threshold",
              "model_version", "stored"):
    assert actual[field] == expected[field], (field, actual[field], expected[field])
print(json.dumps(actual, indent=2))
print("Container and local API predictions match")
'@ | & .\.venv\Scripts\python.exe -
```

Only synthetic request measurements are scored. This comparison never loads
split membership or evaluates train, validation, or TEST data.

## Tests and cleanup

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py" -v
docker logs quality-api-local
docker stop quality-api-local
docker rm quality-api-local
```

Stopping and removing this named demo container leaves the local image intact.
Do not remove unrelated containers or prune Docker state.

## Current verification status

Verification resumed on 2026-10-02. The existing Python suite passed again:
70 tests, zero failures or errors. The frozen bundle's SHA-256 still matches the
value above. The Dockerfile and `.dockerignore` were reused without modification.

Docker Desktop and its CLI are now installed. However, the Linux engine fails
to start. Docker Desktop's backend log reports:

```text
engine linux/wsl failed to start: checking preconditions: Virtual Machine Platform not enabled
No virtualization available
```

The CLI reaches the `desktop-linux` endpoint but returns HTTP 500 because the
engine is unavailable. Enable Windows Virtual Machine Platform manually and
complete any required Windows restart, then confirm that `docker version`
prints both Client and Server information before rerunning the commands above.
No Windows features, WSL installation, or host system settings were changed by
this verification task.

The image build, container run, container HTTP parity, and image size remain
unverified. The example responses above are expectations from the verified host
API, not claims of a completed container smoke test. No cloud writes occurred.

References: [Dockerfile instructions](https://docs.docker.com/reference/dockerfile/),
[build-context exclusions](https://docs.docker.com/build/concepts/context/#dockerignore-files),
and [Docker Desktop Windows installation](https://docs.docker.com/desktop/setup/install/windows-install/).
