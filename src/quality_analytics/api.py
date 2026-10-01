"""Minimal frozen-model inference API with optional synchronous persistence."""

from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .config import BigQueryConfig, InferenceConfig
from .predict import FrozenPredictor, RAW_FEATURE_COUNT

LOGGER = logging.getLogger(__name__)
FiniteMeasurement = Annotated[float, Field(strict=True, allow_inf_nan=False)]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    sample_id: Annotated[str, Field(min_length=1)]
    features: Annotated[list[FiniteMeasurement | None],
                        Field(min_length=RAW_FEATURE_COUNT, max_length=RAW_FEATURE_COUNT)]

    @field_validator("sample_id")
    @classmethod
    def nonempty_sample_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("sample_id must not be empty or whitespace")
        return value


class PredictionResponse(BaseModel):
    prediction_id: str
    sample_id: str
    failure_score: Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
    predicted_failure: bool
    threshold: float
    model_version: str
    stored: bool


def create_app(
    settings: InferenceConfig | None = None,
    prediction_writer: Callable[[dict], None] | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        configuration = settings if settings is not None else InferenceConfig.from_env()
        app.state.predictor = FrozenPredictor.load(configuration.model_path)
        app.state.run_id = f"api-{uuid4().hex}"
        app.state.writer = None
        store = None
        if configuration.persist_predictions:
            if prediction_writer is not None:
                app.state.writer = prediction_writer
            else:
                from .warehouse import OnlinePredictionStore

                store = OnlinePredictionStore(BigQueryConfig.from_env())
                app.state.writer = store.write
        try:
            yield
        finally:
            if store is not None:
                store.close()

    app = FastAPI(title="SECOM Quality Inference", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
        # Exclude input/context: NaN/Infinity cannot be serialized in JSON responses.
        details = [{key: item[key] for key in ("loc", "msg", "type")}
                   for item in error.errors()]
        return JSONResponse(status_code=422, content={"detail": details})

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_version": app.state.predictor.model_version}

    @app.get("/model-info")
    def model_info() -> dict:
        predictor = app.state.predictor
        return {
            "model_version": predictor.model_version,
            "raw_feature_count": len(predictor.raw_feature_names),
            "raw_feature_names": list(predictor.raw_feature_names),
            "positive_class": {"label": 1, "meaning": "manufacturing failure (FAIL)"},
            "threshold": predictor.threshold,
        }

    @app.post("/predict", response_model=PredictionResponse)
    def predict(payload: PredictionRequest) -> PredictionResponse:
        predictor = app.state.predictor
        failure_score = predictor.score(payload.features)
        prediction_id = str(uuid4())
        predicted_failure = failure_score >= predictor.threshold
        stored = False
        if app.state.writer is not None:
            row = {
                "prediction_id": prediction_id, "sample_id": payload.sample_id,
                "run_id": app.state.run_id, "model_version": predictor.model_version,
                "dataset_version": predictor.dataset_version, "source": "api", "split": None,
                "scored_at": datetime.now(timezone.utc).isoformat(),
                "failure_score": failure_score, "predicted_failure": predicted_failure,
                "threshold": predictor.threshold, "actual_failure": None,
            }
            try:
                app.state.writer(row)
            except Exception:
                LOGGER.error("Online prediction persistence failed")
                raise HTTPException(
                    status_code=503, detail="Prediction persistence unavailable",
                ) from None
            stored = True
        return PredictionResponse(
            prediction_id=prediction_id, sample_id=payload.sample_id,
            failure_score=failure_score, predicted_failure=predicted_failure,
            threshold=predictor.threshold, model_version=predictor.model_version, stored=stored,
        )

    return app
