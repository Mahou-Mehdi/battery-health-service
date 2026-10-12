"""Local inference API. Run demo training before starting the server."""

from contextlib import asynccontextmanager
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
import joblib
from pydantic import BaseModel, ConfigDict, Field

from .data import Reading
from .model import predict_history


class ReadingInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    battery_id: str = Field(min_length=1, max_length=100)
    cycle: int = Field(ge=0, strict=True)
    capacity_ah: float = Field(gt=0)
    nominal_capacity_ah: float = Field(gt=0)
    temperature_c: float
    resistance_mohm: float = Field(gt=0)
    charge_rate_c: float = Field(gt=0)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    readings: list[ReadingInput] = Field(min_length=2, max_length=1000)


def create_app(bundle=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.bundle = bundle
        path = Path(os.environ.get("BATTERY_MODEL", "artifacts/model.joblib"))
        if bundle is None and path.is_file():
            # Artifact path is server configuration, never a user-provided URL/path.
            app.state.bundle = joblib.load(path)
        yield

    app = FastAPI(title="Battery Health Forecast API", version="1.0.0", lifespan=lifespan,
                  description="Forecast SOH from recent cycle history. Demo models use synthetic data.")

    @app.get("/health")
    def health():
        return {"status": "ok", "model_ready": app.state.bundle is not None}

    def require_model():
        if app.state.bundle is None:
            raise HTTPException(503, "Model unavailable. Run python -m battery_health.cli demo first.")
        return app.state.bundle

    @app.get("/model")
    def metadata():
        return require_model()["metadata"]

    @app.post("/predict")
    def predict(request: PredictionRequest):
        model_bundle = require_model()
        try:
            return predict_history(model_bundle, [Reading(**row.model_dump()) for row in request.readings])
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    return app


app = create_app()
