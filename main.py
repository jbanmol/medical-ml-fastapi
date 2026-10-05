"""Load the committed pipeline at startup and serve validated predictions."""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import joblib
import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from features import FEATURES

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
METADATA_PATH = BASE_DIR / "model_metadata.json"
DISCLAIMER = "This project is for educational demonstration only and must not be used for medical diagnosis or clinical decision-making."
logger = logging.getLogger(__name__)
Measurement = Annotated[float, Field(ge=0, allow_inf_nan=False, strict=True)]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [json.loads((BASE_DIR / "example_request.json").read_text())]},
    )
    mean_radius: Measurement
    mean_texture: Measurement
    mean_perimeter: Measurement
    mean_area: Measurement
    mean_smoothness: Measurement
    mean_compactness: Measurement
    mean_concavity: Measurement
    mean_concave_points: Measurement
    worst_radius: Measurement
    worst_texture: Measurement


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    app.state.class_mapping = {}
    try:
        model = joblib.load(MODEL_PATH)
        metadata = json.loads(METADATA_PATH.read_text())
        if metadata["sklearn_version"] != sklearn.__version__:
            raise ValueError("Model and runtime scikit-learn versions differ")
        if list(model.feature_names_in_) != list(FEATURES.values()):
            raise ValueError("Model feature order does not match the API")
        mapping = {int(key): value for key, value in metadata["class_mapping"].items()}
        if mapping != {0: "malignant", 1: "benign"} or list(model.classes_) != [0, 1]:
            raise ValueError("Unexpected model class mapping")
        app.state.class_mapping = mapping
        app.state.model = model
    except Exception:
        logger.exception("Could not load the trained model")
    yield
    app.state.model = None


app = FastAPI(
    title="Breast Cancer Classification API", description=DISCLAIMER,
    version="1.0.0", lifespan=lifespan,
)


@app.get("/")
def home():
    return {
        "name": app.title, "message": "Educational ML deployment demonstration",
        "health": "/health", "predict": "/predict", "docs": "/docs",
        "disclaimer": DISCLAIMER,
    }


@app.get("/health")
def health(request: Request):
    loaded = request.app.state.model is not None
    return JSONResponse(
        status_code=200 if loaded else 503,
        content={"status": "ok" if loaded else "error", "model_loaded": loaded},
    )


@app.post("/predict")
def predict(payload: PredictionRequest, request: Request):
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model unavailable")
    values = payload.model_dump()
    frame = pd.DataFrame(
        [[values[field] for field in FEATURES]], columns=list(FEATURES.values())
    )
    try:
        code = int(model.predict(frame)[0])
        probabilities = model.predict_proba(frame)[0]
        class_probabilities = {
            request.app.state.class_mapping[int(label)]: round(float(probability), 4)
            for label, probability in zip(model.classes_, probabilities)
        }
        label = request.app.state.class_mapping[code]
        return {
            "prediction": label, "prediction_code": code,
            "probability": class_probabilities[label],
            "class_probabilities": class_probabilities, "disclaimer": DISCLAIMER,
        }
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from None
