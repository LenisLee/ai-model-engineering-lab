"""FastAPI inference service for text classification."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ai_model_engineering_lab.inference.engine import InferenceEngine


engine: InferenceEngine | None = None


class PredictRequest(BaseModel):
    texts: List[str]


class PredictionResult(BaseModel):
    text: str
    label: int
    probabilities: List[float]


class PredictResponse(BaseModel):
    predictions: List[PredictionResult]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    model_path = os.getenv("MODEL_PATH", "outputs/checkpoints")
    engine = InferenceEngine.from_pretrained(model_path)
    yield
    engine = None


app = FastAPI(title="AI Model Engineering Lab", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": engine is not None}


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided")

    results = engine.predict(request.texts)
    predictions = [
        PredictionResult(**r) for r in results
    ]
    return PredictResponse(predictions=predictions)
