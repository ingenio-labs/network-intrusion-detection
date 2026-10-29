"""FastAPI service that scores network flows with the model bundle saved by src/train.py.

Run from the repo root:  uvicorn src.api:app --port 8000   (docs at http://127.0.0.1:8000/docs)

- The bundle (pipeline + threshold + version) is loaded once at startup. Without it, model endpoints answer 503.
- Pydantic checks every field before the model sees it, so bad input gets a 422, never a 500.
- Every answer returns the score *and* the threshold, so callers can rank alerts, not just filter them.
- One log line per request (request id, status, latency), never the request body.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import joblib
import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from src import config

MAX_BATCH_SIZE = 1_000
# Real byte and packet counts stop at 65,535; 2**32 refuses nothing real but keeps absurd numbers such as
# 10**30 out of the model (they crashed it with a 500 before this limit).
MAX_COUNT = 2**32

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("flow_api")
logger.setLevel(logging.INFO)  # keep INFO logs even if something else configured logging first

# ---------------------------------------------------------------- examples shown in /docs (from the test set)
NORMAL_FLOW = json.loads((config.EXAMPLES_DIR / "normal_flow.json").read_text())
OUTLIER_FLOW = json.loads((config.EXAMPLES_DIR / "outlier_flow.json").read_text())
BAD_FLOW = {key: value for key, value in NORMAL_FLOW.items() if key != "dest_port"} | {"bytes_out": -5}
FLOW_EXAMPLES = {
    "normal": {"summary": "Normal flow (benign, from the test set)", "value": NORMAL_FLOW},
    "outlier": {"summary": "Outlier flow (from the test set)", "value": OUTLIER_FLOW},
    "bad_request": {"summary": "Bad request: negative bytes_out and no dest_port (returns 422)", "value": BAD_FLOW},
}
BATCH_EXAMPLES = {
    "two_flows": {"summary": "One normal and one outlier flow", "value": {"flows": [NORMAL_FLOW, OUTLIER_FLOW]}},
}


# ---------------------------------------------------------------- request and response shapes
class Flow(BaseModel):
    """One network flow: the 10 raw fields the model was trained on. Extra fields (e.g. IPs) are ignored."""

    model_config = ConfigDict(allow_inf_nan=False)  # NaN and infinity are refused with a 422

    avg_ipt: float = Field(ge=0, description="Average time between packets, in milliseconds")
    bytes_in: int = Field(ge=0, le=MAX_COUNT, description="Bytes sent from destination to source")
    bytes_out: int = Field(ge=0, le=MAX_COUNT, description="Bytes sent from source to destination")
    entropy: float = Field(ge=0, description="Payload entropy as reported by the flow sensor")
    num_pkts_out: int = Field(ge=0, le=MAX_COUNT, description="Packets sent from source to destination")
    num_pkts_in: int = Field(ge=0, le=MAX_COUNT, description="Packets sent from destination to source")
    total_entropy: float = Field(ge=0, description="Payload entropy summed over the whole flow")
    duration: float = Field(ge=0, description="Flow duration in seconds")
    proto: int = Field(ge=0, le=255, description="IP protocol number: 6 = TCP, 17 = UDP, 1 = ICMP")
    dest_port: int = Field(ge=0, le=65535, description="Destination port (0 when the protocol has no ports)")


class Prediction(BaseModel):
    is_outlier: bool = Field(description="True when score >= threshold: the flow looks malicious or outlier")
    score: float = Field(description="Model score from 0 to 1. Higher = more suspicious. Use it to rank alerts")
    threshold: float = Field(description="Alert threshold of this model version (see GET /model/info)")
    model_version: str


class BatchRequest(BaseModel):
    flows: list[Flow] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class BatchResponse(BaseModel):
    predictions: list[Prediction] = Field(description="One result per flow, in the same order as the request")
    flagged: int = Field(description="How many flows have is_outlier = true")


# ---------------------------------------------------------------- model
def load_bundle() -> dict[str, Any] | None:
    """The bundle from MODEL_PATH (set in Docker or tests), else models/model.joblib; None if there is no file."""
    path = Path(os.environ.get("MODEL_PATH", config.MODEL_PATH))
    if not path.exists():
        logger.error("no model at %s; run `python -m src.train` first", path)
        return None
    bundle = joblib.load(path)
    logger.info("model loaded: version=%s threshold=%.6f", bundle["version"], bundle["threshold"])
    return bundle


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once, when the server starts, not on every request."""
    app.state.bundle = load_bundle()
    yield


def get_bundle(request: Request) -> dict[str, Any]:
    """The loaded bundle, or a 503 instead of a crash when there is no model."""
    if request.app.state.bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run `python -m src.train`, then restart.")
    return request.app.state.bundle


def score_flows(bundle: dict[str, Any], flows: list[Flow]) -> list[Prediction]:
    """Score flows in one model call. The saved pipeline does all preprocessing, exactly as in training."""
    frame = pd.DataFrame([flow.model_dump() for flow in flows], columns=bundle["features"])
    frame[config.NUMERIC_COLS] = frame[config.NUMERIC_COLS].astype("float64")  # same types as in training
    scores = bundle["pipeline"].predict_proba(frame)[:, 1]
    return [Prediction(is_outlier=bool(score >= bundle["threshold"]), score=float(score),
                       threshold=bundle["threshold"], model_version=bundle["version"]) for score in scores]


# ---------------------------------------------------------------- app and endpoints
app = FastAPI(title="Network Flow Outlier Detection", version="1.0", lifespan=lifespan,
              description="Scores network flows. `is_outlier` = the flow looks malicious or outlier (not benign).")


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """422 that says which field is wrong and why, without echoing the input back.

    FastAPI's default repeats the input: that copies the caller's data into the response, and for NaN
    (not valid JSON) it turned the 422 into a 500.
    """
    errors = [{"loc": list(error["loc"]), "msg": error["msg"], "type": error["type"]} for error in exc.errors()]
    return JSONResponse({"detail": errors}, status_code=422)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """One log line per request with a request id (also sent back in X-Request-ID) and the latency."""
    request_id = uuid.uuid4().hex[:12]
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    logger.info("request_id=%s method=%s path=%s status=%d latency_ms=%.1f", request_id, request.method,
                request.url.path, response.status_code, (time.perf_counter() - started) * 1000)
    return response


@app.get("/health")
def health(request: Request) -> JSONResponse:
    """200 when the model is loaded; 503 otherwise, so a load balancer stops sending traffic."""
    loaded = request.app.state.bundle is not None
    return JSONResponse({"status": "ok" if loaded else "unavailable", "model_loaded": loaded},
                        status_code=200 if loaded else 503)


@app.get("/model/info")
def model_info(request: Request) -> dict[str, Any]:
    """What is being served: version, inputs, threshold and how it was chosen, and the test results."""
    bundle = get_bundle(request)
    metrics = bundle["metrics"]
    return {"version": bundle["version"], "trained_at": bundle["trained_at"], "model_type": metrics["chosen_model"],
            "features": bundle["features"], "threshold": bundle["threshold"],
            "threshold_rule": metrics["threshold"]["rule"],
            "test_metrics": {name: value for name, value in metrics["test"].items() if name != "confusion_matrix"}}


@app.post("/predict")
def predict(request: Request, flow: Annotated[Flow, Body(openapi_examples=FLOW_EXAMPLES)]) -> Prediction:
    """Score one flow."""
    return score_flows(get_bundle(request), [flow])[0]


@app.post("/predict/batch")
def predict_batch(request: Request,
                  batch: Annotated[BatchRequest, Body(openapi_examples=BATCH_EXAMPLES)]) -> BatchResponse:
    """Score up to 1,000 flows in one model call (much faster than 1,000 single requests)."""
    predictions = score_flows(get_bundle(request), batch.flows)
    return BatchResponse(predictions=predictions, flagged=sum(p.is_outlier for p in predictions))
