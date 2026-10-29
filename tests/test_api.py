"""API tests with FastAPI's TestClient against a tiny real model (conftest.py). No CSV or trained model needed.

They cover the API contract: health, info, single and batch predictions, bad input -> 422 (never 500),
no input echoed back, no request body in logs, 503 without a model, and that the API scores exactly like
the saved pipeline (no train/serve skew).
"""
import json
import logging

import joblib
import pandas as pd
import pytest

from src import config
from tests.conftest import BAD_TEST_FLOW, BENIGN_TEST_FLOW, TEST_THRESHOLD


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_model_info_describes_the_served_model(client):
    info = client.get("/model/info").json()
    assert info["version"] == "test-tiny"
    assert info["features"] == config.FEATURE_COLS
    assert info["threshold"] == TEST_THRESHOLD


def test_predict_benign_and_bad_flows(client):
    benign = client.post("/predict", json=BENIGN_TEST_FLOW).json()
    bad = client.post("/predict", json=BAD_TEST_FLOW).json()

    assert benign["is_outlier"] is False and bad["is_outlier"] is True
    assert benign["score"] < TEST_THRESHOLD <= bad["score"]
    assert benign["threshold"] == TEST_THRESHOLD and benign["model_version"] == "test-tiny"


def test_batch_returns_one_result_per_flow_and_stops_at_1000(client):
    body = client.post("/predict/batch", json={"flows": [BENIGN_TEST_FLOW, BAD_TEST_FLOW]}).json()
    assert [p["is_outlier"] for p in body["predictions"]] == [False, True]
    assert body["flagged"] == 1
    assert client.post("/predict/batch", json={"flows": [BENIGN_TEST_FLOW] * 1001}).status_code == 422


def test_missing_field_is_422(client):
    flow = {key: value for key, value in BENIGN_TEST_FLOW.items() if key != "dest_port"}
    response = client.post("/predict", json=flow)
    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "dest_port"]


@pytest.mark.parametrize("overrides", [{"avg_ipt": float("nan")}, {"duration": float("inf")}, {"bytes_in": 10**30}])
def test_nan_infinity_and_huge_numbers_are_422_not_500(client, overrides):
    """These gave 500s before the input limits. `json=` cannot send NaN, so the body is sent as raw text."""
    body = json.dumps({**BENIGN_TEST_FLOW, **overrides})  # Python writes NaN / Infinity tokens
    response = client.post("/predict", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 422


def test_422_does_not_echo_the_input_back(client):
    response = client.post("/predict", json={**BENIGN_TEST_FLOW, "bytes_out": -987654})
    assert response.status_code == 422
    assert "987654" not in response.text


def test_no_model_gives_503(client_without_model):
    assert client_without_model.get("/health").status_code == 503
    assert client_without_model.post("/predict", json=BENIGN_TEST_FLOW).status_code == 503


def test_api_scores_exactly_like_the_saved_pipeline(client, bundle_path):
    """Train/serve parity: the API must not transform inputs differently from training."""
    pipeline = joblib.load(bundle_path)["pipeline"]
    for flow in [BENIGN_TEST_FLOW, BAD_TEST_FLOW]:
        expected = pipeline.predict_proba(pd.DataFrame([flow])[config.FEATURE_COLS])[:, 1][0]
        assert client.post("/predict", json=flow).json()["score"] == pytest.approx(expected, rel=1e-12)


def test_logs_have_request_id_and_latency_but_never_the_body(client, caplog):
    caplog.set_level(logging.INFO, logger="flow_api")
    client.post("/predict", json={**BENIGN_TEST_FLOW, "bytes_out": 424242})

    assert "request_id=" in caplog.text and "latency_ms=" in caplog.text
    assert "424242" not in caplog.text
