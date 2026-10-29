"""Shared test fixtures: a tiny but real model bundle, so the API tests need neither the CSV nor the trained model.

The bundle uses the same pipeline as training (build_preprocessor + HistGradientBoosting), fitted on 200
hand-made flows in about a second. The data and model files are gitignored, so this keeps the tests
running on a fresh clone and in CI.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from src import config
from src.api import app
from src.features import build_preprocessor

TEST_THRESHOLD = 0.5
# Fixed flows that look like the tiny training data below (the real examples in docs/examples do not).
BENIGN_TEST_FLOW = {"avg_ipt": 0.0, "bytes_in": 50, "bytes_out": 6_000, "entropy": 3.0, "num_pkts_out": 5,
                    "num_pkts_in": 2, "total_entropy": 18_000.0, "duration": 0.0005, "proto": 6, "dest_port": 9200}
BAD_TEST_FLOW = {"avg_ipt": 120.0, "bytes_in": 60, "bytes_out": 150, "entropy": 4.0, "num_pkts_out": 4,
                 "num_pkts_in": 3, "total_entropy": 600.0, "duration": 2.0, "proto": 6, "dest_port": 445}


def tiny_flows() -> tuple[pd.DataFrame, pd.Series]:
    """120 benign flows (TCP to Elasticsearch 9200, large outbound) and 80 bad ones (SMB 445 and SSDP 1900 probes)."""
    rng = np.random.default_rng(0)

    def flows(n: int, proto: int, port: int, bytes_out: tuple[int, int], duration: tuple[float, float],
              avg_ipt: tuple[float, float]) -> pd.DataFrame:
        out, entropy = rng.integers(*bytes_out, size=n), rng.uniform(2, 5, size=n)
        return pd.DataFrame({
            "avg_ipt": rng.uniform(*avg_ipt, size=n), "bytes_in": rng.integers(0, 200, size=n),
            "bytes_out": out, "entropy": entropy, "num_pkts_out": rng.integers(1, 10, size=n),
            "num_pkts_in": rng.integers(0, 6, size=n), "total_entropy": entropy * out,
            "duration": rng.uniform(*duration, size=n), "proto": proto, "dest_port": port,
        })

    X = pd.concat([flows(120, 6, 9200, bytes_out=(1_000, 12_000), duration=(0.0001, 0.001), avg_ipt=(0, 0)),
                   flows(40, 6, 445, bytes_out=(50, 300), duration=(0.1, 5.0), avg_ipt=(10, 300)),
                   flows(40, 17, 1900, bytes_out=(50, 300), duration=(0.1, 5.0), avg_ipt=(10, 300))],
                  ignore_index=True)
    return X[config.FEATURE_COLS], pd.Series([0] * 120 + [1] * 80)


@pytest.fixture(scope="session")
def bundle_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A bundle with every key the API reads, shaped like the one src/train.py saves."""
    pipeline = Pipeline([("preprocess", build_preprocessor()),
                         ("model", HistGradientBoostingClassifier(l2_regularization=1.0, random_state=0))])
    pipeline.fit(*tiny_flows())
    bundle = {"pipeline": pipeline, "threshold": TEST_THRESHOLD, "version": "test-tiny",
              "trained_at": "2026-01-01T00:00:00+00:00", "features": config.FEATURE_COLS,
              "metrics": {"chosen_model": "hist_gradient_boosting", "threshold": {"rule": "fixed 0.5 for tests"},
                          "test": {"recall": 1.0, "confusion_matrix": {}}}}
    path = tmp_path_factory.mktemp("model") / "model.joblib"
    joblib.dump(bundle, path)
    return path


@pytest.fixture
def client(bundle_path: Path, monkeypatch: pytest.MonkeyPatch):
    """API client serving the tiny bundle. `with` runs the app's startup, which loads the model."""
    monkeypatch.setenv("MODEL_PATH", str(bundle_path))
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def client_without_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """API client whose model file does not exist."""
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "missing.joblib"))
    with TestClient(app) as test_client:
        yield test_client
