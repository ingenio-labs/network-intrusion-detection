"""Train the served model: five candidates, one alert threshold each, a speed check, one choice, one test.

Run from the repo root:  python -m src.train
Steps: split 60/20/20 -> fit on train -> thresholds, recall and speed on validation -> choose -> test once -> save.
notebooks/02_model_development.ipynb runs these same functions step by step and explains each decision.
"""
from __future__ import annotations

import io
import json
import time
from collections.abc import Callable
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from src import config
from src.features import build_preprocessor, load_raw, port_service, prepare_dataset

FEATURES, TARGET = config.FEATURE_COLS, config.TARGET_COL


class PortLookup(ClassifierMixin, BaseEstimator):
    """Rule baseline, no machine learning: score = share of bad flows seen in training for this (protocol, port).

    Ephemeral ports count as one port, as in the model. A combination never seen in training gets the overall rate.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> PortLookup:
        self.rates_ = pd.Series(np.asarray(y), index=self._keys(X)).groupby(level=0).mean()
        self.default_rate_ = float(np.mean(y))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        bad = self._keys(X).map(self.rates_).fillna(self.default_rate_).to_numpy(dtype="float64")
        return np.column_stack([1 - bad, bad])

    @staticmethod
    def _keys(X: pd.DataFrame) -> pd.Index:
        """One key per flow, e.g. "6/445" (TCP to SMB) or "17/ephemeral"."""
        protocols = X[config.PROTO_COL].astype("int64").astype(str)
        services = pd.Series(port_service(X[[config.PORT_COL]]).ravel(), index=X.index)
        return pd.Index(protocols + "/" + services)


def candidate_models() -> dict[str, BaseEstimator]:
    """Simplest first (a tie goes to the simpler model). The learned ones share one preprocessing pipeline."""
    def with_preprocessing(classifier: BaseEstimator) -> Pipeline:
        return Pipeline([("preprocess", build_preprocessor()), ("model", classifier)])

    seed = config.RANDOM_STATE
    return {
        "dummy": DummyClassifier(),  # the floor: every flow gets the training bad rate
        "port_lookup": PortLookup(),  # is a simple rule enough?
        # max_iter=1000: the default 100 iterations stop before the solver converges on this much data.
        "logistic_regression": with_preprocessing(LogisticRegression(max_iter=1000)),
        # max_depth=4: at most 16 leaves, so the whole model prints as rules on one screen.
        "decision_tree": with_preprocessing(DecisionTreeClassifier(max_depth=4, random_state=seed)),
        # l2_regularization=1.0: without it, benign TCP/9200 flows got runaway scores of exactly 1.0.
        "hist_gradient_boosting": with_preprocessing(
            HistGradientBoostingClassifier(l2_regularization=1.0, random_state=seed)),
    }


def split(clean: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified train / validation / test. The first cut is the same 80/20 as before, so test is unchanged."""
    rest, test = train_test_split(clean, test_size=config.TEST_SIZE, stratify=clean[TARGET],
                                  random_state=config.RANDOM_STATE)
    train, validation = train_test_split(rest, test_size=config.VALIDATION_SIZE / (1 - config.TEST_SIZE),
                                         stratify=rest[TARGET], random_state=config.RANDOM_STATE)  # 25% of 80%
    return train, validation, test


def seconds(call: Callable[[], object]) -> float:
    """Wall-clock seconds of one call."""
    started = time.perf_counter()
    call()
    return time.perf_counter() - started


def fit_models(models: dict[str, BaseEstimator], train: pd.DataFrame) -> dict[str, float]:
    """Fit every model on train only. Returns each model's training time in seconds."""
    times = {name: seconds(lambda: model.fit(train[FEATURES], train[TARGET])) for name, model in models.items()}
    print("Training seconds: " + ", ".join(f"{name} {value:.1f}" for name, value in times.items()))
    return times


def threshold_for_budget(y_true: pd.Series, scores: np.ndarray) -> float:
    """Lowest score whose false-positive rate is within the alert budget (flag if score >= it; inf = flag none)."""
    fpr, _, thresholds = roc_curve(y_true, scores, drop_intermediate=False)
    return float(thresholds[np.flatnonzero(fpr <= config.MAX_FALSE_POSITIVE_RATE)[-1]])


def at_threshold(y_true: pd.Series, scores: np.ndarray, threshold: float) -> dict[str, float]:
    """Recall, false alarms per day (false-positive rate x benign flows a day) and PR-AUC of one model."""
    y, flagged = np.asarray(y_true), scores >= threshold
    return {"recall": flagged[y == 1].mean(),
            "false_alarms_per_day": flagged[y == 0].mean() * config.BENIGN_FLOWS_PER_DAY,
            "pr_auc": average_precision_score(y, scores)}


def results(models: dict[str, BaseEstimator], part: pd.DataFrame, thresholds: pd.Series | None = None) -> pd.DataFrame:
    """Each model's threshold, recall, false alarms a day and PR-AUC; thresholds are set here unless given."""
    rows = {}
    for name, model in models.items():
        scores = model.predict_proba(part[FEATURES])[:, 1]
        threshold = threshold_for_budget(part[TARGET], scores) if thresholds is None else thresholds[name]
        rows[name] = {"threshold": threshold, **at_threshold(part[TARGET], scores, threshold)}
    return pd.DataFrame(rows).T


def speed_results(models: dict[str, BaseEstimator], X: pd.DataFrame) -> pd.DataFrame:
    """Speed through each whole fitted pipeline, called the way the API calls it, plus the model's file size.

    Throughput: median of 20 batches of 1,000 flows. Latency: p50 and p95 of 1,000 single flows. One CPU thread,
    as in a small container (scikit-learn would otherwise use every core of this laptop).
    """
    batches, singles = [X.iloc[i:i + 1_000] for i in range(0, 20_000, 1_000)], [X.iloc[[i]] for i in range(1_000)]
    rows = {}
    with threadpool_limits(limits=1):
        for name, model in models.items():
            model.predict_proba(batches[0])  # warm-up: the first call pays one-off costs
            batch_s = [seconds(lambda: model.predict_proba(batch)) for batch in batches]
            single_ms = [1_000 * seconds(lambda: model.predict_proba(flow)) for flow in singles]
            file = io.BytesIO()
            joblib.dump(model, file)
            rows[name] = {"flows_per_second": 1_000 / np.median(batch_s), "single_p50_ms": np.median(single_ms),
                          "single_p95_ms": np.percentile(single_ms, 95), "size_mb": file.getbuffer().nbytes / 1e6}
    return pd.DataFrame(rows).T


def choose_model(table: pd.DataFrame) -> str:
    """Highest validation recall among models meeting the speed budget; a near-tie goes to the simpler model."""
    fast_enough = table[table["flows_per_second"] >= config.MIN_FLOWS_PER_SECOND]
    best = fast_enough["val_recall"].max()
    return fast_enough.index[fast_enough["val_recall"] >= best - config.RECALL_TIE][0]


# Column: (short name, format) in the comparison table.
SHOWN = {"val_recall": ("val recall", "{:.1%}"), "val_false_alarms_per_day": ("val alarms/day", "{:,.0f}"),
         "val_pr_auc": ("val PR-AUC", "{:.3f}"), "test_recall": ("test recall", "{:.1%}"),
         "test_false_alarms_per_day": ("test alarms/day", "{:,.0f}"), "test_pr_auc": ("test PR-AUC", "{:.3f}"),
         "flows_per_second": ("flows/s", "{:,.0f}"), "single_p95_ms": ("1-flow p95 ms", "{:.1f}"),
         "size_mb": ("size MB", "{:.2f}")}


def comparison(table: pd.DataFrame) -> pd.DataFrame:
    """The comparison table for people: the columns computed so far, each in a readable format."""
    return pd.DataFrame({name: table[column].map(fmt.format) for column, (name, fmt) in SHOWN.items()
                         if column in table})


def save_bundle(model: BaseEstimator, threshold: float, metrics: dict[str, object]) -> str:
    """Save everything the API needs in one file, plus metrics.json for people. Returns the version."""
    trained_at = datetime.now(timezone.utc)
    version = trained_at.strftime("%Y%m%d-%H%M%S")
    bundle = {"pipeline": model, "threshold": threshold, "version": version, "trained_at": trained_at.isoformat(),
              "features": FEATURES, "metrics": metrics, "sklearn_version": sklearn.__version__}
    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, config.MODEL_PATH)
    config.METRICS_PATH.write_text(json.dumps({"version": version, **metrics}, indent=2) + "\n")
    return version


def main() -> None:
    raw = load_raw()
    clean = prepare_dataset(raw)
    train, validation, test = split(clean)
    print(f"Rows: raw {len(raw):,} -> without exact duplicates {len(clean):,} -> train {len(train):,} / "
          f"validation {len(validation):,} / test {len(test):,} (bad share {clean[TARGET].mean():.1%})")
    models = candidate_models()
    train_seconds = fit_models(models, train)

    # Validation: thresholds, recall and speed, then the choice. Test is not touched yet.
    on_validation = results(models, validation)
    table = on_validation.add_prefix("val_").join(speed_results(models, validation[FEATURES]))
    table["train_seconds"] = pd.Series(train_seconds)
    chosen = choose_model(table)
    too_slow = table.index[table["flows_per_second"] < config.MIN_FLOWS_PER_SECOND].tolist()
    print(f"Speed budget {config.MIN_FLOWS_PER_SECOND:,} flows/s; too slow: {too_slow or 'none'}. Chosen: {chosen}")

    # Test, once: every model at the threshold it got on validation.
    on_test = results(models, test, on_validation["threshold"])
    table = table.join(on_test.drop(columns="threshold").add_prefix("test_"))
    print(f"\n{comparison(table).to_string()}\n")
    model, threshold = models[chosen], float(on_validation.loc[chosen, "threshold"])
    flagged = model.predict_proba(test[FEATURES])[:, 1] >= threshold
    by_label = {f"recall_{label}": flagged[(test[config.LABEL_COL] == label).to_numpy()].mean()
                for label in ("malicious", "outlier")}
    print(f"{chosen} on test, recall by label: " + ", ".join(f"{k} {v:.2%}" for k, v in by_label.items()))

    rule = f"lowest validation score with false-positive rate <= {config.MAX_FALSE_POSITIVE_RATE}"
    metrics = {"rows": {"raw": len(raw), "after_exact_duplicates": len(clean), "train": len(train),
                        "validation": len(validation), "test": len(test)},
               "chosen_model": chosen, "threshold": {"value": threshold, "rule": rule},
               "test": {**on_test.loc[chosen].drop("threshold").to_dict(), **by_label},
               "models": json.loads(table.replace(np.inf, np.nan).to_json(orient="index"))}  # infinity -> null
    version = save_bundle(model, threshold, metrics)
    print(f"\nSaved {config.MODEL_PATH.name} and {config.METRICS_PATH.name} in models/ (version {version})")


if __name__ == "__main__":
    main()
