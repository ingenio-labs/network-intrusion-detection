"""Load the flows, build the target, and define the preprocessing shared by training and the API.

`build_preprocessor()` is saved inside the model pipeline, so the API transforms a request exactly the way
training did (no train/serve skew). The pipeline stores `port_service` and `port_range` by name:
retrain after changing them.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import config

EPHEMERAL_PORT_START = 49152  # IANA: ports from here up are picked at random by the OS for replies


def load_raw() -> pd.DataFrame:
    """Read the CSV with no cleaning: DATA_PATH if it is set, otherwise the single CSV in data/raw/."""
    if "DATA_PATH" in os.environ:
        return pd.read_csv(os.environ["DATA_PATH"])
    csv_files = sorted(config.RAW_DIR.glob("*.csv"))
    if len(csv_files) != 1:
        raise FileNotFoundError(f"Expected one CSV in {config.RAW_DIR}, found {len(csv_files)}. Set DATA_PATH.")
    return pd.read_csv(csv_files[0])


def make_target(labels: pd.Series) -> pd.Series:
    """1 = malicious or outlier, 0 = benign. An unknown label stops training instead of being mislabeled."""
    unknown = set(labels.unique()) - set(config.LABEL_TO_TARGET)
    if unknown:
        raise ValueError(f"Unknown labels in data: {sorted(unknown)}")
    return labels.map(config.LABEL_TO_TARGET).astype("int8")


def repair_timestamp(raw: pd.Series) -> pd.Series:
    """Epoch microseconds. In ~10% of rows the fraction lost its trailing zeros (018980 became 01898)."""
    text = raw.astype(str)
    return text.str[:10].astype("int64") * 1_000_000 + text.str[10:].str.ljust(6, "0").astype("int64")


def prepare_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    """One clean table: the 10 model inputs, label, target and event time, sorted by time.

    Only exact copies (all 16 columns equal) are dropped: repeated flows are real traffic (decision 10).
    Sorting by time makes the train/test split independent of the file order. `label` and `event_time`
    are kept for evaluation only (recall per label, the time-based check); they are never model inputs.
    """
    unique = raw.drop_duplicates()
    clean = unique[config.FEATURE_COLS + [config.LABEL_COL]].copy()
    clean[config.TARGET_COL] = make_target(clean[config.LABEL_COL])
    clean["event_time"] = pd.to_datetime(repair_timestamp(unique[config.TIME_COL]), unit="us")
    return clean.sort_values("event_time", kind="stable").reset_index(drop=True)


def port_service(ports: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Keep real service ports (22, 445, 9200...) but turn every random ephemeral port into "ephemeral"."""
    values = np.asarray(ports, dtype="int64").ravel()
    return np.where(values >= EPHEMERAL_PORT_START, "ephemeral", values.astype(str)).reshape(-1, 1)


def port_range(ports: pd.DataFrame | np.ndarray) -> np.ndarray:
    """IANA port range, so a rare or unseen port still means something: no port, well-known, registered, ephemeral."""
    values = np.asarray(ports, dtype="int64").ravel()
    names = np.select([values == 0, values <= 1023, values < EPHEMERAL_PORT_START],
                      ["no_port", "well_known", "registered"], default="ephemeral")
    return names.reshape(-1, 1)


def build_preprocessor() -> ColumnTransformer:
    """Turn the 10 raw flow fields into model features.

    - Numeric: log1p (tames heavy tails, keeps 0 at 0), then StandardScaler.
    - proto: one-hot; an unseen protocol becomes all zeros instead of an error.
    - dest_port: one-hot for common service ports; rare and unseen ports share one "infrequent" column.
    - dest_port range: one-hot of `port_range`.
    """
    numeric = Pipeline([("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                        ("scale", StandardScaler())])
    port = Pipeline([("service", FunctionTransformer(port_service, feature_names_out="one-to-one")),
                     ("one_hot", OneHotEncoder(min_frequency=config.MIN_PORT_FREQUENCY,
                                               handle_unknown="infrequent_if_exist"))])
    port_group = Pipeline([("range", FunctionTransformer(port_range, feature_names_out="one-to-one")),
                           ("one_hot", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer(
        [("numeric", numeric, config.NUMERIC_COLS),
         ("proto", OneHotEncoder(handle_unknown="ignore"), [config.PROTO_COL]),
         ("port", port, [config.PORT_COL]),
         ("port_range", port_group, [config.PORT_COL])],
        sparse_threshold=0,  # dense output: every model we try accepts it
    )
