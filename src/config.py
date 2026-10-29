"""Every project setting in one place: paths, seed, columns, target, split, and the alert and speed budgets."""
from pathlib import Path

# --- Paths (relative to the repo root, so the project runs from any clone) ---
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
EXAMPLES_DIR = ROOT / "docs" / "examples"
FIGURES_DIR = ROOT / "reports" / "figures"

RANDOM_STATE = 42

# --- Target: "not benign" = 1 (known attacks and suspicious traffic), benign = 0 (decision 4) ---
LABEL_COL = "label"
TARGET_COL = "target"
LABEL_TO_TARGET = {"benign": 0, "malicious": 1, "outlier": 1}

# --- Model inputs: the 10 fields a caller sends to the API for one flow ---
NUMERIC_COLS = ["avg_ipt", "bytes_in", "bytes_out", "entropy", "num_pkts_out", "num_pkts_in",
                "total_entropy", "duration"]
PROTO_COL = "proto"
PORT_COL = "dest_port"
FEATURE_COLS = NUMERIC_COLS + [PROTO_COL, PORT_COL]

# --- Never model inputs (notebooks/01_eda.ipynb) ---
# IPs leak the label (labels come from IP reputation); src_port is a random ephemeral number;
# timestamps are partly corrupt, and hour-of-day on a single day would not generalize.
DROP_COLS = ["src_ip", "dest_ip", "src_port", "time_start", "time_end"]
TIME_COL = "time_start"  # used only to sort flows by time, never as an input

# A destination port gets its own one-hot column if at least 0.1% of training rows use it.
MIN_PORT_FREQUENCY = 0.001

# --- Split: train (60%) fits the models, validation (20%) picks thresholds and the model, test (20%) is used once ---
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# --- Budgets the chosen model must meet ---
# Alert budget: at most 0.01% of benign flows may be flagged. With ~1.57M benign flows a day (the one-day
# file), that is ~157 false alarms a day, a volume a small analyst team can review.
MAX_FALSE_POSITIVE_RATE = 0.0001
BENIGN_FLOWS_PER_DAY = 1_574_580
# Speed budget: one process scores at least 4,500 flows a second, 10x the busiest minute in the data
# (26,695 flows at 17:01 UTC, about 445 a second), so scoring keeps up with bursts.
MIN_FLOWS_PER_SECOND = 4_500
# Validation recall within 0.1 point of the best (about 80 bad flows) is a tie; the simpler model wins it.
RECALL_TIE = 0.001
