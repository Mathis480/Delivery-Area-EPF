"""Freeze hashes of loader outputs for a fixed sample of windows (regression reference for feature refactors)."""
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEATURE_SET_BALANCE, FEATURE_SET_FUNDAMENTAL, FEATURE_SET_MACRO, FEATURE_SET_NEIGHBOR, ZONES
from data_loader import QHDataLoader

SETS = [FEATURE_SET_MACRO, FEATURE_SET_NEIGHBOR, FEATURE_SET_FUNDAMENTAL, FEATURE_SET_BALANCE]
QHS = [0, 1, 47, 94, 95]
DATES = ["2024-01-01", "2024-03-31", "2024-06-15", "2024-10-27", "2024-12-31"]
FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "feature_snapshot.json")


def _h(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes()).hexdigest()


def snapshot(loader: QHDataLoader) -> dict:
    out = {}
    for z in ZONES:
        for s in SETS:
            out[f"{z}|{s}|names"] = loader.feature_names(z, feature_set=s)
            for q in QHS:
                for d in DATES:
                    w = loader.get_window(z, q, d, s, corr_filter=True)
                    if w is None:
                        out[f"{z}|{s}|{q}|{d}"] = None
                        continue
                    out[f"{z}|{s}|{q}|{d}"] = {
                        "X_train_full": _h(w.X_train_full), "X_test_full": _h(w.X_test_full),
                        "X_train": _h(w.X_train), "col_mask": _h(w.col_mask),
                        "y_train": _h(w.y_train), "bm_train": _h(w.bm_train),
                    }
    return out


if __name__ == "__main__":
    snap = snapshot(QHDataLoader())
    with open(FIXTURE, "w") as f:
        json.dump(snap, f, indent=0)
    print(f"Wrote {len(snap)} entries to {FIXTURE}")
