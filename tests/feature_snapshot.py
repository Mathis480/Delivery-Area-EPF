"""
Regression check of loader outputs (hashes for a fixed sample of windows) against tests/fixtures/feature_snapshot.json.

Usage: python tests/feature_snapshot.py [--update]   (--update freezes the current outputs as new reference)
"""
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FEATURE_SETS, ZONES
from data_loader import QHDataLoader

QHS = [0, 1, 47, 94, 95]
DATES = ["2024-01-01", "2024-03-31", "2024-06-15", "2024-10-27", "2024-12-31"]
FIELDS = ["X_train_full", "X_test_full", "X_train", "col_mask", "y_train", "bm_train"]
FIXTURE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "feature_snapshot.json")


def snapshot(loader: QHDataLoader) -> dict:
    h = lambda a: hashlib.sha256(np.ascontiguousarray(np.asarray(a, dtype=np.float64)).tobytes()).hexdigest()
    out = {}
    for z in ZONES:
        for s in FEATURE_SETS:
            out[f"{z}|{s}|names"] = loader.feature_names(z, s)
            for q in QHS:
                for d in DATES:
                    w = loader.get_window(z, q, d, s, corr_filter=True)
                    out[f"{z}|{s}|{q}|{d}"] = None if w is None else {f: h(getattr(w, f)) for f in FIELDS}
    return out


if __name__ == "__main__":
    cur = snapshot(QHDataLoader())
    if "--update" in sys.argv:
        json.dump(cur, open(FIXTURE, "w"), indent=0)
        sys.exit(print(f"Wrote {len(cur)} entries to {FIXTURE}"))
    ref = json.load(open(FIXTURE))
    diffs = [k for k in ref if ref[k] != cur.get(k)]
    print(f"{len(ref) - len(diffs)}/{len(ref)} entries identical", *diffs[:20], sep="\n")
    sys.exit(1 if diffs else 0)
