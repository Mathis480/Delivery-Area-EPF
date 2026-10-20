"""Compare current loader outputs against the frozen regression snapshot (tests/fixtures/feature_snapshot.json)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_loader import QHDataLoader
from make_feature_snapshot import FIXTURE, snapshot

if __name__ == "__main__":
    ref = json.load(open(FIXTURE))
    cur = snapshot(QHDataLoader())
    diffs = [k for k in ref if ref[k] != cur.get(k)]
    print(f"{len(ref) - len(diffs)}/{len(ref)} entries identical")
    for k in diffs[:20]:
        if isinstance(ref[k], dict):
            print(k, [f for f in ref[k] if ref[k][f] != (cur[k] or {}).get(f)])
        else:
            print(k, "names differ" if isinstance(ref[k], list) else ref[k], "->", cur.get(k) if not isinstance(ref[k], list) else "")
    sys.exit(1 if diffs else 0)
