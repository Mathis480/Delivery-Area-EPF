"""Consistency of features.csv with the master dataset and the information cutoff."""
import os
import sys
import unittest

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EFFECTIVE_CUTOFF_MINUTES, FEATURES_CSV, MASTER_PARQUET, ZONES

SETS = ["S1", "S2", "S3", "S4"]
EXPECTED_COUNTS = {  # zone -> (S1, S2, S3, S4), as reported in the paper (Table FeatureSets)
    "DE1": (89, 31, 31, 16), "DE2": (89, 31, 31, 16), "DE3": (89, 31, 34, 16), "DE4": (89, 31, 34, 16),
}


class TestFeatureCatalog(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cat = pd.read_csv(FEATURES_CSV)
        cls.master = pq.read_schema(MASTER_PARQUET).names

    def test_catalog_matches_master_one_to_one(self):
        self.assertFalse(self.cat["name"].duplicated().any())
        self.assertEqual(set(self.cat["name"]), set(self.master))

    def test_only_inputs_in_sets(self):
        in_set = self.cat[SETS].notna().any(axis=1)
        self.assertTrue((self.cat.loc[in_set, "tag"] == "input").all())

    def test_inputs_respect_cutoff(self):
        """Window end relative to the target delivery (neighbors: own window end - 15k) must be >= 90 min."""
        inputs = self.cat[self.cat["tag"] == "input"]
        late = inputs[inputs["window_end_min"].notna() & (inputs["window_end_min"] < EFFECTIVE_CUTOFF_MINUTES)]
        self.assertTrue(late.empty, late["name"].tolist())

    def test_set_sizes(self):
        inputs = self.cat[self.cat["tag"] == "input"]
        for zone in ZONES:
            z = inputs[inputs["model"].isin(["all", zone])]
            self.assertEqual(tuple(int(z[s].notna().sum()) for s in SETS), EXPECTED_COUNTS[zone], zone)


if __name__ == "__main__":
    unittest.main()
