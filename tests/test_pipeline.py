"""Behavioral tests of the forecasting pipeline (data-dependent tests are skipped without the master dataset)."""
import os
import re
import sys
import unittest

import numpy as np
import pandas as pd
import scipy.stats
from scipy.spatial.distance import cdist

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from config import (
    FEATURE_SET_MACRO, FEATURE_SET_NEIGHBOR, FEATURES_CSV, LAG_QH_BALANCING, LAG_QH_ENTSOE, MASTER_PARQUET, TEST_END,
    TEST_START,
)
from Models.csvr import _kernel, train_csvr
from Models.ensemble import dm_test, gw_test, inverse_mae_average
from Models.linear import temporal_cv_splits


@unittest.skipUnless(os.path.exists(MASTER_PARQUET), "master dataset not available")
class TestData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from data_loader import QHDataLoader
        cls.loader = QHDataLoader()
        cls.master = pd.read_parquet(MASTER_PARQUET)
        cls.master["Date"] = pd.to_datetime(cls.master["Date"], utc=True)
        cls.master = cls.master.sort_values("Date").reset_index(drop=True)

    def test_test_period(self):
        t = self.loader.test_targets("DE2", TEST_START, TEST_END)
        self.assertEqual(len(t), 35132)
        self.assertAlmostEqual(np.mean(np.abs(t["y"] - t["naive"])), 26.6818, places=4)

    def test_window_is_standardized_on_calibration_days(self):
        w = self.loader.get_window("DE2", 48, "2024-06-15", FEATURE_SET_MACRO, corr_filter=True)
        self.assertEqual(w.X_train_full.shape[0], 822)
        np.testing.assert_allclose(w.y_train.mean(), 0.0, atol=1e-10)
        np.testing.assert_allclose(w.X_train.mean(axis=0), 0.0, atol=1e-8)
        self.assertLess(w.X_train.shape[1], w.X_train_full.shape[1])  # correlation filter active on S1

    def test_ex_post_features_are_lagged(self):
        q, day = 40, "2024-06-15"
        row = self.loader._qh_tables[q]
        row = row[row["_delivery_date"] == pd.Timestamp(day).date()]
        idx = self.master.index[self.master["Date"] == row["Date"].iloc[0]][0]
        for col, lag in (("DE2_SRL_positive", LAG_QH_BALANCING), ("DE2_load_actual", LAG_QH_ENTSOE),
                         ("DE_cross_border_trading", LAG_QH_ENTSOE)):
            self.assertEqual(row[col].iloc[0], self.master.loc[idx - lag, col], col)

    def test_neighbor_features_point_to_admissible_windows(self):
        m = self.master
        np.testing.assert_allclose(m["DE2_nb_k+1_VWAP_105to120"].iloc[100:200], m["DE2_VWAP_105to120"].iloc[101:201])
        np.testing.assert_allclose(m["DE2_nb_k-1_VWAP_75to90"].iloc[100:200], m["DE2_VWAP_75to90"].iloc[99:199])
        self.assertIn("DE2_nb_k+1_VWAP_105to120", self.loader.feature_names("DE2", FEATURE_SET_NEIGHBOR))


class TestModels(unittest.TestCase):
    def test_csvr_kernel_matches_reference(self):
        """Kernel of Puć & Janczura (2024, Eq. 12) with quantile-based widths."""
        rng = np.random.default_rng(0)
        X, naive, y = rng.normal(size=(60, 8)), rng.normal(50, 10, 60), rng.normal(size=60)
        dist = cdist(X, X)
        width = np.log(2 - 2 * 0.75) / np.quantile(dist, 0.5)
        z = (naive - naive.mean()) / naive.std()
        d_naive = np.abs(z[:, None] - z[None, :]) ** 2
        sigma = np.quantile(d_naive, 0.75) / scipy.stats.norm.ppf(0.75)
        k_ref = np.exp(width * dist - 1 / (2 * sigma ** 2) * d_naive)
        model = train_csvr(X, y, naive)
        np.testing.assert_allclose(_kernel(model, X, model.bm_train_std), k_ref, rtol=1e-12)

    def test_lasso_cv_has_no_look_ahead(self):
        for train, test in temporal_cv_splits(100, 14):
            self.assertEqual(max(train) + 1, test[0])
        self.assertEqual(len(temporal_cv_splits(100, 14)), 14)

    def test_inverse_mae_average(self):
        rng = np.random.default_rng(1)
        y = rng.normal(size=(40, 96))
        good, bad = y + rng.normal(0, 0.1, y.shape), y + rng.normal(0, 1.0, y.shape)
        ens, w = inverse_mae_average({"good": good, "bad": bad}, y, calib_window=14, power=1.0)
        np.testing.assert_allclose(w.sum(axis=1), 1.0)
        np.testing.assert_allclose(w[:14], 0.5)
        self.assertTrue((w[14:, 0] > w[14:, 1]).all())

    def test_dm_and_gw_detect_better_forecast(self):
        rng = np.random.default_rng(2)
        loss_b = rng.gamma(2.0, 1.0, 366)
        loss_a = loss_b + 0.5 + rng.normal(0, 0.2, 366)
        self.assertLess(dm_test(loss_a, loss_b)[1], 0.01)
        self.assertLess(gw_test(loss_a, loss_b)[1], 0.01)
        self.assertGreater(dm_test(loss_b, loss_a)[1], 0.99)


class TestCode(unittest.TestCase):
    def test_no_backward_fill(self):
        for f in ("data_loader.py", "feature_engineering.py", "create_master_dataset.ipynb"):
            with open(os.path.join(ROOT, f)) as fh:
                self.assertIsNone(re.search(r"bfill|fillna\(method=.bfill", fh.read()), f)


if __name__ == "__main__":
    unittest.main()
