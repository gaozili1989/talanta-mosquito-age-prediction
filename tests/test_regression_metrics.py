"""Regression MAE arithmetic and public CLI export checks."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from evaluation import regression_metrics, score_mosquitoes, write_regression_metrics


REPO = Path(__file__).resolve().parents[1]


class RegressionMetricsTests(unittest.TestCase):
    def test_mae_is_distinct_from_rmse(self):
        metrics = regression_metrics(np.array([0., 5., 10.]), np.array([-1., 6., 14.]))
        self.assertAlmostEqual(metrics["mae_days"], 2.)
        self.assertAlmostEqual(metrics["rmse_days"], np.sqrt(6.))

    def test_export_uses_equal_mosquito_weight_with_unequal_spectrum_counts(self):
        actual = np.array([0., 0., 0., 10.])
        predicted = np.array([0., 2., 4., 16.])
        metrics = regression_metrics(actual, predicted)
        rows, mosquito_metrics = score_mosquitoes(
            ["A", "A", "A", "B"], actual, predicted, classification=False)
        self.assertEqual([row.n_spectra for row in rows], [3, 1])
        self.assertAlmostEqual(metrics["mae_days"], 3.)
        self.assertAlmostEqual(metrics["rmse_days"], np.sqrt(14.))
        self.assertAlmostEqual(mosquito_metrics["mosquito_mae_days"], 4.)
        self.assertAlmostEqual(mosquito_metrics["mosquito_rmse_days"], np.sqrt(20.))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "regression_metrics.csv"
            write_regression_metrics(path, {**metrics, **mosquito_metrics}, len(actual))
            with path.open() as file:
                exported = list(csv.DictReader(file))
        self.assertEqual([row["evaluation_level"] for row in exported], ["spectrum", "mosquito"])
        self.assertEqual([row["n_observations"] for row in exported], ["4", "2"])
        self.assertEqual([float(row["mae_days"]) for row in exported], [3., 4.])

    def test_no_ids_exports_only_spectrum_and_blank_undefined_r(self):
        metrics = regression_metrics(np.array([5., 5.]), np.array([4., 8.]))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "regression_metrics.csv"
            write_regression_metrics(path, metrics, 2)
            with path.open() as file:
                rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["evaluation_level"], "spectrum")
        self.assertEqual(rows[0]["pearson_r"], "")
        self.assertEqual(float(rows[0]["mae_days"]), 2.)

    def test_cli_all_regressors_with_and_without_ids(self):
        rng = np.random.default_rng(7)
        train_labels = np.repeat([0, 1], 4)
        test_labels = np.array([0, 0, 0, 1])
        pattern = np.array([1., 2., 3., 4.])
        data = dict(
            data_train=train_labels[:, None] * pattern + rng.normal(0, 0.1, (8, 4)),
            data_test=test_labels[:, None] * pattern + rng.normal(0, 0.1, (4, 4)),
            label_train=train_labels, label_test=test_labels,
            spectral_range=np.array([600., 610., 620., 630.]), ages_days=np.array([0., 10.]),
            synthetic_demo=np.array(True),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for with_ids in (False, True):
                with self.subTest(with_ids=with_ids):
                    inputs = root / ("with_ids" if with_ids else "without_ids")
                    inputs.mkdir()
                    ids = (dict(mosquito_id_train=np.array([f"train_{i}" for i in range(8)]),
                                mosquito_id_test=np.array(["A", "A", "A", "B"]))
                           if with_ids else {})
                    np.savez(inputs / "example_fold_0.npz", **data, **ids)
                    output = inputs / "results"
                    run = subprocess.run(
                        [sys.executable, str(REPO / "train.py"), "--data-dir", str(inputs),
                         "--dataset", "example", "--task", "regression", "--model", "all",
                         "--epochs", "2", "--device", "cpu", "--full-range", "--no-normalize",
                         "--no-diagnostics", "--output-dir", str(output)],
                        cwd=REPO, capture_output=True, text=True, timeout=120,
                        env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
                    )
                    self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                    for model in ("ann", "svr", "pls"):
                        dest = output / f"split_1_{model}"
                        metrics = json.loads((dest / "metrics.json").read_text())
                        with (dest / "regression_metrics.csv").open() as file:
                            rows = list(csv.DictReader(file))
                        self.assertEqual(len(rows), 2 if with_ids else 1)
                        self.assertEqual(rows[0]["evaluation_level"], "spectrum")
                        self.assertEqual(rows[0]["n_observations"], "4")
                        for row in rows:
                            prefix = "mosquito_" if row["evaluation_level"] == "mosquito" else ""
                            for key in ("mae_days", "rmse_days", "pearson_r"):
                                if metrics[prefix + key] is None:
                                    self.assertEqual(row[key], "")
                                else:
                                    self.assertAlmostEqual(float(row[key]), metrics[prefix + key])
                        if with_ids:
                            self.assertEqual(rows[1]["evaluation_level"], "mosquito")
                            self.assertEqual(rows[1]["n_observations"], "2")


if __name__ == "__main__":
    unittest.main()
