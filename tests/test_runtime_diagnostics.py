"""Rebuild held-out ANN features from saved full-dataset and legacy runs."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from diagnostics import _saved_ann_features


REPO = Path(__file__).resolve().parents[1]


class RuntimeDiagnosticsTests(unittest.TestCase):
    def test_saved_features_replay_full_dataset_and_legacy_with_optional_pca(self):
        rng = np.random.default_rng(17)
        labels = np.repeat([0, 1], 10)
        axis = np.linspace(500., 1800., 80)
        values = rng.normal(size=(20, 80)) + labels[:, None] * np.sin(axis / 90.)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = [(False, split, method) for method in (None, "ordered") for split in (1, 2, 3)]
            cases.append((True, 1, None))
            for legacy, split, method in cases:
                for pca in (0, 3):
                    with self.subTest(legacy=legacy, split=split, method=method, pca=pca):
                        source_dir = root / f"legacy_{legacy}_split_{split}_method_{method}_pca_{pca}"
                        source_dir.mkdir()
                        if legacy:
                            train = np.r_[0:7, 10:17]
                            test = np.r_[7:10, 17:20]
                            source = source_dir / "example_fold_0.npz"
                            np.savez(source, data_train=values[train], data_test=values[test],
                                     label_train=labels[train], label_test=labels[test],
                                     spectral_range=axis, ages_days=np.array([0., 10.]))
                        else:
                            source = source_dir / "example.npz"
                            np.savez(source, data=values, labels=labels, spectral_range=axis,
                                     ages_days=np.array([0., 10.]))
                        out = source_dir / "results"
                        args = [sys.executable, str(REPO / "train.py"), "--data-dir", str(source_dir),
                                "--dataset", "example", "--task", "regression", "--model", "ann",
                                "--split", str(split), "--epochs", "2", "--device", "cpu", "--full-range",
                                "--second-derivative", "--no-diagnostics", "--output-dir", str(out)]
                        if pca:
                            args.extend(["--pca-components", str(pca)])
                        if method:
                            args.extend(["--split-method", method])
                        process = subprocess.run(args, cwd=REPO, capture_output=True, text=True,
                                                 timeout=120, env={**os.environ, "OMP_NUM_THREADS": "1",
                                                                   "OPENBLAS_NUM_THREADS": "1"})
                        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
                        run = out
                        config = json.loads((run / "metrics.json").read_text())
                        with (run / "predictions.csv").open() as file:
                            rows = list(csv.DictReader(file))
                        actual = np.array([float(row["true_age_days"]) for row in rows])
                        predicted = np.array([float(row["predicted_age_days"]) for row in rows])
                        expected_method = ("saved_partition" if legacy else
                                           "ordered_by_age" if method == "ordered" else "stratified_shuffle")
                        self.assertEqual(config["partition"]["method"], expected_method)
                        self.assertIsNone(config["partition"]["partition_seed"])
                        with patch("sklearn.model_selection.train_test_split",
                                   side_effect=AssertionError("Diagnostics must not draw another split.")):
                            ages, features = _saved_ann_features(run, source_dir, config, actual, predicted)
                        np.testing.assert_array_equal(ages, [0., 10.])
                        self.assertEqual(features.shape, (len(rows), 100))
                        self.assertTrue(np.isfinite(features).all())
                        with self.assertRaisesRegex(ValueError, "input"):
                            _saved_ann_features(run, source_dir, {**config, "input_sha256": "invalid"},
                                                actual, predicted)
                        manifest = run / "partition_indices.csv"
                        manifest_bytes = manifest.read_bytes()
                        manifest.write_bytes(manifest_bytes + b"\n")
                        with self.assertRaisesRegex(ValueError, "partition indices"):
                            _saved_ann_features(run, source_dir, config, actual, predicted)
                        manifest.write_bytes(manifest_bytes)
                        if pca:
                            transform = run / "pca.joblib"
                            transform.write_bytes(transform.read_bytes() + b"\n")
                            with self.assertRaisesRegex(ValueError, "PCA transform"):
                                _saved_ann_features(run, source_dir, config, actual, predicted)


if __name__ == "__main__":
    unittest.main()
