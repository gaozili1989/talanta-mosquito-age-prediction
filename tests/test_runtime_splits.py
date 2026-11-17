"""Check deterministic partition membership, grouped identities, PCA, and model CLI."""

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

from datasets import discover_splits
from preprocessing import training_pca
from splitting import ordered_indices, shuffled_indices, load_partition


REPO = Path(__file__).resolve().parents[1]


class RuntimeSplitTests(unittest.TestCase):
    def test_ordered_three_positions_and_labels_keep_alignment(self):
        # Age-group labels may be interleaved; row order within each is retained.
        labels = np.tile([0, 1], 10)
        expected = {1: [7, 8, 9], 2: [3, 4, 5], 3: [0, 1, 2]}
        for split, local in expected.items():
            with patch('numpy.random.seed', side_effect=AssertionError('No partition seed')):
                train, test, info = ordered_indices(labels, split)
            np.testing.assert_array_equal(test, np.r_[np.array(local) * 2, np.array(local) * 2 + 1])
            self.assertEqual(len(train), 14)
            self.assertEqual(len(set(train) & set(test)), 0)
            self.assertEqual(sorted(np.r_[train, test]), list(range(20)))
            self.assertEqual(info['unit'], 'spectrum')
            self.assertIsNone(info['partition_seed'])

    def test_grouped_mosquitoes_with_unequal_and_noncontiguous_spectra(self):
        labels = np.array([0] * 10 + [1] * 10)
        ids = np.array(['a', 'b', 'c', 'd', 'e', 'a', 'b', 'a', 'd', 'a',
                        'v', 'w', 'x', 'y', 'z', 'v', 'w', 'v', 'y', 'v'])
        expected = {1: {'d', 'e', 'y', 'z'}, 2: {'b', 'c', 'w', 'x'}, 3: {'a', 'b', 'v', 'w'}}
        for split, identities in expected.items():
            train, test, info = ordered_indices(labels, split, ids)
            self.assertEqual(set(ids[test]), identities)
            self.assertFalse(set(ids[train]) & set(ids[test]))
            self.assertEqual(info['unit'], 'mosquito')
            self.assertEqual(sorted(np.r_[train, test]), list(range(len(ids))))
            self.assertEqual(info['counts_by_age'][0]['n_training_units'], 3)

    def test_invalid_identity_metadata_is_rejected(self):
        for identities in (['a', '', 'b', 'c'], ['a', 'b'], ['a', 'b', 'a', 'c']):
            with self.assertRaises(ValueError):
                ordered_indices(np.array([0, 0, 1, 1]), 1, identities)
        with self.assertRaises(ValueError):
            ordered_indices(np.array([0, 1]), 1)

    def test_discovery_rejects_full_and_archived_mix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'example.npz').touch()
            (root / 'example_fold_0.npz').touch()
            with self.assertRaisesRegex(ValueError, 'both a full dataset'):
                discover_splits(root)

    def test_pca_is_training_only_and_transforms_holdout(self):
        train = np.arange(36, dtype=np.float32).reshape(12, 3)
        test = np.full((2, 3), 10000, dtype=np.float32)
        xt, xe, fitted = training_pca(train, test, 2)
        np.testing.assert_allclose(fitted.mean_, train.mean(axis=0))
        np.testing.assert_array_equal(xt, fitted.transform(train))
        np.testing.assert_array_equal(xe, fitted.transform(test))
        self.assertEqual(xt.shape, (12, 2))
        with self.assertRaises(ValueError):
            training_pca(train, test, 100)

    def test_full_dataset_cli_aggregates_verified_mosquito_ids(self):
        rng = np.random.default_rng(91)
        labels = np.repeat([0, 1, 2], 10)
        identities = np.repeat([f"M{i}" for i in range(15)], 2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez(root / 'example.npz', data=rng.normal(size=(30, 6)), labels=labels,
                     mosquito_ids=identities, spectral_range=np.arange(6.) + 600,
                     ages_days=np.array([0., 5., 10.]))
            for task in ('classification', 'regression'):
                output = root / task
                run = subprocess.run([
                    sys.executable, str(REPO / 'train.py'), '--data-dir', str(root),
                    '--dataset', 'example', '--split', '2', '--split-method', 'ordered', '--task', task, '--epochs', '2',
                    '--device', 'cpu', '--full-range', '--no-normalize', '--no-diagnostics',
                    '--output-dir', str(output),
                ], cwd=REPO, capture_output=True, text=True, timeout=120,
                    env={**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                metrics = json.loads((output / 'metrics.json').read_text())
                self.assertEqual(metrics['partition']['unit'], 'mosquito')
                self.assertEqual(metrics['n_holdout_mosquitoes'], 6)
                self.assertEqual(metrics['n_holdout_spectra'], 12)
                self.assertTrue(metrics['mosquito_ids_checked'])
                self.assertTrue((output / 'mosquito_predictions.csv').is_file())

    def test_shuffled_global_ratio_stratification_and_exact_replay(self):
        labels = np.repeat([0, 1], [10, 11])
        train, test, info = shuffled_indices(labels, seed=37)
        self.assertEqual(len(train), 14)
        self.assertEqual(len(test), 7)
        self.assertEqual(sorted(np.r_[train, test]), list(range(21)))
        self.assertFalse(set(train) & set(test))
        self.assertEqual(info['partition_seed'], 37)
        self.assertEqual(info['method'], 'stratified_shuffle')
        self.assertEqual(list(np.bincount(labels[test])), [3, 4])
        with patch('sklearn.model_selection.train_test_split', side_effect=AssertionError('Do not reshuffle on replay')):
            replay_train, replay_test, replay_info = shuffled_indices(labels, seed=37, source_rows=(train, test))
        np.testing.assert_array_equal(train, replay_train)
        np.testing.assert_array_equal(test, replay_test)
        self.assertEqual(info, replay_info)
        with self.assertRaises(ValueError):
            shuffled_indices(labels, source_rows=(train, np.r_[test[:-1], train[0]]))

    def test_shuffled_grouped_ids_preserve_unequal_repeat_counts(self):
        labels, ids = [], []
        for age in range(3):
            for mosquito in range(10):
                repeats = 1 + mosquito % 3
                labels.extend([age] * repeats)
                ids.extend([f'A{age}M{mosquito}'] * repeats)
        labels, ids = np.array(labels), np.array(ids)
        train, test, info = shuffled_indices(labels, ids, seed=19)
        self.assertEqual(len(np.unique(ids[train])), 21)
        self.assertEqual(len(np.unique(ids[test])), 9)
        self.assertFalse(set(ids[train]) & set(ids[test]))
        self.assertEqual(sorted(np.r_[train, test]), list(range(len(labels))))
        self.assertEqual(info['unit'], 'mosquito')
        for age in range(3):
            self.assertEqual(len(np.unique(ids[test][labels[test] == age])), 3)

    def test_default_regression_shares_one_unseeded_draw_among_models(self):
        rng = np.random.default_rng(73)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez(root / 'example.npz', data=rng.normal(size=(30, 6)), labels=np.repeat([0, 1, 2], 10),
                     spectral_range=np.arange(6.) + 600, ages_days=np.array([0., 5., 10.]))
            output = root / 'results'
            run = subprocess.run([
                sys.executable, str(REPO / 'train.py'), '--data-dir', str(root), '--dataset', 'example',
                '--split', 'all', '--task', 'regression', '--model', 'all', '--epochs', '2',
                '--device', 'cpu', '--full-range', '--no-normalize', '--no-diagnostics',
                '--output-dir', str(output),
            ], cwd=REPO, capture_output=True, text=True, timeout=120,
                env={**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            for split in (1, 2, 3):
                manifests = []
                for model in ('ann', 'svr', 'pls'):
                    dest = output / f'split_{split}_{model}'
                    config = json.loads((dest / 'metrics.json').read_text())
                    self.assertEqual(config['partition']['method'], 'stratified_shuffle')
                    self.assertIsNone(config['partition']['partition_seed'])
                    self.assertEqual(config['n_training_spectra'], 21)
                    self.assertEqual(config['n_holdout_spectra'], 9)
                    manifests.append((dest / 'partition_indices.csv').read_text())
                self.assertEqual(len(set(manifests)), 1)

    def test_all_model_cli_shares_indices_and_exports_pca(self):
        rng = np.random.default_rng(73)
        x = rng.normal(size=(30, 6)).astype(np.float32)
        labels = np.repeat([0, 1, 2], 10)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.savez(root / 'example.npz', data=x, labels=labels,
                     spectral_range=np.arange(6.) + 600, ages_days=np.array([0., 5., 10.]))
            all_indices = []
            for task in ('classification', 'regression'):
                output = root / task
                run = subprocess.run([
                    sys.executable, str(REPO / 'train.py'), '--data-dir', str(root),
                    '--dataset', 'example', '--split', 'all', '--split-method', 'ordered', '--task', task, '--model', 'all',
                    '--epochs', '2', '--device', 'cpu', '--full-range', '--no-normalize',
                    '--pca-components', '3', '--no-diagnostics', '--output-dir', str(output),
                ], cwd=REPO, capture_output=True, text=True, timeout=120,
                    env={**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                with (output / 'summary.csv').open() as file:
                    rows = list(csv.DictReader(file))
                self.assertEqual(len(rows), 9)
                for split in (1, 2, 3):
                    names = ('ann', 'svm', 'logistic') if task == 'classification' else ('ann', 'svr', 'pls')
                    indices = []
                    for name in names:
                        dest = output / f'split_{split}_{name}'
                        config = json.loads((dest / 'metrics.json').read_text())
                        self.assertEqual(config['partition']['position'], {1: 'last', 2: 'middle', 3: 'first'}[split])
                        self.assertEqual(config['model_input_features'], 3)
                        self.assertTrue((dest / 'pca.joblib').is_file())
                        self.assertIn('pca_sha256', config)
                        self.assertIn('partition_indices_sha256', config)
                        indices.append((dest / 'partition_indices.csv').read_text())
                        if task == 'regression':
                            self.assertTrue((dest / 'regression_metrics.csv').is_file())
                    self.assertEqual(len(set(indices)), 1)
                    all_indices.append(indices[0])
            self.assertEqual(all_indices[:3], all_indices[3:])


if __name__ == '__main__':
    unittest.main()
