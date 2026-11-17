"""Selected regression memberships must not alter original classification data."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from partition_plan import load_task_partition


class SelectedPartitionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / 'example.npz'
        np.savez(self.source, data=np.random.default_rng(2).normal(size=(20, 6)),
                 labels=np.repeat([0, 1], 10), spectral_range=np.arange(600., 606.), ages_days=[0., 7.])
        self.test = [0, 1, 2, 10, 11, 12]
        self.train = [i for i in range(20) if i not in self.test]
        self.plan = dict(format_version=1, task='regression', datasets={'example': dict(
            input_file=self.source.name, input_sha256=hashlib.sha256(self.source.read_bytes()).hexdigest(),
            splits={'1': dict(train=self.train, holdout=self.test, partition_seed=None)})})
        (self.root / 'regression_partitions.json').write_text(json.dumps(self.plan))

    def tearDown(self):
        self.tmp.cleanup()

    def test_selected_regression_rows_and_original_classification_without_rng(self):
        with patch('sklearn.model_selection.train_test_split', side_effect=AssertionError('Do not draw another split')):
            regression, _, provenance = load_task_partition(self.source, 'example', 1, 'regression')
            classification, _, no_plan = load_task_partition(self.source, 'example', 1, 'classification')
        np.testing.assert_array_equal(regression['source_row_test'], self.test)
        np.testing.assert_array_equal(classification['source_row_test'], [7, 8, 9, 17, 18, 19])
        self.assertTrue(provenance['reused_recorded_rows'])
        self.assertIsNone(no_plan)

    def test_reject_wrong_source_and_allow_explicit_new_draw(self):
        self.plan['datasets']['example']['input_sha256'] = 'wrong'
        (self.root / 'regression_partitions.json').write_text(json.dumps(self.plan))
        with self.assertRaisesRegex(ValueError, 'does not match'):
            load_task_partition(self.source, 'example', 1, 'regression')
        _, info, provenance = load_task_partition(self.source, 'example', 1, 'regression', seed=81)
        self.assertIsNone(provenance)
        self.assertEqual(info['partition_seed'], 81)
        # A corrupt regression plan must not affect classification.
        _, _, provenance = load_task_partition(self.source, 'example', 1, 'classification')
        self.assertIsNone(provenance)

    def test_all_regressors_receive_selected_rows_and_record_plan(self):
        output = self.root / 'output'
        repo = Path(__file__).resolve().parents[1]
        run = subprocess.run([sys.executable, str(repo / 'train.py'), '--data-dir', str(self.root),
                              '--dataset', 'example', '--task', 'regression', '--model', 'all',
                              '--epochs', '2', '--device', 'cpu', '--no-diagnostics', '--output-dir', str(output)],
                             capture_output=True, text=True, timeout=120,
                             env={**os.environ, 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'})
        self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
        for model in ('ann', 'svr', 'pls'):
            dest = output / f'split_1_{model}'
            with (dest / 'partition_indices.csv').open() as f:
                holdout = [int(r['source_row_zero_based']) for r in csv.DictReader(f) if r['partition']=='holdout']
            self.assertEqual(holdout, self.test)
            metrics = json.loads((dest / 'metrics.json').read_text())
            self.assertTrue(metrics['partition_plan']['reused_recorded_rows'])
            self.assertIsNone(metrics['partition']['partition_seed'])


if __name__ == '__main__':
    unittest.main()
