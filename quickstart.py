"""Run a small, synthetic ANN example without downloading study data."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path("runs"), help="Parent folder for the demo.")
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run = args.work_dir / f"demo_{stamp}"
    data_dir = run / "data"
    data_dir.mkdir(parents=True)

    rng = np.random.default_rng(42)
    axis = np.linspace(500, 1800, 131)
    ages = np.array([3, 10, 18])

    def generate(spectra_per_age):
        labels = np.repeat(np.arange(len(ages)), spectra_per_age)
        centers = np.array([810, 1050, 1390])[labels]
        peak = np.exp(-0.5 * ((axis - centers[:, None]) / 32) ** 2)
        shared_peak = 0.35 * np.exp(-0.5 * ((axis - 1600) / 55) ** 2)
        scale = rng.uniform(0.7, 1.3, (len(labels), 1))
        signal = scale * (peak + shared_peak) + rng.normal(0, 0.07, peak.shape)
        return signal, labels

    spectra, labels = generate(26)
    np.savez_compressed(data_dir / "demo.npz", data=spectra, labels=labels,
                        spectral_range=axis, ages_days=ages, synthetic_demo=True)
    print("Running on synthetic spectra. These scores are for demonstration only.", flush=True)
    output = run / "results"
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    subprocess.run([
        sys.executable, str(Path(__file__).with_name("train.py")),
        "--data-dir", str(data_dir), "--dataset", "demo", "--output-dir", str(output),
        "--device", "cpu", "--epochs", "30",
    ], check=True, env=env)
    print(f"Demo complete. Open {output / 'diagnostics'} for the figures and tables.")


if __name__ == "__main__":
    main()
