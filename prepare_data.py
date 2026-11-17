"""Convert a wide CSV with explicit train/hold-out assignments to a runner NPZ."""

import argparse
import csv
from pathlib import Path

import numpy as np

from datasets import DATASET_NAME, PRESET_AGES, check_mosquito_ids


def _wavenumber(column):
    try:
        return float(column.removeprefix("wn_"))
    except ValueError:
        return None


def read_csv(path, age_column, partition_column, id_column):
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fields = reader.fieldnames or []
        if len(set(fields)) != len(fields):
            raise ValueError("Column headers must be unique.")
        identity = id_column or ("mosquito_id" if "mosquito_id" in fields else None)
        required = [age_column, partition_column] + ([identity] if identity else [])
        if len(set(required)) != len(required) or not set(required).issubset(fields):
            raise ValueError(f"CSV must contain distinct metadata columns: {', '.join(required)}.")
        candidates = [field for field in fields if field not in required]
        # Prefer explicitly named spectral columns if present.
        prefixed = [field for field in candidates if field.startswith("wn_")]
        spectral = [(field, _wavenumber(field)) for field in (prefixed or candidates)]
        if prefixed and any(x is None for _, x in spectral):
            raise ValueError("Every wn_ column needs a numeric wavenumber, for example wn_650.5.")
        spectral = sorted(((field, x) for field, x in spectral if x is not None), key=lambda entry: entry[1])
        axis = np.array([x for _, x in spectral])
        if len(axis) < 2 or not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0):
            raise ValueError("Provide at least two distinct numeric wavenumber columns (e.g., wn_600, wn_610).")
        partitions = {"train": ([], [], []), "holdout": ([], [], [])}
        for line, row in enumerate(reader, start=2):
            if None in row or any(row[field] is None for field in fields):
                raise ValueError(f"CSV row {line}: wrong number of fields.")
            partition = row[partition_column].strip().lower()
            if partition == "test":
                partition = "holdout"
            if partition not in partitions:
                raise ValueError(f"CSV row {line}: partition must be train, holdout, or test.")
            try:
                age = float(row[age_column])
                spectrum = [float(row[field]) for field, _ in spectral]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"CSV row {line}: age and spectral intensities must be numeric.") from exc
            if not np.isfinite(age) or age < 0 or not np.isfinite(spectrum).all():
                raise ValueError(f"CSV row {line}: age and intensities must be finite, with age >= 0.")
            spectra, ages, identities = partitions[partition]
            spectra.append(spectrum)
            ages.append(age)
            if identity:
                identities.append(row[identity].strip())
    xt, yt, it = partitions["train"]
    xv, yv, iv = partitions["holdout"]
    if not xt or not xv or len(set(yt)) < 2:
        raise ValueError("Include both partitions and at least two distinct training ages.")
    ages = np.unique(np.r_[yt, yv])
    result = dict(data_train=np.asarray(xt), data_test=np.asarray(xv), spectral_range=axis,
                  label_train=np.searchsorted(ages, yt), label_test=np.searchsorted(ages, yv), ages_days=ages)
    if identity:
        ids = check_mosquito_ids((it, iv), yt, yv)
        result.update(mosquito_id_train=ids[0], mosquito_id_test=ids[1])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--dataset", required=True, help="New dataset name, such as my-study.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--split", type=int, default=1)
    parser.add_argument("--age-column", default="age_days")
    parser.add_argument("--partition-column", default="partition")
    parser.add_argument("--id-column", help="Defaults to mosquito_id when that column exists.")
    args = parser.parse_args()
    if not DATASET_NAME.fullmatch(args.dataset) or args.dataset in {*PRESET_AGES, "demo"} or args.split < 1:
        parser.error("Use a distinct dataset name with letters, numbers, underscores or hyphens, and split >= 1.")
    path = args.data_dir / f"{args.dataset}_fold_{args.split - 1}.npz"
    try:
        arrays = read_csv(args.csv, args.age_column, args.partition_column, args.id_column)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as file:
            np.savez_compressed(file, **arrays)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"{exc}\n")
    print(f"Saved {path}: {len(arrays['label_train'])} training and {len(arrays['label_test'])} held-out spectra.")


if __name__ == "__main__":
    main()
