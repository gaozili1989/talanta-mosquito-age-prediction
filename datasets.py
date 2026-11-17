"""Discover saved datasets and check age mappings and specimen metadata."""

import re

import numpy as np


PRESET_AGES = {"culex": [0, 7, 14, 21, 28], "lakewood": [1, 7, 10, 14, 21, 28, 35],
               "thai-sonication": [3, 10, 18], "thai-bead": [3, 10, 18]}
DATASET_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
SPLIT_FILENAME = re.compile(r"([A-Za-z0-9][A-Za-z0-9_-]*)_fold_(0|[1-9][0-9]*)\.npz\Z")


def discover_splits(data_dir):
    """Discover full dataset files and legacy split files, rejecting ambiguity."""
    found, full_names = {}, set()
    for path in sorted(data_dir.glob("*.npz")):
        if not path.is_file():
            continue
        match = SPLIT_FILENAME.fullmatch(path.name)
        if match:
            dataset, index = match.groups()
            if dataset in full_names:
                raise ValueError(f"{dataset}: both a full dataset and saved splits exist. Use a folder with one format.")
            found.setdefault(dataset, {})[int(index) + 1] = path
        elif DATASET_NAME.fullmatch(path.stem):
            dataset = path.stem
            if dataset in found:
                raise ValueError(f"{dataset}: both a full dataset and saved splits exist. Use a folder with one format.")
            full_names.add(dataset)
            found[dataset] = {split: path for split in (1, 2, 3)}
    return found


def _checked_ages(values):
    ages = np.asarray(values, dtype=float)
    if (ages.ndim != 1 or len(ages) < 2 or not np.isfinite(ages).all()
            or (ages < 0).any() or len(np.unique(ages)) != len(ages)):
        raise ValueError("Age mapping must contain at least two distinct, finite, nonnegative ages in days.")
    if (ages < 2**63).all() and np.all(ages == np.floor(ages)):
        return ages.astype(np.int64)
    return ages


def read_split_metadata(path, dataset, explicit_ages=None, split=1):
    """Compatibility wrapper for callers with an input path."""
    from splitting import load_partition
    arrays, _ = load_partition(path, split)
    return read_partition_metadata(arrays, dataset, explicit_ages)


def read_partition_metadata(data, dataset, explicit_ages=None):
    stored = _checked_ages(data["ages_days"]) if "ages_days" in data else None
    names = ("mosquito_id_train", "mosquito_id_test")
    has_ids = [key in data for key in names]
    if any(has_ids) and not all(has_ids):
        raise ValueError("Supply mosquito IDs for both partitions or neither.")
    ids = tuple(np.asarray(data[key], dtype=str) for key in names) if all(has_ids) else None
    synthetic = bool(data["synthetic_demo"].item()) if "synthetic_demo" in data else False
    preset = PRESET_AGES.get(dataset)
    provided = _checked_ages(explicit_ages) if explicit_ages is not None else None
    if preset is not None:
        preset = _checked_ages(preset)
        if stored is not None and not np.array_equal(stored, preset):
            raise ValueError("Stored ages conflict with the Talanta preset. Use a distinct dataset name for custom data.")
        if provided is not None and not np.array_equal(provided, preset):
            raise ValueError("--ages conflicts with the Talanta preset age mapping.")
    if stored is not None and provided is not None and not np.array_equal(stored, provided):
        raise ValueError("--ages conflicts with the saved ages_days mapping.")
    ages = next((a for a in (stored, provided, preset) if a is not None), None)
    if ages is None:
        raise ValueError("Custom NPZ files need ages_days or --ages DAYS in zero-based label order.")
    return ages, ids, synthetic


def check_mosquito_ids(ids, y_train, y_test):
    """Reject shared IDs and conflicting ages. Do not invent IDs for unlabeled spectra."""
    if ids is None:
        return None
    train, test = (np.char.strip(np.asarray(a, dtype=str)) for a in ids)
    if any(a.ndim != 1 or len(a) != len(y) or np.any(a == "")
           for a, y in ((train, y_train), (test, y_test))):
        raise ValueError("Each spectrum must have one nonblank mosquito ID in both partitions.")
    overlap = set(train) & set(test)
    if overlap:
        raise ValueError(f"{len(overlap)} mosquito ID(s) appear in both training and held-out spectra.")
    for group, labels in ((train, y_train), (test, y_test)):
        by_id = {}
        for identity, age in zip(group, labels):
            if identity in by_id and by_id[identity] != age:
                raise ValueError("One mosquito ID is associated with multiple ages in the same split.")
            by_id[identity] = age
    return train, test
