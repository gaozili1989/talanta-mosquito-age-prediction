"""Create ordered or random age-stratified hold-outs from one full dataset."""

import numpy as np


POSITIONS = {1: "last", 2: "middle", 3: "first"}


def _checked_labels_ids(labels, mosquito_ids):
    labels = np.asarray(labels)
    if labels.ndim != 1 or not len(labels) or not np.isfinite(labels).all():
        raise ValueError("Labels must be a nonempty, finite one-dimensional array.")
    ids = None
    if mosquito_ids is not None:
        ids = np.char.strip(np.asarray(mosquito_ids, dtype=str))
        if ids.ndim != 1 or len(ids) != len(labels) or np.any(ids == ""):
            raise ValueError("Supply one nonblank mosquito ID per spectrum.")
        by_id = {}
        for identity, age in zip(ids, labels):
            if identity in by_id and by_id[identity] != age:
                raise ValueError("One mosquito ID is associated with multiple ages.")
            by_id[identity] = age
    return labels, ids


def ordered_indices(labels, split=1, mosquito_ids=None):
    """Split each age in stored order; preserve whole identified mosquitoes.

    Without IDs, each spectrum is a splitting unit. With IDs, units are the
    mosquitoes in first-appearance order within each age. Partitioning never
    calls a random number generator or uses the ANN initialization seed.
    """
    if split not in POSITIONS:
        raise ValueError("Ordered split must be 1 (last), 2 (middle), or 3 (first).")
    labels, ids = _checked_labels_ids(labels, mosquito_ids)
    training, heldout, counts = [], [], []
    for age in np.unique(labels):
        rows = np.flatnonzero(labels == age)
        units = rows if ids is None else list(dict.fromkeys(ids[rows]))
        n_train = int(0.7 * len(units))
        n_test = len(units) - n_train
        if n_train == 0 or n_test == 0:
            raise ValueError(f"Age label {age}: need at least two splitting units.")
        start = (n_train, n_train // 2, 0)[split - 1]
        stop = start + n_test
        if ids is None:
            test_rows = rows[start:stop]
            train_rows = np.r_[rows[:start], rows[stop:]]
        else:
            mask = np.isin(ids[rows], units[start:stop])
            test_rows, train_rows = rows[mask], rows[~mask]
        training.append(train_rows)
        heldout.append(test_rows)
        counts.append(dict(label=age.item(), n_training_units=n_train, n_holdout_units=n_test,
                           n_training_spectra=len(train_rows), n_holdout_spectra=len(test_rows),
                           holdout_start_zero_based=start, holdout_end_exclusive=stop))
    return (np.concatenate(training), np.concatenate(heldout),
            dict(method="ordered_by_age", position=POSITIONS[split], shuffle=False,
                 partition_seed=None, unit="mosquito" if ids is not None else "spectrum",
                 training_fraction_target=0.7, counts_by_age=counts))


def _checked_source_rows(source_rows, n_rows, ids):
    train, test = (np.asarray(rows) for rows in source_rows)
    if any(rows.ndim != 1 or not len(rows) or rows.dtype.kind not in "iu" for rows in (train, test)):
        raise ValueError("Saved partition indices must be nonempty integer vectors.")
    if not np.array_equal(np.sort(np.r_[train, test]), np.arange(n_rows)):
        raise ValueError("Saved partition indices must assign each input row exactly once.")
    if ids is not None and set(ids[train]) & set(ids[test]):
        raise ValueError("Saved partition places a mosquito in both partitions.")
    return train, test


def shuffled_indices(labels, mosquito_ids=None, seed=None, source_rows=None):
    """Global random age-stratified 70:30 partitioning, using whole mosquitoes when IDs exist.

    With seed=None each call draws a new partition. Replaying source_rows uses
    the recorded row membership directly and never draws another partition.
    """
    labels, ids = _checked_labels_ids(labels, mosquito_ids)
    if seed is not None and not 0 <= seed < 2**32:
        raise ValueError("Split seed must be between 0 and 2**32 - 1.")
    if source_rows is None:
        from sklearn.model_selection import train_test_split
        if ids is None:
            train, test = train_test_split(np.arange(len(labels)), test_size=0.3,
                                           stratify=labels, shuffle=True, random_state=seed)
        else:
            identities = np.array(list(dict.fromkeys(ids)))
            age_by_id = dict(zip(ids, labels))
            unit_labels = np.array([age_by_id[identity] for identity in identities])
            training_ids, test_ids = train_test_split(identities, test_size=0.3,
                                                       stratify=unit_labels, shuffle=True, random_state=seed)
            train = np.flatnonzero(np.isin(ids, training_ids))
            test = np.flatnonzero(np.isin(ids, test_ids))
    else:
        train, test = _checked_source_rows(source_rows, len(labels), ids)
    counts = []
    for age in np.unique(labels):
        train_rows, test_rows = train[labels[train] == age], test[labels[test] == age]
        counts.append(dict(label=age.item(),
                           n_training_units=len(train_rows) if ids is None else len(np.unique(ids[train_rows])),
                           n_holdout_units=len(test_rows) if ids is None else len(np.unique(ids[test_rows])),
                           n_training_spectra=len(train_rows), n_holdout_spectra=len(test_rows)))
    return train, test, dict(method="stratified_shuffle", shuffle=True, partition_seed=seed,
                             unit="mosquito" if ids is not None else "spectrum",
                             training_fraction_target=0.7, counts_by_age=counts)


def load_partition(path, split=1, method="ordered", seed=None, source_rows=None):
    """Read a full NPZ and partition in memory, or read a legacy saved split."""
    if method not in ("ordered", "shuffled"):
        raise ValueError("Partition method must be ordered or shuffled.")
    with np.load(path, allow_pickle=False) as source:
        saved = {key: source[key] for key in source.files}
    full = "data" in saved or "labels" in saved
    legacy = "data_train" in saved or "data_test" in saved
    if full and legacy:
        raise ValueError("NPZ contains both full-dataset and saved-split arrays; choose one format.")
    if full:
        required = {"data", "labels", "spectral_range"}
        if not required.issubset(saved):
            raise ValueError("Full NPZ requires data, labels, and spectral_range.")
        if "mosquito_id_train" in saved or "mosquito_id_test" in saved:
            raise ValueError("Full NPZ uses mosquito_ids, not partition-specific ID arrays.")
        x, y = saved["data"], saved["labels"]
        if x.ndim != 2 or y.ndim != 1 or len(x) != len(y):
            raise ValueError("Full NPZ needs data[n_spectra, n_wavenumbers] and labels[n_spectra].")
        if method == "shuffled":
            train, test, info = shuffled_indices(y, saved.get("mosquito_ids"), seed, source_rows)
        else:
            train, test, info = ordered_indices(y, split, saved.get("mosquito_ids"))
            if source_rows is not None:
                replay_train, replay_test = _checked_source_rows(source_rows, len(y), saved.get("mosquito_ids"))
                if not np.array_equal(train, replay_train) or not np.array_equal(test, replay_test):
                    raise ValueError("Saved row membership differs from the requested ordered split.")
        arrays = {key: saved[key] for key in ("spectral_range", "ages_days", "synthetic_demo") if key in saved}
        arrays.update(data_train=x[train], data_test=x[test], label_train=y[train], label_test=y[test],
                      source_row_train=train, source_row_test=test)
        if "mosquito_ids" in saved:
            arrays.update(mosquito_id_train=saved["mosquito_ids"][train],
                          mosquito_id_test=saved["mosquito_ids"][test])
        info["source_row_index_space"] = "zero-based row of input data array"
        return arrays, info
    required = {"data_train", "data_test", "label_train", "label_test", "spectral_range"}
    if not required.issubset(saved):
        raise ValueError("NPZ must contain either full data/labels or all saved-split arrays.")
    if "mosquito_ids" in saved:
        raise ValueError("Saved-split NPZ requires mosquito_id_train and mosquito_id_test.")
    n_train = len(saved["data_train"])
    saved["source_row_train"] = np.arange(n_train)
    saved["source_row_test"] = np.arange(n_train, n_train + len(saved["data_test"]))
    if source_rows is not None and any(not np.array_equal(expected, rows) for expected, rows in
                                       zip((saved["source_row_train"], saved["source_row_test"]), source_rows)):
        raise ValueError("Saved row membership differs from the archived split.")
    return saved, dict(method="saved_partition", shuffle=False, partition_seed=None,
                       unit="mosquito" if "mosquito_id_train" in saved else "spectrum",
                       source_row_index_space="zero-based row after concatenating input data_train then data_test")
