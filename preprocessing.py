"""Load saved spectral splits and apply per-spectrum preprocessing."""

from pathlib import Path

import numpy as np


def available_bounds(wavenumbers):
    """Return the measured interval for a saved wavenumber axis."""
    axis = np.asarray(wavenumbers, dtype=np.float64).reshape(-1)
    if axis.size < 2 or not np.isfinite(axis).all():
        raise ValueError("The wavenumber axis must contain at least two finite points.")
    return float(axis.min()), float(axis.max())


def read_saved_bounds(path: Path):
    """Inspect the available range without loading spectra or importing PyTorch."""
    with np.load(path, allow_pickle=False) as data:
        return available_bounds(data["spectral_range"])


def norris_second_derivative(spectra, *, segment=9, gap=9):
    """Apply the original notebook's Norris filter without padded edges.

    Returns the filtered spectra and the number of points trimmed at each end.
    """
    if segment < 1 or segment % 2 != 1 or gap < 0:
        raise ValueError("Norris segment must be positive and odd; gap must be nonnegative.")
    x = np.asarray(spectra, dtype=np.float64)
    kernel = np.r_[np.ones(segment), np.zeros(gap), -2 * np.ones(segment),
                   np.zeros(gap), np.ones(segment)]
    if x.ndim != 2 or not x.shape[0] or x.shape[1] < kernel.size:
        raise ValueError(f"Norris filtering requires at least {kernel.size} wavenumber points.")
    filtered = np.stack([np.convolve(row, kernel, mode="valid") for row in x])
    return filtered, kernel.size // 2


def preprocess_spectra(spectra, wavenumbers, *, lower=None, upper=None,
                       normalize=True, second_derivative=False, segment=9, gap=9):
    """Optionally crop, apply row-wise SNV, and apply the Norris filter in that order."""
    x = np.asarray(spectra, dtype=np.float64)
    axis = np.asarray(wavenumbers, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or x.shape[1] != axis.size or not x.shape[0]:
        raise ValueError("Expected a nonempty spectra-by-wavenumber matrix.")
    if not np.isfinite(axis).all() or not np.isfinite(x).all():
        raise ValueError("Spectra and wavenumbers must be finite.")
    if (lower is None) != (upper is None):
        raise ValueError("Provide both range endpoints or neither.")
    if lower is not None:
        if not np.isfinite([lower, upper]).all() or lower >= upper:
            raise ValueError("Range endpoints must be finite with lower < upper.")
        measured_min, measured_max = available_bounds(axis)
        if lower < measured_min or upper > measured_max:
            raise ValueError(
                f"Requested range [{lower:g}, {upper:g}] cm⁻¹ is outside the measured "
                f"range [{measured_min}, {measured_max}] cm⁻¹. "
                "Choose endpoints within that interval, or use --full-range."
            )
        keep = (axis >= lower) & (axis <= upper)
        x, axis = x[:, keep], axis[keep]
    if axis.size < 2:
        raise ValueError("The selected range must contain at least two wavenumber points.")
    if normalize:
        sd = x.std(axis=1, keepdims=True)
        if np.any(sd == 0):
            raise ValueError("SNV is undefined for a constant spectrum.")
        x = (x - x.mean(axis=1, keepdims=True)) / sd
    if second_derivative:
        x, trim = norris_second_derivative(x, segment=segment, gap=gap)
        axis = axis[trim:-trim]
    return x.astype(np.float32), axis


def load_split(path: Path, *, ages, lower=None, upper=None, normalize=True,
               second_derivative=False, segment=9, gap=9, require_all_training_ages=True, split=1):
    """Load a dataset or legacy split, then apply the requested preprocessing."""
    from splitting import load_partition
    arrays, _ = load_partition(path, split)
    return preprocess_partition(arrays, ages=ages, lower=lower, upper=upper, normalize=normalize,
                                second_derivative=second_derivative, segment=segment, gap=gap,
                                require_all_training_ages=require_all_training_ages)


def preprocess_partition(data, *, ages, lower=None, upper=None, normalize=True,
                         second_derivative=False, segment=9, gap=9, require_all_training_ages=True):
    """Apply per-spectrum processing after partitioning, keeping labels aligned."""
    options = dict(lower=lower, upper=upper, normalize=normalize,
                   second_derivative=second_derivative, segment=segment, gap=gap)
    x_train, axis = preprocess_spectra(data["data_train"], data["spectral_range"], **options)
    x_eval, _ = preprocess_spectra(data["data_test"], data["spectral_range"], **options)
    y_train = data["label_train"].reshape(-1)
    y_eval = data["label_test"].reshape(-1)
    valid = np.arange(len(ages))
    if not np.isin(y_train, valid).all() or not np.isin(y_eval, valid).all():
        raise ValueError("Labels must be zero-based indices into the dataset's age list.")
    if len(x_train) != len(y_train) or len(x_eval) != len(y_eval):
        raise ValueError("Spectra and labels do not have matching row counts.")
    if require_all_training_ages and not np.array_equal(np.unique(y_train), valid):
        raise ValueError("Classification requires training spectra for every listed age class.")
    if len(np.unique(y_train)) < 2:
        raise ValueError("Training needs at least two distinct ages.")
    return x_train, y_train.astype(int), x_eval, y_eval.astype(int), axis


def training_pca(x_train, x_eval, n_components):
    """Fit PCA on training spectra only and transform both partitions."""
    if n_components is None:
        return x_train, x_eval, None
    if n_components < 1 or n_components > min(x_train.shape):
        raise ValueError(f"PCA components must be between 1 and {min(x_train.shape)} for this training matrix.")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components, svd_solver="full", whiten=False)
    pca.fit(x_train)
    return (pca.transform(x_train).astype(np.float32),
            pca.transform(x_eval).astype(np.float32), pca)
