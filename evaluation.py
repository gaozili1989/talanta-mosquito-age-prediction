"""Score age predictions and aggregate held-out spectra by mosquito ID."""

from collections import Counter
import csv
from typing import NamedTuple

import numpy as np


class MosquitoPrediction(NamedTuple):
    mosquito_id: str
    n_spectra: int
    true_age_days: float
    predicted_age_days: float | None
    top_vote_fraction: float | None


def regression_metrics(actual, predicted):
    error = predicted - actual
    r = None
    if np.any(actual != actual[0]) and np.any(predicted != predicted[0]):
        r = float(np.corrcoef(actual, predicted)[0, 1])
    return {"mae_days": float(np.mean(np.abs(error))),
            "rmse_days": float(np.sqrt(np.mean(error ** 2))), "pearson_r": r}


def score_mosquitoes(ids, actual, predicted, classification):
    """Return one row per mosquito, followed by mosquito-level metrics."""
    groups = {}
    for identity, age, prediction in zip(ids, actual, predicted, strict=True):
        if not identity or not identity.strip():
            raise ValueError("Each held-out spectrum needs a nonblank mosquito ID.")
        identity = identity.strip()
        true_age, votes = groups.setdefault(identity, (age, []))
        if true_age != age:
            raise ValueError("One mosquito ID is associated with multiple held-out ages.")
        votes.append(prediction)

    rows = []
    for identity, (age, votes) in groups.items():
        n_spectra = len(votes)
        fraction = None
        if classification:
            winner, count = Counter(votes).most_common(1)[0]
            fraction = count / n_spectra
            prediction = winner if 2 * count > n_spectra else None
        else:
            prediction = float(np.mean(votes))
        rows.append(MosquitoPrediction(identity, n_spectra, age, prediction, fraction))

    metrics = {"n_holdout_mosquitoes": len(rows)}
    if classification:
        correct = sum(row.predicted_age_days == row.true_age_days for row in rows)
        metrics.update(correct_mosquitoes=int(correct), mosquito_accuracy=float(correct / len(rows)),
                       n_unassigned_mosquitoes=sum(row.predicted_age_days is None for row in rows))
    else:
        ages = np.asarray([row.true_age_days for row in rows])
        estimates = np.asarray([row.predicted_age_days for row in rows])
        metrics.update({f"mosquito_{name}": value for name, value in regression_metrics(ages, estimates).items()})
    return rows, metrics


def write_mosquito_predictions(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(MosquitoPrediction._fields)
        writer.writerows(rows)


def write_regression_metrics(path, metrics, n_spectra):
    """Write one overall regression score row per available evaluation level."""
    fields = ["evaluation_level", "n_observations", "mae_days", "rmse_days", "pearson_r"]
    rows = [{"evaluation_level": "spectrum", "n_observations": n_spectra,
             **{name: metrics[name] for name in fields[2:]}}]
    if "n_holdout_mosquitoes" in metrics:
        rows.append({"evaluation_level": "mosquito",
                     "n_observations": metrics["n_holdout_mosquitoes"],
                     **{name: metrics[f"mosquito_{name}"] for name in fields[2:]}})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
