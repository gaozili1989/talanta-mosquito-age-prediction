"""Generate held-out spectrum reports for a new or saved model run."""

import argparse
from contextlib import nullcontext
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluation import score_mosquitoes, write_mosquito_predictions


DATASET_NAMES = {"culex": "KNWR Culex", "lakewood": "Lakewood Aedes",
                 "thai-sonication": "Thai sonication", "thai-bead": "Thai bead disruption"}
MODEL_NAMES = {"ann": "ANN", "svm": "SVM", "logistic": "Logistic regression", "svr": "SVR", "pls": "PLS"}


def _csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


def _save(fig, path, plt):
    fig.savefig(path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _classification(out, title, ages, actual, predicted, plt, *, mosquito=False):
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    units = "mosquitoes" if mosquito else "spectra"
    classes = {age: i for i, age in enumerate(ages)}
    unassigned = mosquito and any(age is None for age in predicted)
    try:
        y_true = [classes[age] for age in actual]
        y_pred = [len(ages) if age is None and mosquito else classes[age] for age in predicted]
    except KeyError as exc:
        raise ValueError("The classification CSV contains ages outside the expected class list.") from exc
    counts = confusion_matrix(y_true, y_pred, labels=range(len(ages) + unassigned))[:len(ages)]
    totals = counts.sum(axis=1, keepdims=True)
    pct = np.divide(counts * 100.0, totals, out=np.zeros_like(counts, dtype=float), where=totals != 0)
    header = ["actual_age_days", *[f"predicted_day_{a:g}" for a in ages]]
    tick_labels = [f"{a:g}" for a in ages]
    if unassigned:
        header.append("unassigned")
        tick_labels.append("Unassigned")
    _csv(out / "confusion_counts.csv", header, ([a, *row] for a, row in zip(ages, counts)))
    _csv(out / "confusion_row_percent.csv", header, ([a, *row] for a, row in zip(ages, pct)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(ages)), zero_division=0
    )
    _csv(out / "classification_by_age.csv", ["age_days", f"heldout_{units}", f"correct_{units}", "precision", "recall", "f1"],
         zip(ages, support, np.diag(counts), precision, recall, f1))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8), layout="constrained")
    correct = np.trace(counts)
    fig.suptitle(f"{title}\nHeld-out {units} | {correct}/{len(actual)} correct ({correct / len(actual):.1%})")
    for ax, matrix, label, limit, fmt in zip(
        axes, [counts, pct], ["Mosquito counts" if mosquito else "Spectrum counts", "Percent within actual age"],
        [max(1, counts.max()), 100], ["d", ".1f"],
    ):
        im = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=limit)
        ax.set_title(label)
        ax.set_xticks(range(len(tick_labels)), tick_labels, rotation=30 if unassigned else 0)
        ax.set_yticks(range(len(ages)), [f"{a:g}" for a in ages])
        ax.set_xlabel("Predicted age (days)")
        ax.set_ylabel("Actual age (days)")
        for (r, c), value in np.ndenumerate(matrix):
            ax.text(c, r, format(value, fmt), ha="center", va="center", fontsize=8,
                    color="white" if value > limit * .55 else "#162534")
        fig.colorbar(im, ax=ax, shrink=.78)
    _save(fig, out / "confusion_matrix", plt)


def _regression(out, title, ages, actual, predicted, plt, *, mosquito_ids=None):
    units = "spectra" if mosquito_ids is None else "mosquitoes"
    residual = predicted - actual
    rows = []
    for age in ages:
        error = residual[actual == age]
        if error.size:
            rows.append((age, len(error), np.mean(abs(error)), np.sqrt(np.mean(error ** 2)), np.mean(error)))
    _csv(out / "regression_by_age.csv", ["age_days", f"heldout_{units}", "mae_days", "rmse_days", "mean_error_days"], rows)
    key = "holdout_row_zero_based" if mosquito_ids is None else "mosquito_id"
    identifiers = range(len(actual)) if mosquito_ids is None else mosquito_ids
    _csv(out / "regression_residuals.csv", [key, "actual_age_days", "predicted_age_days", "residual_days"],
         zip(identifiers, actual, predicted, residual))
    lo = min(actual.min(), predicted.min())
    hi = max(actual.max(), predicted.max())
    pad = max((hi - lo) * .04, .25)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.3), layout="constrained")
    fig.suptitle(f"{title}\nHeld-out {units} | RMSE {np.sqrt(np.mean(residual ** 2)):.2f} days")
    axes[0].plot([lo - pad, hi + pad], [lo - pad, hi + pad], "--", color="#52606d", lw=1, label="Perfect prediction")
    axes[0].scatter(actual, predicted, s=13, alpha=.46, color="#276e9e", edgecolors="none")
    axes[0].set(xlabel="Actual age (days)", ylabel="Predicted age (days)", xlim=(lo - pad, hi + pad), ylim=(lo - pad, hi + pad))
    axes[0].legend(loc="upper left", frameon=False, fontsize=8)
    axes[1].scatter(actual, residual, s=13, alpha=.46, color="#a04539", edgecolors="none")
    axes[1].axhline(0, color="#52606d", ls="--", lw=1)
    axes[1].set(xlabel="Actual age (days)", ylabel="Predicted − actual (days)")
    _save(fig, out / "regression_diagnostics", plt)


def _embedding(out, title, actual, predicted, features, classification, seed, plt):
    from sklearn.manifold import TSNE
    from threadpoolctl import threadpool_limits

    if features.ndim != 2 or features.shape[0] != len(actual) or not np.isfinite(features).all():
        raise ValueError("ANN feature rows must match the saved held-out predictions.")
    if len(features) < 3 or np.all(features == features[0]):
        return {"status": "skipped", "reason": "Need at least three spectra and varying ANN features."}
    perplexity = min(30, max(2, (len(features) - 1) // 3))
    with threadpool_limits(limits=1):
        points = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto",
                      random_state=seed, n_jobs=1).fit_transform(features)
    if not np.isfinite(points).all():
        raise RuntimeError("t-SNE produced nonfinite coordinates.")
    _csv(out / "tsne_coordinates.csv", ["holdout_row_zero_based", "true_age_days", "predicted_age_days", "tsne_1", "tsne_2"],
         ((i, a, p, x, y) for i, (a, p, (x, y)) in enumerate(zip(actual, predicted, points))))
    np.savez_compressed(out / "ann_hidden_features.npz", features=features, true_age_days=actual,
                        predicted_age_days=predicted, holdout_row_zero_based=np.arange(len(actual)))
    fig, ax = plt.subplots(figsize=(6.8, 4.6), layout="constrained")
    fig.suptitle(f"{title}\nt-SNE of ANN hidden features | held-out spectra")
    palette = plt.get_cmap("tab10")
    for i, age in enumerate(np.unique(actual)):
        keep = actual == age
        ax.scatter(*points[keep].T, s=20, color=palette(i % 10), alpha=.8, edgecolors="none", label=f"Day {age:g}")
    if classification and np.any(actual != predicted):
        ax.scatter(*points[actual != predicted].T, s=42, facecolors="none", edgecolors="#222222", linewidths=.65,
                   label="Incorrect prediction")
    ax.set(xlabel="t-SNE 1", ylabel="t-SNE 2")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=8, title="Actual age")
    _save(fig, out / "ann_tsne", plt)
    return {"status": "complete", "n_spectra": len(actual), "perplexity": perplexity, "random_seed": seed,
            "features": "Last hidden layer after ReLU", "point_unit": "held-out spectrum"}


def write_diagnostics(out, config, ages, actual, predicted, *, features=None, history=None, mosquito=None):
    """Write CSVs and plots without changing the model or held-out predictions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    actual, predicted = np.asarray(actual), np.asarray(predicted)
    ages = np.asarray(ages)
    if (actual.ndim != 1 or predicted.shape != actual.shape or len(actual) == 0
            or not np.isfinite(actual).all() or not np.isfinite(predicted).all()):
        raise ValueError("Diagnostics require matching, finite one-dimensional predictions and ages.")
    if config.get("n_holdout_spectra", len(actual)) != len(actual):
        raise ValueError("The prediction count does not match the run's held-out spectrum count.")
    out.mkdir(parents=True, exist_ok=False)
    dataset = DATASET_NAMES.get(config["dataset"], config["dataset"])
    model = MODEL_NAMES.get(config.get("model", "ann"), config.get("model", "ann"))
    title = f"{dataset} | split {config['split']} | {model}"
    if config.get("synthetic_demo"):
        title = f"Synthetic demo | split {config['split']} | {model}"
    classification = config["task"] == "classification"
    if classification:
        _classification(out, title, ages, actual, predicted, plt)
    else:
        _regression(out, title, ages, actual, predicted, plt)
    if history is not None and len(history):
        epoch, loss = np.asarray(history).T
        fig, ax = plt.subplots(figsize=(6.3, 3.7), layout="constrained")
        ax.plot(epoch, loss, lw=1.3, color="#276e9e")
        ax.set(xlabel="Epoch", ylabel="Training cross-entropy" if classification else "Training MSE (days²)",
               title=f"{title}\nTraining loss")
        _save(fig, out / "training_loss", plt)
    result = {"dataset": config["dataset"], "split": config["split"], "task": config["task"],
              "model": config.get("model", "ann"), "evaluation_level": "spectrum", "n_holdout_spectra": len(actual),
              "input_sha256": config.get("input_sha256"), "synthetic_demo": config.get("synthetic_demo", False),
              "tsne": {"status": "not requested"}}
    if mosquito is not None:
        rows, metrics = mosquito
        grouped_dir = out / "mosquito"
        grouped_dir.mkdir()
        true_ages = np.asarray([row.true_age_days for row in rows])
        estimates = np.asarray([row.predicted_age_days for row in rows])
        if classification:
            _classification(grouped_dir, title, ages, true_ages, estimates, plt, mosquito=True)
        else:
            _regression(grouped_dir, title, ages, true_ages, estimates, plt,
                        mosquito_ids=[row.mosquito_id for row in rows])
        result.update(metrics)
    if features is not None:
        result["tsne"] = _embedding(out, title, actual, predicted, features, classification, config.get("seed", 42), plt)
    (out / "diagnostics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def _saved_ann_features(run, data_dir, config, actual, predicted):
    import torch
    from ann_model import build_ann
    from preprocessing import preprocess_partition
    from splitting import load_partition

    source = data_dir / Path(config["input_file"]).name
    if hashlib.sha256(source.read_bytes()).hexdigest() != config["input_sha256"]:
        raise ValueError("The input dataset differs from the one used for the saved model.")
    # Older checkpoints wrote TorchVersion rather than a plain string. Allow just that type.
    allow = getattr(torch.serialization, "safe_globals", None)
    scope = allow([torch.torch_version.TorchVersion]) if allow else nullcontext()
    with scope:
        checkpoint = torch.load(run / "model.pt", map_location="cpu", weights_only=True)
    saved = checkpoint["config"]
    fields = ("dataset", "split", "task", "input_sha256", "spectral_range_cm_1", "normalization",
              "second_derivative", "hidden_layers", "partition", "pca", "partition_indices_sha256",
              "pca_sha256", "partition_indices_file", "pca_file")
    if any(saved.get(k) != config.get(k) for k in fields):
        raise ValueError("Checkpoint settings do not match metrics.json.")
    ages = np.asarray(checkpoint["ages"])
    classification = config["task"] == "classification"
    lower, upper = config.get("spectral_range_cm_1") or (None, None)
    der = config.get("second_derivative") or {}
    source_rows = None
    if config.get("partition") is not None:
        indices_file = run / config["partition_indices_file"]
        expected_hash = config.get("partition_indices_sha256")
        if expected_hash is None or hashlib.sha256(indices_file.read_bytes()).hexdigest() != expected_hash:
            raise ValueError("The saved partition indices do not match their recorded hash.")
        with indices_file.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        recorded = [(row["partition"], int(row["source_row_zero_based"])) for row in rows]
        if any(membership not in ("train", "holdout") for membership, _ in recorded):
            raise ValueError("The saved partition indices contain an invalid partition name.")
        source_rows = tuple(np.asarray([i for membership, i in recorded if membership == part], dtype=int)
                            for part in ("train", "holdout"))
    partition_settings = config.get("partition") or {}
    method = {"ordered_by_age": "ordered", "stratified_shuffle": "shuffled",
              "saved_partition": "ordered"}.get(partition_settings.get("method"), "ordered")
    arrays, partition = load_partition(source, split=config["split"], method=method,
                                       seed=partition_settings.get("partition_seed"), source_rows=source_rows)
    if config.get("partition") is not None:
        if partition != config["partition"]:
            raise ValueError("The reconstructed partition differs from the saved run.")
        expected = [("train", int(i)) for i in arrays["source_row_train"]]
        expected.extend(("holdout", int(i)) for i in arrays["source_row_test"])
        if recorded != expected:
            raise ValueError("The reconstructed source row indices differ from the saved run.")
    _, _, x, labels, axis = preprocess_partition(
        arrays, ages=ages, lower=lower, upper=upper,
        normalize=config["normalization"] == "SNV per spectrum", second_derivative=bool(der),
        segment=der.get("segment", 9), gap=der.get("gap", 9), require_all_training_ages=classification,
    )
    np.testing.assert_array_equal(axis, checkpoint["wavenumbers"])
    np.testing.assert_array_equal(ages[labels], actual)
    if config.get("pca") is not None:
        import joblib
        pca_file = run / config["pca_file"]
        expected_hash = config.get("pca_sha256")
        if expected_hash is None or hashlib.sha256(pca_file.read_bytes()).hexdigest() != expected_hash:
            raise ValueError("The saved PCA transform does not match its recorded hash.")
        pca = joblib.load(pca_file)
        if (pca.n_components_ != config["pca"]["n_components"]
                or pca.whiten != config["pca"]["whiten"]
                or pca.svd_solver != config["pca"]["svd_solver"]):
            raise ValueError("The saved PCA transform differs from the run settings.")
        x = pca.transform(x).astype(np.float32)
    model = build_ann(x.shape[1], len(ages) if classification else 1, tuple(saved["hidden_layers"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        features = model[:-1](torch.from_numpy(x))
        scores = model[-1](features).numpy()
    if classification:
        np.testing.assert_array_equal(ages[scores.argmax(axis=1)], predicted)
    else:
        np.testing.assert_allclose(scores[:, 0], predicted, rtol=1e-5, atol=5e-5)
    return ages, features.numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Folder containing metrics.json and predictions.csv.")
    parser.add_argument("--data-dir", type=Path,
                        help="Original full-dataset or legacy split folder; required to rebuild ANN t-SNE.")
    parser.add_argument("--output-dir", type=Path, help="Defaults to a new diagnostics subfolder of the run.")
    parser.add_argument("--tsne", action=argparse.BooleanOptionalAction, default=True,
                        help="Build t-SNE for ANN runs; use --no-tsne for prediction-only reports.")
    args = parser.parse_args()
    run = args.run_dir
    config = json.loads((run / "metrics.json").read_text())
    with (run / "predictions.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if [int(row["holdout_row_zero_based"]) for row in rows] != list(range(len(rows))):
        raise ValueError("Predictions must remain in their original held-out row order.")
    actual = np.array([float(row["true_age_days"]) for row in rows])
    predicted = np.array([float(row["predicted_age_days"]) for row in rows])
    classification = config["task"] == "classification"
    ages = np.asarray(config.get("ages_days", np.union1d(actual, predicted) if classification else np.unique(actual)))
    features = None
    if config.get("model", "ann") == "ann" and args.tsne:
        if args.data_dir is None:
            parser.error("--data-dir is required for ANN t-SNE; use --no-tsne to report predictions only.")
        ages, features = _saved_ann_features(run, args.data_dir, config, actual, predicted)
    history = None
    if (run / "training_loss.csv").is_file():
        with (run / "training_loss.csv").open(newline="", encoding="utf-8") as file:
            history = [(int(row["epoch"]), float(row["training_loss"])) for row in csv.DictReader(file)]
    out = args.output_dir or run / "diagnostics"
    mosquito = None
    if rows and "mosquito_id" in rows[0]:
        mosquito = score_mosquitoes([row["mosquito_id"] for row in rows], actual, predicted, classification)
    result = write_diagnostics(out, config, ages, actual, predicted, features=features, history=history,
                               mosquito=mosquito)
    counts = f"{len(actual)} held-out spectra"
    if mosquito is not None:
        write_mosquito_predictions(out / "mosquito_predictions.csv", mosquito[0])
        counts += f" and {result['n_holdout_mosquitoes']} mosquitoes"
    print(f"Saved {result['model']} diagnostics for {counts}: {out}")


if __name__ == "__main__":
    main()
