"""Partition one spectral dataset and compare ANN and traditional models."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from preprocessing import preprocess_partition, read_saved_bounds, training_pca
from partition_plan import load_task_partition
from datasets import PRESET_AGES, check_mosquito_ids, discover_splits, read_partition_metadata
from evaluation import (regression_metrics, score_mosquitoes,
                        write_mosquito_predictions, write_regression_metrics)


RANGES = {"classification": (650, 1850), "regression": (650, 1850)}
MODELS = {"classification": ("ann", "svm", "logistic"), "regression": ("ann", "svr", "pls")}
SUMMARY_FIELDS = ["dataset", "split", "task", "model", "n_training_spectra", "n_holdout_spectra",
                  "correct_spectra", "accuracy", "rmse_days", "mae_days", "pearson_r",
                  "input_sha256", "evaluation_level", "mosquito_ids_checked", "synthetic_demo",
                  "n_holdout_mosquitoes", "correct_mosquitoes", "n_unassigned_mosquitoes", "mosquito_accuracy",
                  "mosquito_rmse_days", "mosquito_mae_days", "mosquito_pearson_r"]


def split_argument(value):
    if value == "all":
        return value
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Choose a positive split number or all.") from exc
    if number < 1:
        raise argparse.ArgumentTypeError("Choose a positive split number or all.")
    return number


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Folder of full NPZ datasets or legacy saved splits (default: data).")
    parser.add_argument("--output-dir", type=Path, help="New run folder (default: a timestamped folder under runs/).")
    parser.add_argument("--dataset", help="Dataset filename stem, such as culex or my-study.")
    parser.add_argument("--list-datasets", action="store_true", help="List datasets and split numbers found in --data-dir.")
    parser.add_argument("--ages", type=float, nargs="+", metavar="DAYS",
                        help="For custom NPZ files without ages_days: list ages in zero-based label order.")
    parser.add_argument("--split", type=split_argument, nargs="+", default=[1], metavar="SPLIT",
                        help="Split/repeat numbers 1, 2, 3, or all. Default: 1.")
    parser.add_argument("--split-method", choices=["ordered", "shuffled"],
                        help="Full datasets: default original ordered partitions for classification, selected partitions for study regression. Legacy splits remain as saved.")
    parser.add_argument("--split-seed", type=int,
                        help="Optional new partition seed; repeat n uses seed+n-1. Overrides recorded regression memberships.")
    parser.add_argument("--reshuffle", action="store_true",
                        help="Draw new random age-stratified partitions instead of reusing selected study regression memberships.")
    parser.add_argument("--task", choices=RANGES, default="classification")
    parser.add_argument("--model", choices=["ann", "svm", "logistic", "svr", "pls", "all"], default="ann",
                        help="Choose a model, or all models compatible with the selected task.")
    parser.add_argument("--show-range", action="store_true", help="Print the measured range without training.")
    crop = parser.add_mutually_exclusive_group()
    crop.add_argument("--range", dest="wavenumber_range", type=float, nargs=2, metavar=("LOW", "HIGH"),
                      help="Choose any inclusive range inside the measured interval in cm⁻¹.")
    crop.add_argument("--full-range", action="store_true", help="Use the full saved wavenumber range.")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True,
                        help="Apply per-spectrum SNV (default: on).")
    parser.add_argument("--second-derivative", action=argparse.BooleanOptionalAction, default=False,
                        help="Apply the Norris second derivative after range selection and optional SNV (default: off).")
    parser.add_argument("--norris-segment", type=int, default=9)
    parser.add_argument("--norris-gap", type=int, default=9)
    parser.add_argument("--pca-components", type=int, help="Optional training-only PCA, e.g. 100; off by default.")
    parser.add_argument("--epochs", type=int, default=10000, help="ANN only; default 10000.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="ANN only.")
    parser.add_argument("--seed", type=int, default=42,
                        help="ANN initialization seed only; independent of --split-seed.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="ANN only.")
    parser.add_argument("--diagnostics", action=argparse.BooleanOptionalAction, default=True,
                        help="Save per-age tables and figures (default: on).")
    parser.add_argument("--tsne", action=argparse.BooleanOptionalAction, default=True,
                        help="Include ANN t-SNE in diagnostics (default: on).")
    args = parser.parse_args()
    if "all" in args.split and args.split != ["all"]:
        parser.error("Use --split all by itself, or select split numbers.")
    if not args.list_datasets and args.dataset is None:
        parser.error("Choose --dataset NAME, or use --list-datasets to see available datasets.")
    if args.model != "all" and args.model not in MODELS[args.task]:
        parser.error(f"--model {args.model} is not available for {args.task}.")
    if args.epochs < 1 or not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Epochs and learning rate must be positive.")
    if args.wavenumber_range and (not np.isfinite(args.wavenumber_range).all()
                                  or args.wavenumber_range[0] >= args.wavenumber_range[1]):
        parser.error("Use finite range endpoints with LOW < HIGH.")
    if args.norris_segment < 1 or args.norris_segment % 2 != 1 or args.norris_gap < 0:
        parser.error("Norris segment must be positive and odd; gap must be nonnegative.")
    if args.split_seed is not None and not 0 <= args.split_seed < 2**32:
        parser.error("Split seed must be between 0 and 2**32 - 1.")
    if args.pca_components is not None and args.pca_components < 1:
        parser.error("PCA components must be positive.")
    return args


def fit_ann(args, x_train, target, x_eval, n_classes):
    import torch
    from torch import nn
    from ann_model import build_ann

    classification = args.task == "classification"
    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    x = torch.from_numpy(x_train).to(device)
    y = torch.as_tensor(target, dtype=torch.long if classification else torch.float32, device=device)
    model = build_ann(x.shape[1], n_classes if classification else 1).to(device)
    criterion = nn.CrossEntropyLoss() if classification else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    history = []
    model.train()
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output if classification else output[:, 0], y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite training loss at epoch {epoch}.")
        loss.backward()
        optimizer.step()
        history.append((epoch, loss.item()))
        if epoch == 1 or epoch % 100 == 0 or epoch == args.epochs:
            print(f"Epoch {epoch}: training loss {loss.item():.6g}", flush=True)
    # Score once at the prespecified final epoch; do not tune on the hold-out.
    model.eval()
    features = None
    with torch.no_grad():
        evaluation = torch.from_numpy(x_eval).to(device)
        if args.diagnostics and args.tsne:
            latent = model[:-1](evaluation)
            out = model[-1](latent).cpu().numpy()
            features = latent.cpu().numpy()
        else:
            out = model(evaluation).cpu().numpy()
    predicted = out.argmax(axis=1) if classification else out[:, 0]
    state = {key: value.cpu() for key, value in model.state_dict().items()}
    settings = dict(seed=args.seed, epochs=args.epochs, learning_rate=args.learning_rate,
                    hidden_layers=[100, 100], device=device, torch_version=str(torch.__version__))
    return predicted, state, history, settings, features


def fit_traditional(name, x_train, target, x_eval):
    import sklearn
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC, SVR

    if name == "svm":
        params = dict(kernel="rbf", C=1.0, gamma="scale")
        model = make_pipeline(StandardScaler(), SVC(**params))
    elif name == "logistic":
        params = dict(C=1.0, solver="lbfgs", max_iter=10000)
        model = make_pipeline(StandardScaler(), LogisticRegression(**params))
    elif name == "svr":
        params = dict(kernel="rbf", C=1.0, gamma="scale", epsilon=0.1)
        model = make_pipeline(StandardScaler(), SVR(**params))
    else:
        params = dict(n_components=min(2, x_train.shape[1], len(x_train) - 1), scale=True)
        model = PLSRegression(**params)
    # All feature scaling and fitting use training spectra only.
    model.fit(x_train, target)
    settings = dict(model_parameters=params, device="cpu", sklearn_version=sklearn.__version__,
                    feature_scaling="PLS internal scaling" if name == "pls" else "training StandardScaler")
    return np.asarray(model.predict(x_eval)).reshape(-1), model, None, settings, None


def save_run(dest, name, config, metrics, actual, predicted, model, history, ages, axis, ids=None, source_rows=None, pca=None):
    dest.mkdir(parents=True, exist_ok=False)
    if source_rows is not None:
        index_path = dest / "partition_indices.csv"
        with index_path.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["partition", "source_row_zero_based"])
            for membership, rows in zip(("train", "holdout"), source_rows):
                writer.writerows((membership, int(row)) for row in rows)
        config["partition_indices_file"] = index_path.name
        config["partition_indices_sha256"] = hashlib.sha256(index_path.read_bytes()).hexdigest()
    if pca is not None:
        import joblib
        pca_path = dest / "pca.joblib"
        joblib.dump(pca, pca_path)
        config["pca_file"] = pca_path.name
        config["pca_sha256"] = hashlib.sha256(pca_path.read_bytes()).hexdigest()
    (dest / "metrics.json").write_text(json.dumps({**config, **metrics}, indent=2) + "\n")
    if config["task"] == "regression":
        write_regression_metrics(dest / "regression_metrics.csv", metrics, len(actual))
    with (dest / "predictions.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        header = ["holdout_row_zero_based", "true_age_days", "predicted_age_days"]
        writer.writerow(header + (["mosquito_id"] if ids is not None else []))
        for i, (a, p) in enumerate(zip(actual, predicted)):
            writer.writerow([i, a.item(), p.item()] + ([ids[i]] if ids is not None else []))
    if name == "ann":
        import torch
        with (dest / "training_loss.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["epoch", "training_loss"])
            writer.writerows(history)
        torch.save({"state_dict": model, "config": config, "ages": ages.tolist(), "wavenumbers": axis.tolist()},
                   dest / "model.pt")
    else:
        import joblib
        joblib.dump({"estimator": model, "config": config, "ages": ages.tolist(), "wavenumbers": axis.tolist()},
                    dest / "model.joblib")


def main():
    args = parse_args()
    try:
        found = discover_splits(args.data_dir)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.list_datasets:
        if not found:
            print(f"No <dataset>.npz or <dataset>_fold_<index>.npz files found in {args.data_dir}. Try python quickstart.py.")
        for name, paths in found.items():
            print(f"{name}: splits {', '.join(map(str, sorted(paths)))}")
        return
    available = found.get(args.dataset)
    if not available:
        raise SystemExit(f"Dataset {args.dataset!r} not found in {args.data_dir}. Use --list-datasets to check filenames.")
    splits = sorted(available) if args.split == ["all"] else list(dict.fromkeys(args.split))
    missing = [split for split in splits if split not in available]
    if missing:
        raise SystemExit(f"Requested splits {missing} not found for {args.dataset}. Available: {sorted(available)}.")
    sources = {s: available[s] for s in splits}
    if args.show_range:
        for split, source in sources.items():
            lo, hi = read_saved_bounds(source)
            print(f"{args.dataset}, split {split}: {lo} to {hi} cm⁻¹")
        return
    if args.output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        args.output_dir = Path("runs") / f"{args.dataset}_{args.task}_{stamp}"
    if args.output_dir.exists():
        raise FileExistsError(f"Choose a new output directory: {args.output_dir}")
    print(f"Output: {args.output_dir}", flush=True)
    names = MODELS[args.task] if args.model == "all" else (args.model,)
    multiple = len(splits) * len(names) > 1
    default_range = RANGES[args.task] if args.dataset in PRESET_AGES else (None, None)
    lower, upper = (None, None) if args.full_range else args.wavenumber_range or default_range
    classification = args.task == "classification"
    records = []
    for split, source in sources.items():
        try:
            method = args.split_method or ("ordered" if classification else "shuffled")
            effective_seed = args.split_seed + split - 1 if args.split_seed is not None else None
            arrays, partition, selected_plan = load_task_partition(
                source, args.dataset, split, args.task, method=method,
                seed=effective_seed, reshuffle=args.reshuffle)
            ages, ids, synthetic = read_partition_metadata(arrays, args.dataset, args.ages)
            x_train, y_train, x_eval, y_eval, axis = preprocess_partition(
                arrays, ages=ages, lower=lower, upper=upper, normalize=args.normalize,
                second_derivative=args.second_derivative, segment=args.norris_segment, gap=args.norris_gap,
                require_all_training_ages=classification,
            )
            ids = check_mosquito_ids(ids, y_train, y_eval)
            x_train, x_eval, pca = training_pca(x_train, x_eval, args.pca_components)
        except ValueError as exc:
            raise SystemExit(f"{source.name}: {exc}") from exc
        actual = ages[y_eval]
        target = y_train if classification else ages[y_train]
        config = dict(dataset=args.dataset, split=split, task=args.task, input_file=source.name,
                      input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      spectral_range_cm_1=None if lower is None else [lower, upper],
                      effective_range_cm_1=[float(axis.min()), float(axis.max())],
                      normalization="SNV per spectrum" if args.normalize else "none",
                      second_derivative=(dict(method="Norris", segment=args.norris_segment, gap=args.norris_gap)
                                         if args.second_derivative else None),
                      n_training_spectra=len(x_train), n_holdout_spectra=len(x_eval),
                      partition=partition, partition_plan=selected_plan, model_input_features=x_train.shape[1],
                      pca=(dict(n_components=args.pca_components, svd_solver="full", whiten=False,
                                fitted_on="training spectra only") if pca is not None else None),
                      evaluation_level="spectrum", numpy_version=np.__version__, ages_days=ages.tolist(),
                      mosquito_ids_checked=ids is not None, synthetic_demo=synthetic)
        for name in names:
            print(f"{args.dataset} split {split}: {name} ({args.task})", flush=True)
            if name == "ann":
                pred, model, history, settings, features = fit_ann(args, x_train, target, x_eval, len(ages))
            else:
                pred, model, history, settings, features = fit_traditional(name, x_train, target, x_eval)
            if classification:
                predicted = ages[pred.astype(int)]
                correct = int(np.sum(predicted == actual))
                metrics = {"correct_spectra": correct, "accuracy": correct / len(actual)}
            else:
                predicted = pred.astype(float)
                metrics = regression_metrics(actual, predicted)
            mosquito = None
            if ids is not None:
                mosquito = score_mosquitoes(ids[1], actual, predicted, classification)
                metrics.update(mosquito[1])
            dest = args.output_dir / f"split_{split}_{name}" if multiple else args.output_dir
            run_config = {**config, "model": name, **settings}
            save_run(dest, name, run_config, metrics, actual, predicted, model, history, ages, axis,
                     ids=ids[1] if ids is not None else None,
                     source_rows=(arrays["source_row_train"], arrays["source_row_test"]), pca=pca)
            if mosquito is not None:
                write_mosquito_predictions(dest / "mosquito_predictions.csv", mosquito[0])
            if args.diagnostics:
                from diagnostics import write_diagnostics
                write_diagnostics(dest / "diagnostics", run_config, ages, actual, predicted,
                                  features=features, history=history, mosquito=mosquito)
            records.append({**run_config, **metrics})
            print(json.dumps(metrics), flush=True)
        if multiple:
            with (args.output_dir / "summary.csv").open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(records)


if __name__ == "__main__":
    main()
