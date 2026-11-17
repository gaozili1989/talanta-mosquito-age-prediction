"""Keep selected study regression memberships separate from original classification."""

import hashlib
import json

from splitting import load_partition


def load_task_partition(source, dataset, split, task, *, method=None, seed=None, reshuffle=False):
    """Use recorded regression rows when available; never apply them to classification."""
    method = method or ("ordered" if task == "classification" else "shuffled")
    if reshuffle and method != "shuffled":
        raise ValueError("--reshuffle requires shuffled partitioning.")
    plan_path = source.parent / "regression_partitions.json"
    use_plan = (task == "regression" and method == "shuffled" and seed is None
                and not reshuffle and source.name == f"{dataset}.npz" and plan_path.is_file())
    if not use_plan:
        arrays, info = load_partition(source, split, method=method, seed=seed)
        return arrays, info, None
    plan = json.loads(plan_path.read_text())
    if plan.get("format_version") != 1 or plan.get("task") != "regression":
        raise ValueError("Unsupported or incorrectly labelled regression partition plan.")
    if dataset not in plan.get("datasets", {}):
        raise ValueError(f"No saved regression partition for {dataset}. Use --reshuffle for a new draw.")
    entry = plan["datasets"][dataset]
    if entry["input_file"] != source.name or entry["input_sha256"] != hashlib.sha256(source.read_bytes()).hexdigest():
        raise ValueError("Regression partition plan does not match the source dataset.")
    if str(split) not in entry["splits"]:
        raise ValueError(f"Regression partition plan has no repeat {split}.")
    selected = entry["splits"][str(split)]
    arrays, info = load_partition(source, split, method="shuffled", seed=selected["partition_seed"],
                                  source_rows=(selected["train"], selected["holdout"]))
    return arrays, info, dict(file=plan_path.name, sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                             selected_repeat=split, reused_recorded_rows=True)
