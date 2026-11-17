"""Aggregate grammar-comparison outputs into CSV tables and plots."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt


SIZE_RE = re.compile(
    r"train_(?P<train>\d+)_dev_(?P<dev>\d+)_test_(?P<test>\d+)"
)
SEED_RE = re.compile(r"seed[_-]?(?P<seed>\d+)", re.IGNORECASE)


def parse_thresholds(value: str) -> List[int]:
    thresholds: List[int] = []
    for raw in str(value).split(","):
        raw = raw.strip()
        if not raw:
            continue
        thresholds.append(int(raw))
    return sorted(set(thresholds))


def format_size(value: int) -> str:
    if value >= 1000 and value % 1000 == 0:
        return f"{value // 1000}k"
    return str(value)


def model_label_from_path(path: Path) -> Optional[str]:
    parts = {part.lower() for part in path.parts}
    if "trained_model" in parts:
        return "trained"
    if "random_baseline" in parts:
        return "random"

    text = str(path).lower()
    if "random" in text:
        return "random"
    if "trained" in text or "vilares_grammar_induction/final_layer" in text.replace("\\", "/"):
        return "trained"
    return None


def sizes_from_path(path: Path) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    for part in reversed(path.parts):
        match = SIZE_RE.fullmatch(part)
        if match:
            return (
                int(match.group("train")),
                int(match.group("dev")),
                int(match.group("test")),
            )
    return None, None, None


def seed_from_text(text: object) -> str:
    match = SEED_RE.search(str(text))
    return match.group("seed") if match else ""


def first_seed_from_values(values: Iterable[object]) -> str:
    for value in values:
        if value in {None, ""}:
            continue
        seed = seed_from_text(value)
        if seed:
            return seed
    return ""


def tree_path_values(summary: Dict[str, object], result: Optional[Dict[str, object]]) -> List[object]:
    values: List[object] = []
    split_info = result.get("split_info", {}) if isinstance(result, dict) else {}
    if isinstance(split_info, dict):
        values.extend(
            split_info.get(key)
            for key in ("train_trees_path", "dev_trees_path", "test_trees_path")
        )
    config = summary.get("config", {}) if isinstance(summary, dict) else {}
    if isinstance(config, dict):
        values.extend(
            config.get(key)
            for key in ("train_trees_path", "dev_trees_path", "test_trees_path")
        )
    return values


def infer_model_seed(path: Path, summary: Dict[str, object], result: Optional[Dict[str, object]]) -> str:
    for key in ("model_init_seed", "model_seed"):
        value = summary.get(key) if isinstance(summary, dict) else None
        if value not in {None, ""}:
            return str(value)
    if isinstance(result, dict):
        for key in ("model_init_seed", "model_seed"):
            value = result.get(key)
            if value not in {None, ""}:
                return str(value)

    path_candidates: List[object] = []
    if isinstance(result, dict):
        path_candidates.append(result.get("model_path"))
    if isinstance(summary, dict):
        path_candidates.extend(
            summary.get(key)
            for key in ("model_path", "model_root", "output_root")
        )
    path_candidates.append(path)
    return first_seed_from_values(path_candidates)


def infer_data_seed(summary: Dict[str, object], result: Optional[Dict[str, object]]) -> str:
    for key in ("data_seed",):
        value = summary.get(key) if isinstance(summary, dict) else None
        if value not in {None, ""}:
            return str(value)
    return first_seed_from_values(tree_path_values(summary, result))


def infer_probe_seed(summary: Dict[str, object], result: Optional[Dict[str, object]]) -> str:
    for key in ("seed", "probe_seed"):
        value = summary.get(key) if isinstance(summary, dict) else None
        if value not in {None, ""}:
            return str(value)
    if isinstance(result, dict):
        probe = result.get("probe", {})
        if isinstance(probe, dict):
            value = probe.get("seed")
            if value not in {None, ""}:
                return str(value)
    return ""


def find_nearest_summary(path: Path) -> Optional[Path]:
    for parent in [path.parent, *path.parents]:
        for pattern in (
            "individual_layer_*.json",
            "layers_summary.json",
            "checkpoints_layer_*.json",
        ):
            matches = sorted(parent.glob(pattern))
            if matches:
                return matches[0]
    return None


def metadata_from_summary(summary_path: Optional[Path]) -> Dict[str, object]:
    if summary_path is None or not summary_path.exists():
        return {}
    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def threshold_row(report: Dict[str, object], min_count: int) -> Dict[str, object]:
    for row in report.get("thresholds", []):
        if int(row.get("min_count", -1)) == int(min_count):
            return row
    if int(report.get("min_count", 1)) == int(min_count):
        return report
    raise KeyError(f"No threshold row for min_count={min_count}.")


def as_float(value: object, default: Optional[float] = None) -> Optional[float]:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: object, default: Optional[int] = None) -> Optional[int]:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_divide(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def f1_from_pr(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)


def rounded(value: object, digits: int = 3) -> str:
    number = as_float(value)
    return "" if number is None else f"{number:.{digits}f}"


def percent_rounded(value: object, digits: int = 1) -> str:
    number = as_float(value)
    return "" if number is None else f"{100.0 * number:.{digits}f}"


def result_matches_path(result: Dict[str, object], path: Path) -> bool:
    artifacts = result.get("artifacts", {}) if isinstance(result.get("artifacts", {}), dict) else {}
    resolved_path = path.resolve()
    supergrammar_path = artifacts.get("supergrammar_check_path")
    if supergrammar_path and resolved_path == Path(str(supergrammar_path)).resolve():
        return True
    run_dir = artifacts.get("run_dir")
    if run_dir:
        resolved_run_dir = Path(str(run_dir)).resolve()
        if resolved_path == resolved_run_dir or resolved_run_dir in resolved_path.parents:
            return True
    return False


def result_from_summary(summary: Dict[str, object], path: Path) -> Optional[Dict[str, object]]:
    result = summary.get("result") if isinstance(summary, dict) else None
    if isinstance(result, dict):
        return result

    results = summary.get("results") if isinstance(summary, dict) else None
    if not isinstance(results, list):
        return None

    for candidate in results:
        if isinstance(candidate, dict) and result_matches_path(candidate, path):
            return candidate

    path_name = path.parent.name
    layer_match = re.search(r"_layer_(\d+)$", path_name)
    path_layer = int(layer_match.group(1)) if layer_match else None
    path_checkpoint = path_name[: layer_match.start()] if layer_match else None
    for candidate in results:
        if not isinstance(candidate, dict):
            continue
        candidate_layer = as_int(candidate.get("layer"))
        candidate_checkpoint = str(candidate.get("checkpoint_label", "final"))
        if path_layer is not None and candidate_layer == path_layer:
            if path_checkpoint is None or candidate_checkpoint == path_checkpoint:
                return candidate
    return results[0] if results and isinstance(results[0], dict) else None


def checkpoint_sort_key(name: object) -> Tuple[int, str]:
    text = str(name)
    if text == "checkpoint-initial":
        return (-1, text)
    if text == "final":
        return (10**12, text)
    match = re.search(r"checkpoint-(\d+)$", text)
    if match:
        return (int(match.group(1)), text)
    return (10**11, text)


def parse_checkpoint_step(name: object) -> Optional[int]:
    text = str(name)
    if text == "checkpoint-initial":
        return 0
    match = re.search(r"checkpoint-(\d+)$", text)
    if match:
        return int(match.group(1))
    return None


def extract_probe_metrics(summary: Dict[str, object], result: Optional[Dict[str, object]]) -> Dict[str, object]:
    """Extract threshold-independent probe/tree and grammar-distribution metrics.

    The generated CSV is meant to be the single source of truth for LaTeX tables.
    Rule-inventory counts that depend on a min-support threshold are added later
    from supergrammar_check.json; here we read the non-thresholded summary from
    individual_layer_*.json.
    """
    empty: Dict[str, object] = {
        "probe_test_f1": None,
        "probe_test_accuracy": None,
        "probe_test_macro_f1": None,
        "probe_best_dev_f1": None,
        "probe_best_epoch": None,
        "decoded_tree_precision": None,
        "decoded_tree_recall": None,
        "decoded_tree_f1": None,
        "decoded_gold_tree_precision": None,
        "decoded_gold_tree_recall": None,
        "decoded_gold_tree_f1": None,
        "grammar_rule_precision": None,
        "grammar_rule_recall": None,
        "grammar_rule_f1": None,
        "gold_rule_count": None,
        "model_rule_count": None,
        "shared_rule_count": None,
        "gold_only_rule_count": None,
        "model_only_rule_count": None,
        "model_spurious_rule_pct": None,
        "gold_rule_recovery_pct": None,
        "grammar_global_jsd": None,
        "grammar_per_lhs_macro_jsd": None,
        "grammar_per_lhs_weighted_jsd": None,
    }

    if not isinstance(result, dict):
        return empty

    probe = result.get("probe", {}) if isinstance(result.get("probe", {}), dict) else {}
    grammar = {}
    comparisons = result.get("grammar_comparisons", {})
    if isinstance(comparisons, dict):
        selected = comparisons.get("model_vs_test_gold_original")
        if isinstance(selected, dict):
            grammar = selected
    if not grammar:
        selected = result.get("grammar_comparison", {})
        if isinstance(selected, dict):
            grammar = selected
    overlap = grammar.get("rule_overlap", {}) if isinstance(grammar.get("rule_overlap", {}), dict) else {}
    jsd = grammar.get("rule_distribution_jsd", {}) if isinstance(grammar.get("rule_distribution_jsd", {}), dict) else {}
    test_metrics = probe.get("test_metrics", {}) if isinstance(probe.get("test_metrics", {}), dict) else {}
    tree_f1 = probe.get("tree_f1_test", {}) if isinstance(probe.get("tree_f1_test", {}), dict) else {}
    decoded_gold_tree_f1 = probe.get("decoded_gold_tree_f1_test", {})
    if not isinstance(decoded_gold_tree_f1, dict):
        decoded_gold_tree_f1 = {}

    out = dict(empty)
    out.update(
        {
            "probe_test_f1": as_float(test_metrics.get("f1")),
            "probe_test_accuracy": as_float(test_metrics.get("accuracy")),
            "probe_test_macro_f1": as_float(test_metrics.get("macro_f1")),
            "probe_best_dev_f1": as_float(probe.get("best_dev_f1")),
            "probe_best_epoch": as_int(probe.get("best_epoch")),
            "decoded_tree_precision": as_float(tree_f1.get("precision")),
            "decoded_tree_recall": as_float(tree_f1.get("recall")),
            "decoded_tree_f1": as_float(tree_f1.get("f1")),
            "decoded_gold_tree_precision": as_float(decoded_gold_tree_f1.get("precision")),
            "decoded_gold_tree_recall": as_float(decoded_gold_tree_f1.get("recall")),
            "decoded_gold_tree_f1": as_float(decoded_gold_tree_f1.get("f1")),
            "grammar_rule_precision": as_float(overlap.get("precision")),
            "grammar_rule_recall": as_float(overlap.get("recall")),
            "grammar_rule_f1": as_float(overlap.get("f1")),
            "gold_rule_count": as_int(overlap.get("gold_rule_count")),
            "model_rule_count": as_int(overlap.get("model_rule_count")),
            "shared_rule_count": as_int(overlap.get("shared_rule_count")),
            "gold_only_rule_count": as_int(overlap.get("gold_only_rule_count")),
            "model_only_rule_count": as_int(overlap.get("model_only_rule_count")),
            "model_spurious_rule_pct": as_float(overlap.get("model_spurious_rule_pct")),
            "gold_rule_recovery_pct": as_float(overlap.get("gold_rule_recovery_pct")),
            "grammar_global_jsd": as_float(jsd.get("global_jsd")),
            "grammar_per_lhs_macro_jsd": as_float(jsd.get("per_lhs_macro_jsd")),
            "grammar_per_lhs_weighted_jsd": as_float(jsd.get("per_lhs_weighted_jsd")),
        }
    )
    return out


def record_from_report(path: Path, default_train_size: Optional[int]) -> Dict[str, object]:
    report = json.loads(path.read_text(encoding="utf-8"))
    train_size, dev_size, test_size = sizes_from_path(path)
    summary_path = find_nearest_summary(path)
    summary = metadata_from_summary(summary_path)
    result = result_from_summary(summary, path)

    train_size = train_size or summary.get("train_trees") or default_train_size
    dev_size = dev_size or summary.get("dev_trees")
    test_size = test_size or summary.get("fixed_test_size")
    model_label = model_label_from_path(path) or summary.get("model_source")
    if model_label == "trained_model":
        model_label = "trained"
    if model_label == "random_baseline":
        model_label = "random"
    if model_label not in {"trained", "random"}:
        raise ValueError(f"Could not infer trained/random label for {path}. Use --trained-root/--random-root.")

    run_mode = str(summary.get("run_mode", "unknown")) if isinstance(summary, dict) else "unknown"
    layer = as_int(result.get("layer")) if isinstance(result, dict) else None
    checkpoint_label = str(result.get("checkpoint_label", "")) if isinstance(result, dict) else ""
    checkpoint_step = parse_checkpoint_step(checkpoint_label) if checkpoint_label else None
    model_seed = infer_model_seed(path, summary, result)
    data_seed = infer_data_seed(summary, result)
    probe_seed = infer_probe_seed(summary, result)

    metrics = extract_probe_metrics(summary, result)
    return {
        "path": str(path),
        "summary_path": str(summary_path) if summary_path else "",
        "seed": model_seed,
        "model_seed": model_seed,
        "data_seed": data_seed,
        "probe_seed": probe_seed,
        "run_mode": run_mode,
        "model": model_label,
        "train_size": int(train_size) if train_size is not None else -1,
        "dev_size": int(dev_size) if dev_size is not None else -1,
        "test_size": int(test_size) if test_size is not None else -1,
        "layer": int(layer) if layer is not None else -1,
        "checkpoint_label": checkpoint_label,
        "checkpoint_step": int(checkpoint_step) if checkpoint_step is not None else "",
        "report": report,
        **metrics,
    }


def collect_records(
    roots: Sequence[Tuple[Path, Optional[str]]],
    default_train_size: Optional[int],
    final_step: Optional[int] = None,
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for root, forced_model in roots:
        for path in sorted(root.rglob("supergrammar_check.json")):
            record = record_from_report(path, default_train_size)
            if forced_model is not None:
                record["model"] = forced_model
            records.append(record)
    checkpoint_order = {
        label: idx
        for idx, label in enumerate(
            sorted(
                {str(record.get("checkpoint_label")) for record in records if record.get("checkpoint_label")},
                key=checkpoint_sort_key,
            ),
            start=1,
        )
    }
    for record in records:
        label = str(record.get("checkpoint_label", ""))
        record["checkpoint_order"] = checkpoint_order.get(label, "")

    numeric_checkpoint_steps = sorted(
        {
            int(record["checkpoint_step"])
            for record in records
            if record.get("checkpoint_step") not in {"", None}
        }
    )
    if final_step is not None:
        chosen_final_step = int(final_step)
    elif numeric_checkpoint_steps:
        if len(numeric_checkpoint_steps) > 1:
            inferred_stride = numeric_checkpoint_steps[-1] - numeric_checkpoint_steps[-2]
        else:
            inferred_stride = max(1, numeric_checkpoint_steps[-1])
        chosen_final_step = numeric_checkpoint_steps[-1] + max(1, inferred_stride)
    else:
        chosen_final_step = None

    if chosen_final_step is not None:
        for record in records:
            if record.get("checkpoint_label") == "final" and record.get("checkpoint_step") in {"", None}:
                record["checkpoint_step"] = chosen_final_step
            record["final_step_explicit"] = final_step is not None

    unique: Dict[Tuple[str, int, str], Dict[str, object]] = {}
    for record in records:
        key = (str(record["model"]), int(record["train_size"]), str(record["path"]))
        unique[key] = record
    return sorted(
        unique.values(),
        key=lambda row: (
            int(row["train_size"]),
            str(row["model"]),
            str(row.get("seed", "")),
            int(row.get("layer", -1)),
            as_int(row.get("checkpoint_order"), 10**12) or 10**12,
            str(row["path"]),
        ),
    )


def paper_row_from_record(record: Dict[str, object], min_count: int) -> Dict[str, object]:
    """Build one LaTeX-friendly CSV row for one run and one min-support threshold."""
    row = threshold_row(record["report"], min_count)

    gold_rule_types = as_int(row.get("gold_rule_type_count"), as_int(record.get("gold_rule_count"), 0)) or 0
    pred_rule_types = as_int(row.get("model_rule_type_count"), as_int(record.get("model_rule_count"), 0)) or 0
    extra_rule_types = as_int(row.get("model_only_rule_type_count"), as_int(record.get("model_only_rule_count"), 0)) or 0
    gold_hit_types = max(pred_rule_types - extra_rule_types, 0)
    gold_only_types = max(gold_rule_types - gold_hit_types, 0)

    rule_precision = safe_divide(gold_hit_types, pred_rule_types)
    rule_recall = safe_divide(gold_hit_types, gold_rule_types)
    rule_f1 = f1_from_pr(rule_precision, rule_recall)

    in_sg_types = as_int(row.get("model_only_in_supergrammar_rule_type_count"), 0) or 0
    in_sg_occ = as_int(row.get("model_only_in_supergrammar_occurrence_count"), 0) or 0
    extra_occ = as_int(row.get("model_only_occurrence_count"), 0) or 0
    out_sg_types = as_int(row.get("model_only_not_in_supergrammar_rule_type_count"), max(extra_rule_types - in_sg_types, 0)) or 0
    out_sg_occ = as_int(row.get("model_only_not_in_supergrammar_occurrence_count"), max(extra_occ - in_sg_occ, 0)) or 0

    # Since all recovered gold rules are inherited from the supergrammar, total predicted type
    # coverage by the supergrammar is recovered-gold types + model-only-in-supergrammar types.
    pred_in_sg_types = gold_hit_types + in_sg_types
    pred_in_sg_type_pct = safe_divide(pred_in_sg_types, pred_rule_types)

    decoded_tree_f1 = record.get("decoded_tree_f1")
    if decoded_tree_f1 is None:
        decoded_tree_f1 = record.get("probe_test_f1")

    train_size = int(record["train_size"])
    model = str(record["model"])

    out: Dict[str, object] = {
        "model": model,
        "model_label": "Trained" if model == "trained" else "Random",
        "seed": record.get("seed", ""),
        "model_seed": record.get("model_seed", record.get("seed", "")),
        "data_seed": record.get("data_seed", ""),
        "probe_seed": record.get("probe_seed", ""),
        "run_mode": record.get("run_mode", ""),
        "train_size": train_size,
        "train_label": format_size(train_size),
        "dev_size": int(record["dev_size"]),
        "test_size": int(record["test_size"]),
        "layer": record.get("layer", ""),
        "checkpoint_label": record.get("checkpoint_label", ""),
        "checkpoint_step": record.get("checkpoint_step", ""),
        "checkpoint_order": record.get("checkpoint_order", ""),
        "x_axis": record.get("x_axis", ""),
        "x_value": record.get("x_value", ""),
        "min_count": int(min_count),

        # Probe / tree metrics.
        "decoded_tree_f1": decoded_tree_f1,
        "decoded_tree_precision": record.get("decoded_tree_precision"),
        "decoded_tree_recall": record.get("decoded_tree_recall"),
        "probe_test_accuracy": record.get("probe_test_accuracy"),
        "probe_test_macro_f1": record.get("probe_test_macro_f1"),
        "probe_best_dev_f1": record.get("probe_best_dev_f1"),
        "probe_best_epoch": record.get("probe_best_epoch"),
        "decoded_gold_tree_f1": record.get("decoded_gold_tree_f1"),

        # Rule inventory metrics at this support threshold.
        "rule_precision": rule_precision,
        "rule_recall": rule_recall,
        "rule_f1": rule_f1,
        "gold_rule_types": gold_rule_types,
        "gold_hit_types": gold_hit_types,
        "pred_rule_types": pred_rule_types,
        "extra_rule_types": extra_rule_types,
        "gold_only_rule_types": gold_only_types,
        "model_spurious_rule_pct": safe_divide(extra_rule_types, pred_rule_types),

        # Rule distribution metrics from the run summary.
        "global_jsd": record.get("grammar_global_jsd"),
        "weighted_lhs_jsd": record.get("grammar_per_lhs_weighted_jsd"),
        "macro_lhs_jsd": record.get("grammar_per_lhs_macro_jsd"),

        # Supergrammar compatibility for model-only rules.
        "extra_rule_occurrences": extra_occ,
        "extra_in_supergrammar_types": in_sg_types,
        "extra_in_supergrammar_type_pct": safe_divide(in_sg_types, extra_rule_types),
        "extra_in_supergrammar_occurrences": in_sg_occ,
        "extra_in_supergrammar_occurrence_pct": safe_divide(in_sg_occ, extra_occ),
        "extra_not_in_supergrammar_types": out_sg_types,
        "extra_not_in_supergrammar_occurrences": out_sg_occ,

        # Supergrammar coverage over all predicted rule types.
        "pred_in_supergrammar_types": pred_in_sg_types,
        "pred_in_supergrammar_type_pct": pred_in_sg_type_pct,

        "source_json": record.get("path", ""),
        "summary_json": record.get("summary_path", ""),
    }

    # Pre-rounded string columns are convenient for pgfplotstable or direct table generation.
    for key in [
        "decoded_tree_f1", "rule_precision", "rule_recall", "rule_f1",
        "global_jsd", "weighted_lhs_jsd", "decoded_gold_tree_f1",
    ]:
        out[f"{key}_3"] = rounded(out.get(key), 3)
    for key in [
        "extra_in_supergrammar_type_pct", "extra_in_supergrammar_occurrence_pct",
        "pred_in_supergrammar_type_pct", "model_spurious_rule_pct",
    ]:
        out[f"{key}_pct1"] = percent_rounded(out.get(key), 1)

    return out


PAPER_GROUP_COLUMNS = [
    "model",
    "model_label",
    "data_seed",
    "probe_seed",
    "run_mode",
    "train_size",
    "train_label",
    "dev_size",
    "test_size",
    "layer",
    "checkpoint_label",
    "checkpoint_step",
    "checkpoint_order",
    "x_axis",
    "x_value",
    "min_count",
]

PAPER_NUMERIC_METRIC_COLUMNS = [
    "decoded_tree_f1",
    "decoded_tree_precision",
    "decoded_tree_recall",
    "probe_test_accuracy",
    "probe_test_macro_f1",
    "probe_best_dev_f1",
    "probe_best_epoch",
    "decoded_gold_tree_f1",
    "rule_precision",
    "rule_recall",
    "rule_f1",
    "gold_rule_types",
    "gold_hit_types",
    "pred_rule_types",
    "extra_rule_types",
    "gold_only_rule_types",
    "model_spurious_rule_pct",
    "global_jsd",
    "weighted_lhs_jsd",
    "macro_lhs_jsd",
    "extra_rule_occurrences",
    "extra_in_supergrammar_types",
    "extra_in_supergrammar_type_pct",
    "extra_in_supergrammar_occurrences",
    "extra_in_supergrammar_occurrence_pct",
    "extra_not_in_supergrammar_types",
    "extra_not_in_supergrammar_occurrences",
    "pred_in_supergrammar_types",
    "pred_in_supergrammar_type_pct",
]

PAPER_ROUNDED_3_COLUMNS = [
    "decoded_tree_f1",
    "rule_precision",
    "rule_recall",
    "rule_f1",
    "global_jsd",
    "weighted_lhs_jsd",
    "decoded_gold_tree_f1",
]

PAPER_PERCENT_COLUMNS = [
    "extra_in_supergrammar_type_pct",
    "extra_in_supergrammar_occurrence_pct",
    "pred_in_supergrammar_type_pct",
    "model_spurious_rule_pct",
]


def mean_and_std(values: Sequence[float]) -> Tuple[float, float]:
    mean = sum(values) / len(values)
    if len(values) <= 1:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, variance ** 0.5


def model_seed_for_row(row: Dict[str, object]) -> str:
    return str(row.get("model_seed") or row.get("seed") or "unseeded")


def should_aggregate_paper_rows(rows: Sequence[Dict[str, object]]) -> bool:
    return len({model_seed_for_row(row) for row in rows}) > 1


def aggregate_metric_by_seed(rows: Sequence[Dict[str, object]], metric: str) -> Tuple[Optional[float], Optional[float], int]:
    values_by_seed: Dict[str, List[float]] = {}
    for row in rows:
        value = as_float(row.get(metric))
        if value is None:
            continue
        values_by_seed.setdefault(model_seed_for_row(row), []).append(value)
    seed_values = [
        sum(values) / len(values)
        for values in values_by_seed.values()
        if values
    ]
    if not seed_values:
        return None, None, 0
    mean, std = mean_and_std(seed_values)
    return mean, std, len(seed_values)


def aggregate_paper_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[Tuple[object, ...], List[Dict[str, object]]] = {}
    for row in rows:
        key = tuple(row.get(column, "") for column in PAPER_GROUP_COLUMNS)
        grouped.setdefault(key, []).append(row)

    aggregated_rows: List[Dict[str, object]] = []
    for key in sorted(grouped, key=lambda item: tuple(str(value) for value in item)):
        group_rows = grouped[key]
        first = group_rows[0]
        model_seeds = sorted({model_seed_for_row(row) for row in group_rows})

        out: Dict[str, object] = {}
        for column in first:
            if column == "seed":
                out[column] = "mean"
            elif column == "model_seed":
                out[column] = ",".join(model_seeds)
            elif column in PAPER_GROUP_COLUMNS:
                out[column] = first.get(column, "")
            elif column in PAPER_NUMERIC_METRIC_COLUMNS:
                mean, _std, _n = aggregate_metric_by_seed(group_rows, column)
                out[column] = "" if mean is None else mean
            elif column in {"source_json", "summary_json"}:
                out[column] = ""
            elif column.endswith("_3") or column.endswith("_pct1"):
                out[column] = ""
            else:
                out[column] = first.get(column, "")

        out["seed_count"] = len(model_seeds)
        out["model_seeds"] = ",".join(model_seeds)
        out["source_jsons"] = ";".join(sorted({str(row.get("source_json", "")) for row in group_rows if row.get("source_json")}))
        out["summary_jsons"] = ";".join(sorted({str(row.get("summary_json", "")) for row in group_rows if row.get("summary_json")}))

        for column in PAPER_NUMERIC_METRIC_COLUMNS:
            mean, std, seed_count = aggregate_metric_by_seed(group_rows, column)
            out[f"{column}_std"] = "" if std is None else std
            out[f"{column}_seed_count"] = seed_count

        for column in PAPER_ROUNDED_3_COLUMNS:
            out[f"{column}_3"] = rounded(out.get(column), 3)
            out[f"{column}_std_3"] = rounded(out.get(f"{column}_std"), 3)
            mean = as_float(out.get(column))
            std = as_float(out.get(f"{column}_std"))
            out[f"{column}_mean_std_3"] = "" if mean is None or std is None else f"{mean:.3f}+/-{std:.3f}"

        for column in PAPER_PERCENT_COLUMNS:
            out[f"{column}_pct1"] = percent_rounded(out.get(column), 1)
            out[f"{column}_std_pct1"] = percent_rounded(out.get(f"{column}_std"), 1)
            mean = as_float(out.get(column))
            std = as_float(out.get(f"{column}_std"))
            out[f"{column}_mean_std_pct1"] = "" if mean is None or std is None else f"{100.0 * mean:.1f}+/-{100.0 * std:.1f}"

        aggregated_rows.append(out)
    return aggregated_rows


def write_paper_results_csv(records: Sequence[Dict[str, object]], thresholds: Sequence[int], output_path: Path) -> None:
    """Write one comprehensive CSV meant to be used directly by LaTeX/Overleaf."""
    rows = [paper_row_from_record(record, threshold) for record in records for threshold in thresholds]
    if not rows:
        return
    if should_aggregate_paper_rows(rows):
        rows = aggregate_paper_rows(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_aggregate_csv(records: Sequence[Dict[str, object]], thresholds: Sequence[int], output_path: Path) -> None:
    """Backward-compatible aggregate CSV, now with all core table counts included."""
    rows = [paper_row_from_record(record, threshold) for record in records for threshold in thresholds]
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_best_metrics_csv(records: Sequence[Dict[str, object]], thresholds: Sequence[int], output_path: Path) -> None:
    """Compact CSV for quick inspection; paper_results.csv is the main table source."""
    keep = [
        "model", "model_label", "seed", "model_seed", "data_seed", "probe_seed",
        "run_mode", "train_size", "train_label", "dev_size", "test_size",
        "layer", "checkpoint_label", "checkpoint_step", "checkpoint_order", "x_axis", "x_value", "min_count",
        "decoded_tree_f1", "rule_precision", "rule_recall", "rule_f1",
        "gold_rule_types", "gold_hit_types", "pred_rule_types", "extra_rule_types",
        "extra_rule_occurrences", "global_jsd", "weighted_lhs_jsd",
        "extra_in_supergrammar_types", "extra_in_supergrammar_type_pct",
        "extra_in_supergrammar_occurrences", "extra_in_supergrammar_occurrence_pct",
        "pred_in_supergrammar_types", "pred_in_supergrammar_type_pct",
    ]
    rows: List[Dict[str, object]] = []
    for record in records:
        for threshold in thresholds:
            row = paper_row_from_record(record, threshold)
            rows.append({key: row.get(key, "") for key in keep})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keep)
        writer.writeheader()
        writer.writerows(rows)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 160,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "lines.linewidth": 2.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def infer_x_axis(records: Sequence[Dict[str, object]], requested: str) -> str:
    if requested != "auto":
        return requested
    modes = {str(record.get("run_mode", "")) for record in records}
    layers = {int(record.get("layer", -1)) for record in records if int(record.get("layer", -1)) > 0}
    checkpoints = {str(record.get("checkpoint_label", "")) for record in records if record.get("checkpoint_label")}
    train_sizes = {int(record.get("train_size", -1)) for record in records if int(record.get("train_size", -1)) > 0}
    if "layers" in modes or (len(layers) > 1 and len(train_sizes) <= 1):
        return "layer"
    if "checkpoints" in modes or len(checkpoints) > 1:
        return "checkpoint"
    return "train_size"


def x_value_for_record(record: Dict[str, object], x_axis: str) -> Optional[float]:
    if x_axis == "layer":
        value = as_int(record.get("layer"))
    elif x_axis == "checkpoint":
        value = as_int(record.get("checkpoint_step"))
        if value is None:
            value = as_int(record.get("checkpoint_order"))
    else:
        value = as_int(record.get("train_size"))
    return float(value) if value is not None and value >= 0 else None


def x_label_for_axis(x_axis: str) -> str:
    if x_axis == "layer":
        return "Transformer layer"
    if x_axis == "checkpoint":
        return "Training step"
    return "Probe training trees"


def x_ticks_for_records(records: Sequence[Dict[str, object]], x_axis: str) -> Tuple[List[float], List[str]]:
    if x_axis == "checkpoint":
        pairs = sorted(
            {
                (int(record["checkpoint_step"]), str(record["checkpoint_label"]))
                for record in records
                if record.get("checkpoint_step") not in {"", None}
            },
            key=lambda item: item[0],
        )
        return (
            [float(step) for step, _label in pairs],
            [
                "0"
                if step == 0
                else (
                    format_size(step)
                    if label == "final" and any(bool(record.get("final_step_explicit")) for record in records)
                    else ("final" if label == "final" else format_size(step))
                )
                for step, label in pairs
            ],
        )

    values = sorted(
        {
            int(value)
            for value in (x_value_for_record(record, x_axis) for record in records)
            if value is not None and value > 0
        }
    )
    labels = [format_size(value) if x_axis == "train_size" else str(value) for value in values]
    return [float(value) for value in values], labels


def thinned_train_size_ticks(values: Sequence[int]) -> List[int]:
    if len(values) <= 5:
        return list(values)

    keep = {values[0], values[-1]}
    for value in values:
        first_digit = int(str(value)[0])
        if first_digit in {1, 2, 5}:
            keep.add(value)
    return [value for value in values if value in keep]


def display_x_ticks_for_records(records: Sequence[Dict[str, object]], x_axis: str) -> Tuple[List[float], List[str]]:
    if x_axis == "checkpoint":
        steps = sorted(
            {
                int(record["checkpoint_step"])
                for record in records
                if record.get("checkpoint_step") not in {"", None}
            }
        )
        if not steps:
            return x_ticks_for_records(records, x_axis)

        if len(steps) <= 6:
            shown_steps = steps
        else:
            target_count = 6
            indices = {
                round(i * (len(steps) - 1) / (target_count - 1))
                for i in range(target_count)
            }
            shown_steps = [steps[index] for index in sorted(indices)]

        final_steps = {
            int(record["checkpoint_step"])
            for record in records
            if record.get("checkpoint_label") == "final"
            and record.get("checkpoint_step") not in {"", None}
        }
        final_step_explicit = any(bool(record.get("final_step_explicit")) for record in records)
        labels = [
            "0"
            if step == 0
            else (
                format_size(step)
                if final_step_explicit and step in final_steps
                else ("final" if step in final_steps else format_size(step))
            )
            for step in shown_steps
        ]
        return [float(step) for step in shown_steps], labels

    if x_axis != "train_size":
        return x_ticks_for_records(records, x_axis)

    values = sorted(
        {
            int(value)
            for value in (x_value_for_record(record, x_axis) for record in records)
            if value is not None and value > 0
        }
    )
    values = thinned_train_size_ticks(values)
    return [float(value) for value in values], [format_size(value) for value in values]


def x_tick_label_kwargs(x_axis: str, tick_count: int) -> Dict[str, object]:
    if x_axis == "checkpoint":
        return {"rotation": 0, "ha": "center"}
    if x_axis == "train_size" and tick_count >= 5:
        return {"rotation": 30, "ha": "right", "rotation_mode": "anchor"}
    return {"rotation": 0, "ha": "center"}


def output_dir_with_mode_suffix(output_dir: Path, records: Sequence[Dict[str, object]], x_axis: str) -> Path:
    modes = {
        str(record.get("run_mode", ""))
        for record in records
        if str(record.get("run_mode", "")) not in {"", "unknown"}
    }
    if len(modes) == 1:
        mode = next(iter(modes))
    elif len(modes) > 1:
        mode = "mixed"
    elif x_axis in {"layer", "checkpoint"}:
        mode = f"{x_axis}s"
    else:
        mode = "individual"

    if mode == "individual":
        return output_dir
    suffix = f"_{mode}"
    if output_dir.name.endswith(suffix):
        return output_dir
    return output_dir.with_name(f"{output_dir.name}{suffix}")


def validate_data_seed_consistency(
    records: Sequence[Dict[str, object]],
    x_axis: str,
    *,
    allow_mixed_data_seeds: bool = False,
) -> None:
    if allow_mixed_data_seeds:
        return

    data_seeds_by_group: Dict[Tuple[str, float], set[str]] = {}
    for record in records:
        x_value = x_value_for_record(record, x_axis)
        if x_value is None:
            continue
        data_seed = str(record.get("data_seed", ""))
        model_seed = str(record.get("model_seed", record.get("seed", "")))
        if not data_seed or not model_seed:
            continue
        key = (str(record.get("model", "")), x_value)
        data_seeds_by_group.setdefault(key, set()).add(data_seed)

    mixed = {
        key: sorted(data_seeds)
        for key, data_seeds in data_seeds_by_group.items()
        if len(data_seeds) > 1
    }
    if not mixed:
        return

    examples = ", ".join(
        f"model={model} x={x_value:g} data_seeds={data_seeds}"
        for (model, x_value), data_seeds in sorted(mixed.items())[:5]
    )
    raise ValueError(
        "Refusing to aggregate plot lines across different data seeds. "
        f"Examples: {examples}. Use --allow-mixed-data-seeds to override."
    )


def values_for(
    records: Sequence[Dict[str, object]],
    model: str,
    threshold: int,
    metric: str,
    x_axis: str,
) -> Tuple[List[float], List[float], List[float], List[int]]:
    points_by_x_and_seed: Dict[float, Dict[str, List[float]]] = {}
    for record in records:
        if record["model"] != model:
            continue
        row = threshold_row(record["report"], threshold)
        value = record.get(metric)
        if value is None:
            value = row.get(metric)
        if value is None:
            continue
        x_value = x_value_for_record(record, x_axis)
        if x_value is None:
            continue
        seed = str(record.get("model_seed", record.get("seed", ""))) or "unseeded"
        points_by_x_and_seed.setdefault(x_value, {}).setdefault(seed, []).append(float(value))

    xs = sorted(points_by_x_and_seed)
    means: List[float] = []
    stds: List[float] = []
    ns: List[int] = []
    for x_value in xs:
        seed_values = [
            sum(values) / len(values)
            for values in points_by_x_and_seed[x_value].values()
            if values
        ]
        n = len(seed_values)
        mean = sum(seed_values) / n
        if n > 1:
            variance = sum((value - mean) ** 2 for value in seed_values) / (n - 1)
            std = variance ** 0.5
        else:
            std = 0.0
        means.append(mean)
        stds.append(std)
        ns.append(n)
    return xs, means, stds, ns


def band_bounds(
    means: Sequence[float],
    stds: Sequence[float],
    *,
    log_y: bool = False,
) -> Tuple[List[float], List[float]]:
    lowers = [mean - std for mean, std in zip(means, stds)]
    uppers = [mean + std for mean, std in zip(means, stds)]
    if not log_y:
        return lowers, uppers

    positive_values = [value for value in [*means, *uppers] if value > 0.0]
    floor = min(positive_values) * 0.1 if positive_values else 1e-12
    return [max(value, floor) for value in lowers], uppers


def plot_metric(
    records: Sequence[Dict[str, object]],
    *,
    threshold: int,
    metric: str,
    ylabel: str,
    output_path: Path,
    x_axis: str,
    percent: bool = False,
    log_y: bool = False,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(5.1, 3.35))
    styles = {
        "trained": {"label": "Trained", "marker": "o", "color": "#2f5d8c"},
        "random": {"label": "Random baseline", "marker": "s", "color": "#9b4d3f"},
    }
    for model in ("trained", "random"):
        xs, ys, stds, ns = values_for(records, model, threshold, metric, x_axis)
        if not xs:
            continue
        if percent:
            ys = [100.0 * y for y in ys]
            stds = [100.0 * std for std in stds]
        ax.plot(xs, ys, linewidth=1.8, markersize=4.5, **styles[model])
        if any(n > 1 for n in ns) and any(std > 0.0 for std in stds):
            lower, upper = band_bounds(ys, stds, log_y=log_y)
            ax.fill_between(
                xs,
                lower,
                upper,
                color=styles[model]["color"],
                alpha=0.18,
                linewidth=0.0,
            )

    if x_axis == "train_size":
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel(x_label_for_axis(x_axis))
    ax.set_ylabel(ylabel)
    xticks, xticklabels = display_x_ticks_for_records(records, x_axis)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, **x_tick_label_kwargs(x_axis, len(xticks)))
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        borderaxespad=0.0,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path)
    plt.close(fig)


def plot_type_and_occurrence_coverage(
    records: Sequence[Dict[str, object]],
    threshold: int,
    output_path: Path,
    x_axis: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.35), sharex=True)
    configs = [
        ("model_only_in_supergrammar_rule_type_pct", "Type coverage (%)"),
        ("model_only_in_supergrammar_occurrence_pct", "Occurrence coverage (%)"),
    ]
    styles = {
        "trained": {"label": "Trained", "marker": "o", "color": "#2f5d8c"},
        "random": {"label": "Random baseline", "marker": "s", "color": "#9b4d3f"},
    }
    xticks, xticklabels = display_x_ticks_for_records(records, x_axis)
    for ax, (metric, ylabel) in zip(axes, configs):
        for model in ("trained", "random"):
            xs, ys, stds, ns = values_for(records, model, threshold, metric, x_axis)
            ys = [100.0 * y for y in ys]
            stds = [100.0 * std for std in stds]
            if xs:
                ax.plot(xs, ys, linewidth=1.8, markersize=4.5, **styles[model])
            if xs and any(n > 1 for n in ns) and any(std > 0.0 for std in stds):
                lower, upper = band_bounds(ys, stds)
                ax.fill_between(
                    xs,
                    lower,
                    upper,
                    color=styles[model]["color"],
                    alpha=0.18,
                    linewidth=0.0,
                )
        if x_axis == "train_size":
            ax.set_xscale("log")
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, **x_tick_label_kwargs(x_axis, len(xticks)))
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.0, 100.0)
        ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
    )
    fig.supxlabel(x_label_for_axis(x_axis), y=0.02)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0.04, 1, 0.90))
    fig.savefig(output_path)
    plt.close(fig)


def make_plots(records: Sequence[Dict[str, object]], output_dir: Path, thresholds: Sequence[int], x_axis: str) -> None:
    setup_style()
    for threshold in thresholds:
        suffix = f"min{threshold}"
        plot_type_and_occurrence_coverage(
            records,
            threshold,
            output_dir / f"supergrammar_coverage_{suffix}.pdf",
            x_axis,
        )
        plot_metric(
            records,
            threshold=threshold,
            metric="model_only_occurrence_count",
            ylabel="Model-only rule occurrences",
            output_path=output_dir / f"model_only_occurrences_{suffix}.pdf",
            x_axis=x_axis,
            log_y=True,
        )
        plot_metric(
            records,
            threshold=threshold,
            metric="model_only_rule_type_count",
            ylabel="Model-only rule types",
            output_path=output_dir / f"model_only_types_{suffix}.pdf",
            x_axis=x_axis,
            log_y=True,
        )
        plot_metric(
            records,
            threshold=threshold,
            metric="model_only_in_supergrammar_rule_type_pct",
            ylabel="Type coverage (%)",
            output_path=output_dir / f"type_coverage_{suffix}.pdf",
            x_axis=x_axis,
            percent=True,
            ylim=(0.0, 100.0),
        )
        plot_metric(
            records,
            threshold=threshold,
            metric="model_only_in_supergrammar_occurrence_pct",
            ylabel="Occurrence coverage (%)",
            output_path=output_dir / f"occurrence_coverage_{suffix}.pdf",
            x_axis=x_axis,
            percent=True,
            ylim=(0.0, 100.0),
        )

        plot_metric(
            records,
            threshold=threshold,
            metric="grammar_global_jsd",
            ylabel="Global rule-distribution JSD",
            output_path=output_dir / f"grammar_global_jsd_{suffix}.pdf",
            x_axis=x_axis,
            ylim=(0.0, 1.0),
        )
        plot_metric(
            records,
            threshold=threshold,
            metric="grammar_per_lhs_weighted_jsd",
            ylabel="Weighted per-LHS JSD",
            output_path=output_dir / f"grammar_weighted_per_lhs_jsd_{suffix}.pdf",
            x_axis=x_axis,
            ylim=(0.0, 1.0),
        )

    # These are threshold-independent probe summaries from individual_layer_*.json when available.
    plot_metric(
        records,
        threshold=thresholds[-1],
        metric="probe_test_f1",
        ylabel="Decoded tree F1",
        output_path=output_dir / "probe_tree_f1.pdf",
        x_axis=x_axis,
        ylim=(0.0, 1.0),
    )
    plot_metric(
        records,
        threshold=thresholds[-1],
        metric="grammar_rule_f1",
        ylabel="Strict grammar rule F1",
        output_path=output_dir / "strict_grammar_rule_f1.pdf",
        x_axis=x_axis,
        ylim=(0.0, 1.0),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate appendix-ready plots from supergrammar_check.json probe outputs."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        action="append",
        default=[],
        help="Root to scan recursively for trained_model/random_baseline supergrammar_check.json files.",
    )
    parser.add_argument("--trained-root", type=Path, action="append", default=[], help="Root containing trained runs.")
    parser.add_argument("--random-root", type=Path, action="append", default=[], help="Root containing random-baseline runs.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for PDF figures and aggregate CSV.")
    parser.add_argument("--thresholds", default="5,10", help="Comma-separated min-support thresholds to plot.")
    parser.add_argument(
        "--x-axis",
        choices=["auto", "train_size", "layer", "checkpoint"],
        default="auto",
        help="Plot x-axis. Auto uses layer/checkpoint for layers/checkpoints runs and train size otherwise.",
    )
    parser.add_argument("--only-dev-size", type=int, default=None)
    parser.add_argument(
        "--allow-mixed-data-seeds",
        action="store_true",
        help="Allow mean/std plot aggregation even when known data seeds differ at the same plotted x-value.",
    )
    parser.add_argument(
        "--default-train-size",
        type=int,
        default=None,
        help="Fallback train size for legacy single-run directories that do not encode train size.",
    )
    parser.add_argument(
        "--final-step",
        type=int,
        default=None,
        help=(
            "Training step to assign to checkpoint label 'final' in checkpoint plots. "
            "If omitted, it is inferred as one checkpoint stride after the last numeric checkpoint."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roots: List[Tuple[Path, Optional[str]]] = []
    roots.extend((root, None) for root in args.results_root)
    roots.extend((root, "trained") for root in args.trained_root)
    roots.extend((root, "random") for root in args.random_root)
    if not roots:
        raise ValueError("Provide at least one --results-root, --trained-root, or --random-root.")

    thresholds = parse_thresholds(args.thresholds)
    records = collect_records(roots, args.default_train_size, args.final_step)
    if args.only_dev_size is not None:
        records = [
            record for record in records
            if int(record["dev_size"]) == int(args.only_dev_size)
        ]
    if not records:
        raise ValueError("No supergrammar_check.json files found under the provided roots.")

    x_axis = infer_x_axis(records, args.x_axis)
    for record in records:
        record["x_axis"] = x_axis
        value = x_value_for_record(record, x_axis)
        record["x_value"] = "" if value is None else value
    validate_data_seed_consistency(
        records,
        x_axis,
        allow_mixed_data_seeds=bool(args.allow_mixed_data_seeds),
    )

    output_dir = output_dir_with_mode_suffix(args.output_dir, records, x_axis)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_paper_results_csv(records, thresholds, output_dir / "paper_results.csv")
    write_aggregate_csv(records, thresholds, output_dir / "supergrammar_plot_data.csv")
    write_best_metrics_csv(records, thresholds, output_dir / "supergrammar_compact_metrics.csv")
    make_plots(records, output_dir, thresholds, x_axis)
    print(f"loaded_runs={len(records)}")
    print(f"x_axis={x_axis}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
