"""Probe decoding, grammar induction, and grammar-level comparison.

This is the main block-C implementation. It trains atomic constituency probes on
model hidden states, decodes predicted label sequences into trees, induces
grammars from gold and predicted evaluation trees, compares induced grammars to
gold evaluation grammars, and optionally checks model-only rules against the
broader source grammar. It also supports trained-model vs random-representation
baselines, layer/checkpoint analyses, and probe-size sweeps.
"""

from __future__ import annotations
import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple
from nltk import Nonterminal, PCFG, Tree

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_DIR = REPO_ROOT / "src"
if SRC_DIR.is_dir() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from C_Probing_and_Evaluation.vilares_constituency_probe import (
        DEFAULT_DEV_TREES_PATH,
        DEFAULT_MODEL_PATH,
        DEFAULT_TEST_TREES_PATH,
        DEFAULT_TRAIN_TREES_PATH,
        ProbeExample,
        ProbeTrainingConfig,
        build_probe_examples_from_trees,
        compact_metrics,
        decode_sequence_labels,
        labeled_bracketing_scores,
        load_model,
        load_tokenizer,
        load_random_baseline_model,
        predict_labels_for_example,
        prepare_tree_splits,
        resolve_device,
        train_probe,
        validate_requested_layer,
    )
except ModuleNotFoundError:
    from vilares_constituency_probe import (
        DEFAULT_DEV_TREES_PATH,
        DEFAULT_MODEL_PATH,
        DEFAULT_TEST_TREES_PATH,
        DEFAULT_TRAIN_TREES_PATH,
        ProbeExample,
        ProbeTrainingConfig,
        build_probe_examples_from_trees,
        compact_metrics,
        decode_sequence_labels,
        labeled_bracketing_scores,
        load_model,
        load_tokenizer,
        load_random_baseline_model,
        predict_labels_for_example,
        prepare_tree_splits,
        resolve_device,
        train_probe,
        validate_requested_layer,
    )
from A_PCFG.LLk_core import (
    prune_nonproductive_structural_backbone,
    prune_unreachable_and_renormalize,
    prune_unreachable_structural_backbone,
    build_final_pcfg,
    extend_lexicon_to_cover_terminal_vocabulary,
    global_rule_distribution,
    jensen_shannon_divergence,
    mle_estimate_pcfg,
    preterminal_labels_from_grammar,
    save_grammar_earleyx,
    structural_rules_from_grammar,
    tree_terminal_productions,
    tree_structural_productions,
    terminal_rules_from_grammar
)
from A_PCFG.llk_parser import ROOT_SYMBOL as START_SYMBOL

DEFAULT_RULE_SUPPORT_THRESHOLDS = (1, 2, 3, 5, 10)


@dataclass
class GrammarInductionBundle:
    final_grammar: PCFG
    raw_grammar: PCFG
    structural_rule_counts: Counter
    terminal_rule_counts: Counter
    structural_rules: Set[Tuple[str, Tuple[str, ...]]]
    reachable_preterminals: Set[str]
    lexical_extension_missing_words: List[str]


def tree_node_to_nltk(node) -> Tree:
    children: List[object] = []
    for child in node.children:
        if isinstance(child, str):
            children.append(child)
        else:
            children.append(tree_node_to_nltk(child))
    return Tree(node.label, children)


def ensure_start_symbol(tree: Tree, start_symbol: str = START_SYMBOL) -> Tree:
    if tree.label() == start_symbol:
        return tree
    return Tree(start_symbol, [tree])


def normalize_distribution(dist: Dict[object, float]) -> Dict[object, float]:
    total = float(sum(dist.values()))
    if total <= 0.0:
        return {}
    return {key: float(value) / total for key, value in dist.items() if value > 0.0}


def estimate_rule_counts(trees: Sequence[Tree]) -> Tuple[Counter, Counter]:
    structural_counts: Counter = Counter()
    terminal_counts: Counter = Counter()
    for tree in trees:
        structural_counts.update(tree_structural_productions(tree))
        terminal_counts.update(tree_terminal_productions(tree))
    return structural_counts, terminal_counts


def stable_prune_structural_backbone(
    structural_rules: Set[Tuple[str, Tuple[str, ...]]],
    preterminal_labels: Set[str],
    start_symbol: str,
) -> Tuple[Set[Tuple[str, Tuple[str, ...]]], Set[str]]:
    current_rules = set(structural_rules)
    current_preterminals = set(preterminal_labels)
    while True:
        productive_rules, productive_symbols = prune_nonproductive_structural_backbone(
            current_rules,
            current_preterminals,
        )
        productive_preterminals = current_preterminals & productive_symbols
        reachable_rules, _reachable_nonterminals, reachable_preterminals = (
            prune_unreachable_structural_backbone(
                productive_rules,
                productive_preterminals,
                start_symbol,
            )
        )
        next_rules = set(reachable_rules)
        next_preterminals = set(reachable_preterminals)
        if next_rules == current_rules and next_preterminals == current_preterminals:
            return next_rules, next_preterminals
        current_rules = next_rules
        current_preterminals = next_preterminals


def lexical_assignment_from_grammar(
    grammar: PCFG,
    allowed_preterminals: Set[str],
) -> Dict[str, Dict[str, float]]:
    by_lhs: Dict[str, Dict[str, float]] = defaultdict(dict)
    for (lhs, word), prob in terminal_rules_from_grammar(grammar).items():
        if lhs in allowed_preterminals:
            by_lhs[lhs][word] = float(prob)
    return dict(by_lhs)


def induce_grammar_from_treebank(
    trees: Sequence[Tree],
    *,
    start_symbol: str,
) -> GrammarInductionBundle:
    if not trees:
        raise ValueError("Cannot induce a grammar from an empty treebank.")

    raw_grammar = mle_estimate_pcfg(Nonterminal(start_symbol), trees)
    structural_counts, terminal_counts = estimate_rule_counts(trees)
    structural_rules = set(structural_rules_from_grammar(raw_grammar))
    preterminal_labels = set(preterminal_labels_from_grammar(raw_grammar))
    pruned_rules, reachable_preterminals = stable_prune_structural_backbone(
        structural_rules,
        preterminal_labels,
        start_symbol,
    )

    lexical_assignment = lexical_assignment_from_grammar(raw_grammar, reachable_preterminals)
    lexical_assignment, missing_words = extend_lexicon_to_cover_terminal_vocabulary(
        raw_grammar,
        lexical_assignment,
        terminal_counts,
        reachable_preterminals,
    )
    final_grammar = build_final_pcfg(
        raw_grammar,
        pruned_rules,
        lexical_assignment,
        start_symbol,
    )
    final_grammar = prune_unreachable_and_renormalize(final_grammar)
    return GrammarInductionBundle(
        final_grammar=final_grammar,
        raw_grammar=raw_grammar,
        structural_rule_counts=structural_counts,
        terminal_rule_counts=terminal_counts,
        structural_rules=pruned_rules,
        reachable_preterminals=reachable_preterminals,
        lexical_extension_missing_words=missing_words,
    )


def rule_overlap_metrics(
    gold_rules: Set[Tuple[str, Tuple[str, ...]]],
    model_rules: Set[Tuple[str, Tuple[str, ...]]],
) -> Dict[str, object]:
    shared = gold_rules & model_rules
    model_only = model_rules - gold_rules
    gold_only = gold_rules - model_rules
    precision = len(shared) / len(model_rules) if model_rules else 0.0
    recall = len(shared) / len(gold_rules) if gold_rules else 0.0
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "gold_rule_count": len(gold_rules),
        "model_rule_count": len(model_rules),
        "shared_rule_count": len(shared),
        "gold_rule_recovery_pct": recall,
        "model_spurious_rule_pct": len(model_only) / len(model_rules) if model_rules else 0.0,
        "gold_only_rule_count": len(gold_only),
        "model_only_rule_count": len(model_only),
        "shared_rules_preview": [serialize_rule(rule) for rule in sorted(shared)[:25]],
        "gold_only_rules_preview": [serialize_rule(rule) for rule in sorted(gold_only)[:25]],
        "model_only_rules_preview": [serialize_rule(rule) for rule in sorted(model_only)[:25]],
    }


def rules_with_min_count(counts: Counter, min_count: int) -> Set[Tuple[str, Tuple[str, ...]]]:
    return {rule for rule, count in counts.items() if int(count) >= int(min_count)}


def rule_overlap_from_counts(
    gold_counts: Counter,
    model_counts: Counter,
    min_count: int,
) -> Dict[str, object]:
    return rule_overlap_metrics(
        rules_with_min_count(gold_counts, min_count),
        rules_with_min_count(model_counts, min_count),
    )


def serialize_rule(rule: Tuple[str, Tuple[str, ...]]) -> Dict[str, object]:
    lhs, rhs = rule
    return {"lhs": lhs, "rhs": list(rhs)}


def serialize_rule_text(rule: Tuple[str, Tuple[str, ...]]) -> Tuple[str, str]:
    lhs, rhs = rule
    return lhs, " ".join(rhs)


def per_lhs_distributions(counts: Counter) -> Dict[str, Dict[Tuple[str, ...], float]]:
    grouped: Dict[str, Counter] = defaultdict(Counter)
    for (lhs, rhs), count in counts.items():
        grouped[lhs][rhs] += int(count)
    return {lhs: normalize_distribution(dict(rhs_counts)) for lhs, rhs_counts in grouped.items()}


def per_lhs_jsd_metrics(gold_counts: Counter, model_counts: Counter) -> Dict[str, object]:
    gold_dists = per_lhs_distributions(gold_counts)
    model_dists = per_lhs_distributions(model_counts)
    all_lhs = sorted(set(gold_dists) | set(model_dists))
    rows: List[Dict[str, object]] = []
    for lhs in all_lhs:
        jsd = jensen_shannon_divergence(gold_dists.get(lhs, {}), model_dists.get(lhs, {}))
        rows.append(
            {
                "lhs": lhs,
                "jsd": jsd,
                "gold_expansion_count": len(gold_dists.get(lhs, {})),
                "model_expansion_count": len(model_dists.get(lhs, {})),
            }
        )
    if not rows:
        return {"macro_jsd": 0.0, "weighted_jsd": 0.0, "rows": []}

    gold_total_by_lhs = defaultdict(int)
    for (lhs, _rhs), count in gold_counts.items():
        gold_total_by_lhs[lhs] += int(count)
    model_total_by_lhs = defaultdict(int)
    for (lhs, _rhs), count in model_counts.items():
        model_total_by_lhs[lhs] += int(count)

    macro_jsd = sum(row["jsd"] for row in rows) / len(rows)
    weights = {
        lhs: gold_total_by_lhs.get(lhs, 0) + model_total_by_lhs.get(lhs, 0)
        for lhs in all_lhs
    }
    total_weight = sum(weights.values())
    weighted_jsd = (
        sum(row["jsd"] * weights[row["lhs"]] for row in rows) / total_weight if total_weight > 0 else macro_jsd
    )
    rows.sort(key=lambda row: (-float(row["jsd"]), row["lhs"]))
    return {
        "macro_jsd": macro_jsd,
        "weighted_jsd": weighted_jsd,
        "rows": rows,
    }


def grammar_size_metrics(grammar: PCFG) -> Dict[str, int]:
    nonterminals = {str(prod.lhs()) for prod in grammar.productions()}
    structural_rules = structural_rules_from_grammar(grammar)
    lexical_rules = terminal_rules_from_grammar(grammar)
    return {
        "nonterminal_count": len(nonterminals),
        "structural_rule_count": len(structural_rules),
        "lexical_rule_count": len(lexical_rules),
        "total_rule_count": len(structural_rules) + len(lexical_rules),
    }


def grammar_comparison_report(
    gold_bundle: GrammarInductionBundle,
    model_bundle: GrammarInductionBundle,
) -> Dict[str, object]:
    gold_global = global_rule_distribution(gold_bundle.structural_rule_counts)
    model_global = global_rule_distribution(model_bundle.structural_rule_counts)
    per_lhs = per_lhs_jsd_metrics(gold_bundle.structural_rule_counts, model_bundle.structural_rule_counts)
    return {
        "rule_overlap": rule_overlap_metrics(gold_bundle.structural_rules, model_bundle.structural_rules),
        "rule_distribution_jsd": {
            "global_jsd": jensen_shannon_divergence(gold_global, model_global),
            "per_lhs_macro_jsd": per_lhs["macro_jsd"],
            "per_lhs_weighted_jsd": per_lhs["weighted_jsd"],
            "per_lhs_rows": per_lhs["rows"][:50],
        },
        "grammar_sizes": {
            "gold": grammar_size_metrics(gold_bundle.final_grammar),
            "model": grammar_size_metrics(model_bundle.final_grammar),
        },
        "induction_diagnostics": {
            "gold_missing_words_after_lexical_extension": gold_bundle.lexical_extension_missing_words,
            "model_missing_words_after_lexical_extension": model_bundle.lexical_extension_missing_words,
            "gold_reachable_preterminal_count": len(gold_bundle.reachable_preterminals),
            "model_reachable_preterminal_count": len(model_bundle.reachable_preterminals),
        },
    }


def parse_rule_support_thresholds(value: str) -> List[int]:
    thresholds: List[int] = []
    for raw in str(value).split(","):
        raw = raw.strip()
        if not raw:
            continue
        threshold = int(raw)
        if threshold <= 0:
            raise ValueError(f"Rule support thresholds must be positive, got {threshold}.")
        thresholds.append(threshold)
    return sorted(set(thresholds)) or list(DEFAULT_RULE_SUPPORT_THRESHOLDS)


def token_oov_metrics(train_examples: Sequence[ProbeExample], test_examples: Sequence[ProbeExample]) -> Dict[str, object]:
    train_atomic = {label.atomic_string() for example in train_examples for label in example.labels}
    train_n = {label.n for example in train_examples for label in example.labels}
    train_c = {label.c for example in train_examples for label in example.labels}
    train_u = {label.u for example in train_examples for label in example.labels}

    test_labels = [label for example in test_examples for label in example.labels]
    test_atomic_oov = [label for label in test_labels if label.atomic_string() not in train_atomic]
    test_n_oov = [label for label in test_labels if label.n not in train_n]
    test_c_oov = [label for label in test_labels if label.c not in train_c]
    test_u_oov = [label for label in test_labels if label.u not in train_u]
    total = len(test_labels)

    return {
        "test_token_count": total,
        "train_atomic_class_count": len(train_atomic),
        "test_atomic_oov_count": len(test_atomic_oov),
        "test_atomic_oov_rate": len(test_atomic_oov) / total if total else 0.0,
        "train_n_class_count": len(train_n),
        "test_n_oov_count": len(test_n_oov),
        "test_n_oov_rate": len(test_n_oov) / total if total else 0.0,
        "train_c_class_count": len(train_c),
        "test_c_oov_count": len(test_c_oov),
        "test_c_oov_rate": len(test_c_oov) / total if total else 0.0,
        "train_u_class_count": len(train_u),
        "test_u_oov_count": len(test_u_oov),
        "test_u_oov_rate": len(test_u_oov) / total if total else 0.0,
        "test_atomic_oov_types": len({label.atomic_string() for label in test_atomic_oov}),
        "test_n_oov_types": len({label.n for label in test_n_oov}),
        "test_c_oov_types": len({label.c for label in test_c_oov}),
        "test_u_oov_types": len({label.u for label in test_u_oov}),
    }


def write_compact_analysis_csv(
    output_path: Path,
    *,
    run: Dict[str, object],
    probe_result: Dict[str, object],
    tree_scores: Dict[str, float],
    oov_metrics: Dict[str, object],
    gold_original_bundle: GrammarInductionBundle,
    decoded_gold_bundle: GrammarInductionBundle,
    model_channel_bundle: GrammarInductionBundle,
    rule_support_thresholds: Sequence[int],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []

    # Total structural-rule occurrences in each treebank-derived grammar.
    # These are the denominators for per-rule occurrence percentages.
    gold_original_total = int(sum(gold_original_bundle.structural_rule_counts.values()))
    model_channel_total = int(sum(model_channel_bundle.structural_rule_counts.values()))

    def pct(count: object, total: int) -> object:
        if count == "" or total <= 0:
            return ""
        return float(count) / float(total)

    def add_metric(
        section: str,
        metric: str,
        value: object,
        comparison: str = "",
        threshold: object = "",
    ) -> None:
        rows.append(
            {
                "section": section,
                "comparison": comparison,
                "threshold": threshold,
                "metric": metric,
                "value": value,
                "lhs": "",
                "rhs": "",
                "gold_count": "",
                "model_count": "",
                "original_count": "",
                "gold_pct": "",
                "model_pct": "",
                "original_pct": "",
            }
        )

    def add_rule(
        section: str,
        comparison: str,
        rule: Tuple[str, Tuple[str, ...]],
        *,
        gold_count: object = "",
        model_count: object = "",
        original_count: object = "",
        gold_pct: object = "",
        model_pct: object = "",
        original_pct: object = "",
    ) -> None:
        lhs, rhs = serialize_rule_text(rule)
        rows.append(
            {
                "section": section,
                "comparison": comparison,
                "threshold": "",
                "metric": "rule",
                "value": "",
                "lhs": lhs,
                "rhs": rhs,
                "gold_count": gold_count,
                "model_count": model_count,
                "original_count": original_count,
                "gold_pct": gold_pct,
                "model_pct": model_pct,
                "original_pct": original_pct,
            }
        )

    add_metric("run", "layer", int(run["layer"]))
    add_metric("run", "best_epoch", int(probe_result["best_epoch"]))
    add_metric("probe", "test_tree_precision", float(tree_scores["precision"]))
    add_metric("probe", "test_tree_recall", float(tree_scores["recall"]))
    add_metric("probe", "test_tree_f1", float(tree_scores["f1"]))

    for key, value in compact_metrics(probe_result["test_metrics"]).items():
        add_metric("probe", f"test_{key}", value)

    for key, value in oov_metrics.items():
        add_metric("oov", key, value)

    comparisons = {
        "model_vs_gold_original": (gold_original_bundle, model_channel_bundle),
    }

    for comparison_name, (gold_bundle, model_bundle) in comparisons.items():
        for threshold in rule_support_thresholds:
            overlap = rule_overlap_from_counts(
                gold_bundle.structural_rule_counts,
                model_bundle.structural_rule_counts,
                int(threshold),
            )
            for metric in (
                "precision",
                "recall",
                "f1",
                "gold_rule_count",
                "model_rule_count",
                "shared_rule_count",
                "gold_only_rule_count",
                "model_only_rule_count",
                "model_spurious_rule_pct",
            ):
                add_metric(
                    "rule_overlap_min_support",
                    metric,
                    overlap[metric],
                    comparison=comparison_name,
                    threshold=threshold,
                )

    gold_original_rules = set(gold_original_bundle.structural_rule_counts)
    model_channel_rules = set(model_channel_bundle.structural_rule_counts)

    model_only_rules = sorted(model_channel_rules - gold_original_rules)
    gold_only_rules = sorted(gold_original_rules - model_channel_rules)

    for max_count in rule_support_thresholds:
        count = sum(
            1
            for rule in model_only_rules
            if int(model_channel_bundle.structural_rule_counts[rule]) <= int(max_count)
        )
        add_metric(
            "model_only_low_support",
            "model_only_rules_with_count_at_most_threshold",
            count,
            comparison="model_vs_gold_original",
            threshold=max_count,
        )

    for rule in sorted(
        model_only_rules,
        key=lambda r: (int(model_channel_bundle.structural_rule_counts[r]), r),
    ):
        model_count = int(model_channel_bundle.structural_rule_counts[rule])
        add_rule(
            "model_only_rules",
            "model_vs_gold_original",
            rule,
            gold_count=0,
            model_count=model_count,
            gold_pct=0.0,
            model_pct=pct(model_count, model_channel_total),
        )

    # Rules that occur in the gold-original grammar but are missing
    # from the model-channel grammar.
    for rule in sorted(
        gold_only_rules,
        key=lambda r: (int(gold_original_bundle.structural_rule_counts[r]), r),
    ):
        gold_count = int(gold_original_bundle.structural_rule_counts[rule])
        add_rule(
            "gold_only_rules",
            "model_vs_gold_original",
            rule,
            gold_count=gold_count,
            model_count=0,
            gold_pct=pct(gold_count, gold_original_total),
            model_pct=0.0,
        )

    fieldnames = [
        "section",
        "comparison",
        "threshold",
        "metric",
        "value",
        "lhs",
        "rhs",
        "gold_count",
        "model_count",
        "original_count",
        "gold_pct",
        "model_pct",
        "original_pct",
    ]

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
def append_model_only_hierarchy_csv(
    output_path: Path,
    hierarchy: Dict[str, object],
) -> None:
    rows: List[Dict[str, object]] = []

    fieldnames = [
        "section",
        "comparison",
        "threshold",
        "metric",
        "value",
        "lhs",
        "rhs",
        "gold_count",
        "model_count",
        "original_count",
    ]

    def add_metric(metric: str, value: object) -> None:
        rows.append(
            {
                "section": "model_only_train_hierarchy",
                "comparison": "model_vs_test_gold_original",
                "threshold": "",
                "metric": metric,
                "value": value,
                "lhs": "",
                "rhs": "",
                "gold_count": "",
                "model_count": "",
                "original_count": "",
            }
        )

    scalar_keys = [
        key for key, value in hierarchy.items()
        if isinstance(value, (int, float))
    ]

    for key in sorted(scalar_keys):
        add_metric(key, hierarchy[key])

    def add_rule_rows(section: str, preview_key: str) -> None:
        for row in hierarchy.get(preview_key, []):
            rows.append(
                {
                    "section": section,
                    "comparison": "model_vs_test_gold_original",
                    "threshold": "",
                    "metric": "rule",
                    "value": "",
                    "lhs": row["lhs"],
                    "rhs": " ".join(row["rhs"]),
                    "gold_count": row.get("test_gold_original_count", ""),
                    "model_count": row.get("model_test_count", ""),
                    "original_count": row.get("train_gold_original_count", ""),
                }
            )

    add_rule_rows(
        "model_only_seen_in_train_decoded",
        "seen_in_train_decoded_preview",
    )
    add_rule_rows(
        "model_only_seen_in_train_original_only",
        "seen_in_train_original_only_preview",
    )
    add_rule_rows(
        "model_only_novel_vs_train_decoded_and_original",
        "novel_vs_train_decoded_and_original_preview",
    )

    with output_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writerows(rows)


def write_treebank_jsonl(trees: Sequence[Tree], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for tree in trees:
            sentence = " ".join(tree.leaves())
            handle.write(json.dumps({"sentence": sentence, "tree": tree.pformat(margin=10**9)}) + "\n")


def save_induced_grammar_bundle(bundle: GrammarInductionBundle, output_dir: Path, prefix: str) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    grammar_path = output_dir / f"{prefix}.grammar.txt"
    raw_grammar_path = output_dir / f"{prefix}.raw_grammar.txt"
    save_grammar_earleyx(bundle.final_grammar, grammar_path)
    save_grammar_earleyx(bundle.raw_grammar, raw_grammar_path)
    return {
        "grammar_path": str(grammar_path),
        "raw_grammar_path": str(raw_grammar_path),
    }


def load_supergrammar_structural_rules(path: Path) -> Set[Tuple[str, Tuple[str, ...]]]:
    rules: Set[Tuple[str, Tuple[str, ...]]] = set()
    line_re = re.compile(r"^\s*(?P<lhs>\S+)\s*->\s*\[(?P<rhs>.*?)\]\s*:")
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            match = line_re.match(line)
            if not match:
                raise ValueError(f"Could not parse supergrammar line {line_no} from {path}: {line!r}")

            lhs = match.group("lhs")
            rhs_items = tuple(item for item in match.group("rhs").split() if item)
            if not rhs_items:
                continue

            # EarleyX grammar files prefix lexical terminals with "_".
            # For structural-rule comparison, keep only nonterminal RHS symbols.
            if any(not item.startswith("_") for item in rhs_items):
                rhs = tuple(item for item in rhs_items if not item.startswith("_"))
                rules.add((lhs, rhs))
    return rules


def supergrammar_threshold_report(
    *,
    model_counts: Counter,
    gold_counts: Counter,
    supergrammar_rules: Set[Tuple[str, Tuple[str, ...]]],
    min_count: int,
) -> Dict[str, object]:
    model_rules = rules_with_min_count(model_counts, min_count)
    gold_rules = rules_with_min_count(gold_counts, min_count)
    model_only_rules = model_rules - gold_rules
    model_only_in_supergrammar = model_only_rules & supergrammar_rules
    model_only_not_in_supergrammar = model_only_rules - supergrammar_rules

    model_only_occurrences = int(sum(model_counts[rule] for rule in model_only_rules))
    in_supergrammar_occurrences = int(sum(model_counts[rule] for rule in model_only_in_supergrammar))
    not_in_supergrammar_occurrences = int(sum(model_counts[rule] for rule in model_only_not_in_supergrammar))

    def occurrence_pct(count: int, total: int) -> float:
        return float(count) / float(total) if total else 0.0

    def preview(rules: Set[Tuple[str, Tuple[str, ...]]]) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for rule in sorted(rules, key=lambda item: (-int(model_counts[item]), item[0], item[1]))[:50]:
            lhs, rhs = rule
            rows.append(
                {
                    "lhs": lhs,
                    "rhs": " ".join(rhs),
                    "model_only_count": int(model_counts[rule]),
                }
            )
        return rows

    return {
        "min_count": int(min_count),
        "model_rule_type_count": len(model_rules),
        "gold_rule_type_count": len(gold_rules),
        "supergrammar_structural_rule_type_count": len(supergrammar_rules),
        "model_only_rule_type_count": len(model_only_rules),
        "model_only_occurrence_count": model_only_occurrences,
        "model_only_in_supergrammar_rule_type_count": len(model_only_in_supergrammar),
        "model_only_in_supergrammar_rule_type_pct": occurrence_pct(
            len(model_only_in_supergrammar),
            len(model_only_rules),
        ),
        "model_only_in_supergrammar_occurrence_count": in_supergrammar_occurrences,
        "model_only_in_supergrammar_occurrence_pct": occurrence_pct(
            in_supergrammar_occurrences,
            model_only_occurrences,
        ),
        "model_only_not_in_supergrammar_rule_type_count": len(model_only_not_in_supergrammar),
        "model_only_not_in_supergrammar_occurrence_count": not_in_supergrammar_occurrences,
        "model_only_in_supergrammar_preview": preview(model_only_in_supergrammar),
        "model_only_not_in_supergrammar_preview": preview(model_only_not_in_supergrammar),
    }


def supergrammar_comparison_report(
    *,
    model_counts: Counter,
    gold_counts: Counter,
    supergrammar_rules: Set[Tuple[str, Tuple[str, ...]]],
    thresholds: Sequence[int],
) -> Dict[str, object]:
    threshold_rows = [
        supergrammar_threshold_report(
            model_counts=model_counts,
            gold_counts=gold_counts,
            supergrammar_rules=supergrammar_rules,
            min_count=threshold,
        )
        for threshold in thresholds
    ]
    base = supergrammar_threshold_report(
        model_counts=model_counts,
        gold_counts=gold_counts,
        supergrammar_rules=supergrammar_rules,
        min_count=1,
    )
    base["thresholds"] = threshold_rows
    return base


def write_supergrammar_rule_csv(
    output_path: Path,
    *,
    model_counts: Counter,
    gold_counts: Counter,
    supergrammar_rules: Set[Tuple[str, Tuple[str, ...]]],
) -> None:
    model_only_rules = set(model_counts) - set(gold_counts)
    fieldnames = ["section", "lhs", "rhs", "model_only_count", "in_supergrammar"]
    rows: List[Dict[str, object]] = []
    for lhs, rhs in sorted(model_only_rules, key=lambda rule: (-int(model_counts[rule]), rule[0], rule[1])):
        rule = (lhs, rhs)
        rows.append(
            {
                "section": "model_only_rule",
                "lhs": lhs,
                "rhs": " ".join(rhs),
                "model_only_count": int(model_counts[rule]),
                "in_supergrammar": rule in supergrammar_rules,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def predict_treebank(
    probe_result: Dict[str, object],
    examples: Sequence[ProbeExample],
    config: ProbeTrainingConfig,
) -> List[Tree]:
    predicted: List[Tree] = []
    for example in examples:
        predicted_labels = predict_labels_for_example(probe_result["model"], probe_result["vocab"], example, config)
        predicted_tree = decode_sequence_labels(example.words, predicted_labels)
        predicted.append(ensure_start_symbol(tree_node_to_nltk(predicted_tree)))
    return predicted


def gold_treebank(examples: Sequence[ProbeExample]) -> List[Tree]:
    return [ensure_start_symbol(tree_node_to_nltk(example.gold_tree)) for example in examples]


def decoded_gold_treebank(examples: Sequence[ProbeExample]) -> List[Tree]:
    return [
        ensure_start_symbol(tree_node_to_nltk(decode_sequence_labels(example.words, example.labels)))
        for example in examples
    ]


def run_probe_training(
    args: argparse.Namespace,
    *,
    model_path: str,
    tokenizer_path: str,
    layer: int,
    split_override: Optional[Tuple[List[str], List[str], List[str], Dict[str, object]]] = None,
) -> Dict[str, object]:
    device = resolve_device(args.device)
    tokenizer = load_tokenizer(tokenizer_path, local_files_only=args.local_files_only)

    if args.run_baseline:
        model = load_random_baseline_model(
            model_path,
            local_files_only=args.local_files_only,
            device=device,
            model_init_seed=int(args.model_init_seed),
        )
    else:
        model = load_model(model_path, local_files_only=args.local_files_only, device=device)
    layer = validate_requested_layer(model, int(layer))
    if split_override is None:
        train_tree_strings, dev_tree_strings, test_tree_strings, split_info = prepare_tree_splits(args)
    else:
        train_tree_strings, dev_tree_strings, test_tree_strings, split_info = split_override
    train_examples = build_probe_examples_from_trees(
        train_tree_strings,
        model,
        layer,
        local_files_only=args.local_files_only,
        device=device,
        batch_size=int(args.extract_batch_size),
        tokenizer=tokenizer,
    )
    dev_examples = build_probe_examples_from_trees(
        dev_tree_strings,
        model,
        layer,
        local_files_only=args.local_files_only,
        device=device,
        batch_size=int(args.extract_batch_size),
        tokenizer=tokenizer,
    )
    test_examples = build_probe_examples_from_trees(
        test_tree_strings,
        model,
        layer,
        local_files_only=args.local_files_only,
        device=device,
        batch_size=int(args.extract_batch_size),
        tokenizer=tokenizer,
    )
    config = ProbeTrainingConfig(
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        patience=int(args.patience),
        device=device,
        seed=int(args.seed),
    )
    trained_probe = train_probe(train_examples, dev_examples, test_examples, config)
    return {
        "probe_result": trained_probe,
        "config": config,
        "split_info": split_info,
        "train_examples": train_examples,
        "dev_examples": dev_examples,
        "test_examples": test_examples,
        "layer": layer,
    }


def compare_for_trained_probe(
    args: argparse.Namespace,
    *,
    model_path: str,
    tokenizer_path: str,
    layer: int,
    checkpoint_label: str,
    output_root: Path,
    split_override: Optional[Tuple[List[str], List[str], List[str], Dict[str, object]]] = None,
) -> Dict[str, object]:
    run = run_probe_training(
        args,
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        layer=layer,
        split_override=split_override,
    )

    probe_result = run["probe_result"]
    config = run["config"]
    train_examples = run["train_examples"]
    test_examples = run["test_examples"]

    # ------------------------------------------------------------------
    # Build treebanks
    # ------------------------------------------------------------------

    test_gold_original_trees = gold_treebank(test_examples)
    test_decoded_gold_trees = decoded_gold_treebank(test_examples)
    test_model_channel_trees = predict_treebank(probe_result, test_examples, config)

    train_gold_original_trees = gold_treebank(train_examples)
    train_decoded_gold_trees = decoded_gold_treebank(train_examples)

    # ------------------------------------------------------------------
    # Induce grammars / count rules
    # ------------------------------------------------------------------

    test_gold_original_bundle = induce_grammar_from_treebank(
        test_gold_original_trees,
        start_symbol=START_SYMBOL,
    )
    test_decoded_gold_bundle = induce_grammar_from_treebank(
        test_decoded_gold_trees,
        start_symbol=START_SYMBOL,
    )
    test_model_channel_bundle = induce_grammar_from_treebank(
        test_model_channel_trees,
        start_symbol=START_SYMBOL,
    )

    train_gold_original_bundle = induce_grammar_from_treebank(
        train_gold_original_trees,
        start_symbol=START_SYMBOL,
    )
    train_decoded_gold_bundle = induce_grammar_from_treebank(
        train_decoded_gold_trees,
        start_symbol=START_SYMBOL,
    )

    # ------------------------------------------------------------------
    # Main grammar comparisons
    # ------------------------------------------------------------------

    model_vs_original_gold = grammar_comparison_report(
        test_gold_original_bundle,
        test_model_channel_bundle,
    )

    # ------------------------------------------------------------------
    # Four-way hierarchy for model-only rules
    # ------------------------------------------------------------------

    test_model_channel_rules = set(test_model_channel_bundle.structural_rule_counts)
    train_decoded_gold_rules = set(train_decoded_gold_bundle.structural_rule_counts)
    train_gold_original_rules = set(train_gold_original_bundle.structural_rule_counts)

    test_gold_original_rules = set(test_gold_original_bundle.structural_rule_counts)

    # 1. Rules produced by the model that are not in the test gold-original grammar.
    model_only_vs_test_original = test_model_channel_rules - test_gold_original_rules

    # 2. Of those, rules already present in the decoded training labels.
    seen_in_train_decoded = model_only_vs_test_original & train_decoded_gold_rules

    # 3. Of the remaining rules, rules present in train-original.
    remaining_after_train_decoded = model_only_vs_test_original - seen_in_train_decoded
    seen_in_train_original_only = remaining_after_train_decoded & train_gold_original_rules

    # 4. Rules absent from both decoded training labels and train-original.
    novel_vs_both_train = remaining_after_train_decoded - seen_in_train_original_only

    # Useful extra categories.
    seen_in_train_original_any = model_only_vs_test_original & train_gold_original_rules
    seen_in_either_train = model_only_vs_test_original & (
        train_decoded_gold_rules | train_gold_original_rules
    )

    def model_rule_token_count(rules: Set[Tuple[str, Tuple[str, ...]]]) -> int:
        return int(
            sum(test_model_channel_bundle.structural_rule_counts[rule] for rule in rules)
        )

    def train_decoded_token_count(rules: Set[Tuple[str, Tuple[str, ...]]]) -> int:
        return int(
            sum(train_decoded_gold_bundle.structural_rule_counts.get(rule, 0) for rule in rules)
        )

    def train_original_token_count(rules: Set[Tuple[str, Tuple[str, ...]]]) -> int:
        return int(
            sum(train_gold_original_bundle.structural_rule_counts.get(rule, 0) for rule in rules)
        )

    def sorted_rules_by_model_count(
        rules: Set[Tuple[str, Tuple[str, ...]]],
    ) -> List[Tuple[str, Tuple[str, ...]]]:
        return sorted(
            rules,
            key=lambda rule: (
                -int(test_model_channel_bundle.structural_rule_counts[rule]),
                rule[0],
                rule[1],
            ),
        )

    def rule_rows(
        rules: Set[Tuple[str, Tuple[str, ...]]],
        limit: int = 50,
    ) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for rule in sorted_rules_by_model_count(rules)[:limit]:
            lhs, rhs = rule
            rows.append(
                {
                    "lhs": lhs,
                    "rhs": list(rhs),
                    "model_test_count": int(
                        test_model_channel_bundle.structural_rule_counts.get(rule, 0)
                    ),
                    "test_decoded_gold_count": int(
                        test_decoded_gold_bundle.structural_rule_counts.get(rule, 0)
                    ),
                    "train_decoded_gold_count": int(
                        train_decoded_gold_bundle.structural_rule_counts.get(rule, 0)
                    ),
                    "train_gold_original_count": int(
                        train_gold_original_bundle.structural_rule_counts.get(rule, 0)
                    ),
                    "test_gold_original_count": int(
                        test_gold_original_bundle.structural_rule_counts.get(rule, 0)
                    ),
                }
            )
        return rows

    model_only_train_hierarchy = {
        "definition": (
            "Rules are predicted structural rule types that are absent from "
            "the test gold-original grammar. They are then classified by whether "
            "they appear in decoded training labels, then train gold-original, or neither."
        ),

        # 1. model-only vs test gold-original
        "model_only_vs_test_gold_original_rule_count": len(model_only_vs_test_original),
        "model_only_vs_test_gold_original_model_count": model_rule_token_count(
            model_only_vs_test_original
        ),

        # 2. seen in decoded training labels
        "seen_in_train_decoded_rule_count": len(seen_in_train_decoded),
        "seen_in_train_decoded_model_count": model_rule_token_count(
            seen_in_train_decoded
        ),
        "seen_in_train_decoded_train_count": train_decoded_token_count(
            seen_in_train_decoded
        ),

        # 3. not decoded training labels, but seen in train-original
        "seen_in_train_original_only_rule_count": len(seen_in_train_original_only),
        "seen_in_train_original_only_model_count": model_rule_token_count(
            seen_in_train_original_only
        ),
        "seen_in_train_original_only_train_count": train_original_token_count(
            seen_in_train_original_only
        ),

        # 4. absent from both decoded training labels and train-original
        "novel_vs_train_decoded_and_original_rule_count": len(novel_vs_both_train),
        "novel_vs_train_decoded_and_original_model_count": model_rule_token_count(
            novel_vs_both_train
        ),

        # Extra summary numbers.
        "seen_in_train_original_any_rule_count": len(seen_in_train_original_any),
        "seen_in_train_original_any_model_count": model_rule_token_count(
            seen_in_train_original_any
        ),
        "seen_in_either_train_rule_count": len(seen_in_either_train),
        "seen_in_either_train_model_count": model_rule_token_count(
            seen_in_either_train
        ),

        # Percentages over model-only rules.
        "seen_in_train_decoded_rule_pct": (
            len(seen_in_train_decoded) / len(model_only_vs_test_original)
            if model_only_vs_test_original
            else 0.0
        ),
        "seen_in_train_original_only_rule_pct": (
            len(seen_in_train_original_only) / len(model_only_vs_test_original)
            if model_only_vs_test_original
            else 0.0
        ),
        "novel_vs_train_decoded_and_original_rule_pct": (
            len(novel_vs_both_train) / len(model_only_vs_test_original)
            if model_only_vs_test_original
            else 0.0
        ),

        "seen_in_train_decoded_model_count_pct": (
            model_rule_token_count(seen_in_train_decoded)
            / model_rule_token_count(model_only_vs_test_original)
            if model_rule_token_count(model_only_vs_test_original)
            else 0.0
        ),
        "seen_in_train_original_only_model_count_pct": (
            model_rule_token_count(seen_in_train_original_only)
            / model_rule_token_count(model_only_vs_test_original)
            if model_rule_token_count(model_only_vs_test_original)
            else 0.0
        ),
        "novel_vs_train_decoded_and_original_model_count_pct": (
            model_rule_token_count(novel_vs_both_train)
            / model_rule_token_count(model_only_vs_test_original)
            if model_rule_token_count(model_only_vs_test_original)
            else 0.0
        ),

        # Previews, sorted by model test count descending.
        "model_only_vs_test_gold_original_preview": rule_rows(
            model_only_vs_test_original,
            limit=50,
        ),
        "seen_in_train_decoded_preview": rule_rows(
            seen_in_train_decoded,
            limit=50,
        ),
        "seen_in_train_original_only_preview": rule_rows(
            seen_in_train_original_only,
            limit=50,
        ),
        "novel_vs_train_decoded_and_original_preview": rule_rows(
            novel_vs_both_train,
            limit=50,
        ),
    }



    # ------------------------------------------------------------------
    # Probe / tree metrics
    # ------------------------------------------------------------------

    oov_metrics = token_oov_metrics(train_examples, test_examples)

    rule_support_thresholds = parse_rule_support_thresholds(
        getattr(args, "rule_support_thresholds", "1,2,3,5,10")
    )

    predicted_trees_for_scoring = [
        decode_sequence_labels(
            example.words,
            predict_labels_for_example(
                probe_result["model"],
                probe_result["vocab"],
                example,
                config,
            ),
        )
        for example in test_examples
    ]

    tree_scores = labeled_bracketing_scores(
        [example.gold_tree for example in test_examples],
        predicted_trees_for_scoring,
    )

    oracle_channel_tree_scores = labeled_bracketing_scores(
        [example.gold_tree for example in test_examples],
        [
            decode_sequence_labels(example.words, example.labels)
            for example in test_examples
        ],
    )

    # ------------------------------------------------------------------
    # Save artifacts
    # ------------------------------------------------------------------

    run_dir = output_root / f"{checkpoint_label}_layer_{int(layer):02d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    treebank_paths = {
        "test_gold_original_treebank_path": str(run_dir / "test_gold_original_treebank.jsonl"),
        "test_model_treebank_path": str(run_dir / "test_model_treebank.jsonl"),
    }

    write_treebank_jsonl(
        test_gold_original_trees,
        Path(treebank_paths["test_gold_original_treebank_path"]),
    )
    write_treebank_jsonl(
        test_model_channel_trees,
        Path(treebank_paths["test_model_treebank_path"]),
    )

    grammar_paths = {
        "test_gold_original": save_induced_grammar_bundle(
            test_gold_original_bundle,
            run_dir,
            "test_gold_original",
        ),
        "test_model": save_induced_grammar_bundle(
            test_model_channel_bundle,
            run_dir,
            "test_model",
        ),
    }

    analysis_csv_path = run_dir / "analysis.csv"

    write_compact_analysis_csv(
        analysis_csv_path,
        run=run,
        probe_result=probe_result,
        tree_scores=tree_scores,
        oov_metrics=oov_metrics,
        gold_original_bundle=test_gold_original_bundle,
        decoded_gold_bundle=test_decoded_gold_bundle,
        model_channel_bundle=test_model_channel_bundle,
        rule_support_thresholds=rule_support_thresholds,
    )

    append_model_only_hierarchy_csv(
        analysis_csv_path,
        model_only_train_hierarchy,
    )

    # Save the hierarchy separately as JSON for now.
    model_only_hierarchy_path = run_dir / "model_only_train_hierarchy.json"
    model_only_hierarchy_path.write_text(
        json.dumps(model_only_train_hierarchy, indent=2),
        encoding="utf-8",
    )

    supergrammar_report = None
    supergrammar_paths: Dict[str, str] = {}
    supergrammar_path_value = getattr(args, "supergrammar_path", None)
    if supergrammar_path_value:
        supergrammar_path = Path(supergrammar_path_value).resolve()
        supergrammar_rules = load_supergrammar_structural_rules(supergrammar_path)
        supergrammar_report = supergrammar_comparison_report(
            model_counts=test_model_channel_bundle.structural_rule_counts,
            gold_counts=test_gold_original_bundle.structural_rule_counts,
            supergrammar_rules=supergrammar_rules,
            thresholds=rule_support_thresholds,
        )
        supergrammar_json_path = run_dir / "supergrammar_check.json"
        supergrammar_csv_path = run_dir / "supergrammar_model_only_rules.csv"
        supergrammar_json_path.write_text(
            json.dumps(supergrammar_report, indent=2),
            encoding="utf-8",
        )
        write_supergrammar_rule_csv(
            supergrammar_csv_path,
            model_counts=test_model_channel_bundle.structural_rule_counts,
            gold_counts=test_gold_original_bundle.structural_rule_counts,
            supergrammar_rules=supergrammar_rules,
        )
        supergrammar_paths = {
            "supergrammar_path": str(supergrammar_path),
            "supergrammar_check_path": str(supergrammar_json_path),
            "supergrammar_model_only_rules_csv_path": str(supergrammar_csv_path),
        }

    return {
        "checkpoint_label": checkpoint_label,
        "model_path": str(Path(model_path).resolve()),
        "layer": int(run["layer"]),
        "split_info": run["split_info"],

        "probe": {
            "best_epoch": int(probe_result["best_epoch"]),
            "best_dev_f1": float(probe_result["best_dev_f1"]),
            "test_metrics": compact_metrics(probe_result["test_metrics"]),
            "tree_f1_test": {
                "precision": float(tree_scores["precision"]),
                "recall": float(tree_scores["recall"]),
                "f1": float(tree_scores["f1"]),
            },
            "decoded_gold_tree_f1_test": {
                "precision": float(oracle_channel_tree_scores["precision"]),
                "recall": float(oracle_channel_tree_scores["recall"]),
                "f1": float(oracle_channel_tree_scores["f1"]),
            },
        },

        "grammar_comparison": model_vs_original_gold,
        "grammar_comparison_target": "model_vs_test_gold_original",

        "grammar_comparisons": {
            "model_vs_test_gold_original": model_vs_original_gold,
        },

        "model_only_train_hierarchy": model_only_train_hierarchy,

        "model_only_train_diagnostic": model_only_train_hierarchy,

        "supergrammar_comparison": supergrammar_report,

        "artifacts": {
            **treebank_paths,
            **supergrammar_paths,
            "run_dir": str(run_dir),
            "analysis_csv_path": str(analysis_csv_path),
            "model_only_train_hierarchy_path": str(model_only_hierarchy_path),
            "grammars": grammar_paths,
        },
    }


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def parse_int_list(value: Optional[str]) -> List[int]:
    if value is None or str(value).strip() == "":
        return []
    values: List[int] = []
    for raw in str(value).split(","):
        raw = raw.strip()
        if not raw:
            continue
        parsed = int(raw)
        if parsed <= 0:
            raise ValueError(f"Size values must be positive, got {parsed}.")
        values.append(parsed)
    return values


def discover_checkpoints(model_path: str) -> List[Tuple[str, str]]:
    model_root = Path(model_path)
    checkpoint_dirs = [path for path in model_root.glob("checkpoint-*") if path.is_dir()]

    initial = [path for path in checkpoint_dirs if path.name == "checkpoint-initial"]
    numbered = [path for path in checkpoint_dirs if path.name != "checkpoint-initial"]

    def sort_key(path: Path) -> Tuple[int, str]:
        match = re.search(r"checkpoint-(\d+)$", path.name)
        return (int(match.group(1)) if match else 10**12, path.name)

    discovered = [(str(path), path.name) for path in sorted(initial) + sorted(numbered, key=sort_key)]
    discovered.append((str(model_root), "final"))
    return discovered


def seed_label_from_model_path(model_path: str) -> str:
    path = Path(model_path).resolve()
    for part in reversed(path.parts):
        if part.startswith("seed_"):
            return part
    return "seed_unknown"


def default_output_root_for_model(model_path: str) -> Path:
    path = Path(model_path).resolve()
    seed_label = seed_label_from_model_path(str(path))
    parts = list(path.parts)
    if seed_label in parts:
        seed_index = parts.index(seed_label)
        base_dir = Path(*parts[:seed_index])
        if base_dir.name.lower() in {"gpt2", "models", "model", "checkpoints"}:
            base_dir = base_dir.parent
    else:
        base_dir = path.parent
    return base_dir / "probing_grammar_induction" / seed_label


def model_source_label(args: argparse.Namespace) -> str:
    return "random_baseline" if bool(args.run_baseline) else "trained_model"


def size_run_label(args: argparse.Namespace) -> str:
    return (
        f"train_{int(args.train_trees):06d}"
        f"_dev_{int(args.dev_trees):06d}"
        f"_test_{int(args.fixed_test_size):06d}"
    )


def resolved_output_root(args: argparse.Namespace, *, include_size: bool = False) -> Path:
    base = output_dir_with_mode_suffix(Path(args.output_dir).resolve(), str(args.run_mode))
    root = base / model_source_label(args)
    if include_size:
        root = root / size_run_label(args)
    return root


def output_dir_with_mode_suffix(path: Path, run_mode: str) -> Path:
    if run_mode == "individual":
        return path
    suffix = f"_{run_mode}"
    if path.name.endswith(suffix):
        return path
    if not path.name:
        return path / run_mode
    return path.with_name(f"{path.name}{suffix}")


def infer_num_layers(args: argparse.Namespace, model_path: str) -> int:
    device = resolve_device(args.device)
    model = load_model(model_path, local_files_only=args.local_files_only, device=device)
    n_layers = validate_requested_layer(model, int(getattr(model.config, "n_layer")))
    del model
    return int(n_layers)


def apply_default_probe_paths(args: argparse.Namespace) -> argparse.Namespace:
    if getattr(args, "train_trees_path", None) is None:
        args.train_trees_path = str(DEFAULT_TRAIN_TREES_PATH)
    if getattr(args, "dev_trees_path", None) is None:
        args.dev_trees_path = str(DEFAULT_DEV_TREES_PATH)
    if getattr(args, "test_trees_path", None) is None:
        args.test_trees_path = str(DEFAULT_TEST_TREES_PATH)
    if getattr(args, "model_path", None) is None:
        args.model_path = str(DEFAULT_MODEL_PATH)
    if getattr(args, "output_dir", None) is None:
        args.output_dir = str(default_output_root_for_model(args.model_path))
    return args


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_individual(args: argparse.Namespace) -> Dict[str, object]:
    output_root = resolved_output_root(args)
    layer = int(args.layer)
    result = compare_for_trained_probe(
        args,
        model_path=args.model_path,
        tokenizer_path=args.model_path,
        layer=layer,
        checkpoint_label="final",
        output_root=output_root,
    )
    summary = {
        "run_mode": "individual",
        "model_source": model_source_label(args),
        "model_path": str(Path(args.model_path).resolve()),
        "output_root": str(output_root),
        "train_trees": int(args.train_trees),
        "dev_trees": int(args.dev_trees),
        "fixed_test_size": int(args.fixed_test_size),
        "result": result,
    }
    save_json(output_root / f"individual_layer_{layer:02d}.json", summary)
    return summary


def run_layers(args: argparse.Namespace) -> Dict[str, object]:
    output_root = resolved_output_root(args)
    return run_layers_at_output_root(args, output_root)


def run_checkpoints(args: argparse.Namespace) -> Dict[str, object]:
    output_root = resolved_output_root(args)
    rows: List[Dict[str, object]] = []
    split_override = prepare_tree_splits(args)
    for model_path, checkpoint_label in discover_checkpoints(args.model_path):
        rows.append(
            compare_for_trained_probe(
                args,
                model_path=model_path,
                tokenizer_path=args.model_path,
                layer=int(args.layer),
                checkpoint_label=checkpoint_label,
                output_root=output_root,
                split_override=split_override,
            )
        )
    summary = {
        "run_mode": "checkpoints",
        "model_source": model_source_label(args),
        "model_root": str(Path(args.model_path).resolve()),
        "output_root": str(output_root),
        "layer": int(args.layer),
        "train_trees": int(args.train_trees),
        "dev_trees": int(args.dev_trees),
        "fixed_test_size": int(args.fixed_test_size),
        "results": rows,
    }
    save_json(output_root / f"checkpoints_layer_{int(args.layer):02d}.json", summary)
    return summary


def run_one(args: argparse.Namespace) -> Dict[str, object]:
    if args.run_mode == "individual":
        return run_individual(args)
    if args.run_mode == "layers":
        return run_layers(args)
    if args.run_mode == "checkpoints":
        return run_checkpoints(args)
    raise ValueError(f"Unsupported run_mode={args.run_mode!r}")


def run_size_sweep(args: argparse.Namespace) -> Dict[str, object]:
    train_sizes = parse_int_list(args.train_tree_sizes) or [int(args.train_trees)]
    dev_sizes = parse_int_list(args.dev_tree_sizes) or [int(args.dev_trees)]
    test_sizes = parse_int_list(args.eval_tree_sizes) or [int(args.fixed_test_size)]

    base_output_dir = Path(args.output_dir).resolve()
    rows: List[Dict[str, object]] = []
    for train_size in train_sizes:
        for dev_size in dev_sizes:
            for test_size in test_sizes:
                run_args = copy(args)
                run_args.train_trees = int(train_size)
                run_args.dev_trees = int(dev_size)
                run_args.fixed_test_size = int(test_size)
                run_args.output_dir = str(base_output_dir)
                run_args.force_size_subdir = True

                size_root = resolved_output_root(run_args, include_size=True)
                run_args.output_dir = str(size_root.parent.parent)
                summary = run_one_with_output_root(run_args, size_root)
                rows.append(
                    {
                        "train_trees": int(train_size),
                        "dev_trees": int(dev_size),
                        "fixed_test_size": int(test_size),
                        "run_mode": summary["run_mode"],
                        "model_source": model_source_label(run_args),
                        "output_root": str(size_root),
                        "summary_path": str(size_root / summary_filename(summary)),
                        "summary": summary,
                    }
                )

    sweep_root = output_dir_with_mode_suffix(base_output_dir, str(args.run_mode)) / model_source_label(args)
    payload = {
        "run_mode": "size_sweep",
        "model_source": model_source_label(args),
        "model_path": str(Path(args.model_path).resolve()),
        "train_tree_sizes": train_sizes,
        "dev_tree_sizes": dev_sizes,
        "eval_tree_sizes": test_sizes,
        "results": rows,
    }
    save_json(sweep_root / "sweep_summary.json", payload)
    return payload


def summary_filename(summary: Dict[str, object]) -> str:
    run_mode = str(summary["run_mode"])
    if run_mode == "individual":
        layer = int(summary["result"]["layer"])
        return f"individual_layer_{layer:02d}.json"
    if run_mode == "layers":
        return "layers_summary.json"
    if run_mode == "checkpoints":
        return f"checkpoints_layer_{int(summary['layer']):02d}.json"
    return f"{run_mode}_summary.json"


def run_one_with_output_root(args: argparse.Namespace, output_root: Path) -> Dict[str, object]:
    old_output_dir = args.output_dir
    try:
        args.output_dir = str(output_root.parent.parent)
        if args.run_mode == "individual":
            return run_individual_at_output_root(args, output_root)
        if args.run_mode == "layers":
            return run_layers_at_output_root(args, output_root)
        if args.run_mode == "checkpoints":
            return run_checkpoints_at_output_root(args, output_root)
        raise ValueError(f"Unsupported run_mode={args.run_mode!r}")
    finally:
        args.output_dir = old_output_dir

def run_individual_at_output_root(args: argparse.Namespace, output_root: Path) -> Dict[str, object]:
    layer = int(args.layer)
    result = compare_for_trained_probe(
        args,
        model_path=args.model_path,
        tokenizer_path=args.model_path,
        layer=layer,
        checkpoint_label="final",
        output_root=output_root,
    )
    summary = {
        "run_mode": "individual",
        "model_source": model_source_label(args),
        "model_path": str(Path(args.model_path).resolve()),
        "output_root": str(output_root),
        "train_trees": int(args.train_trees),
        "dev_trees": int(args.dev_trees),
        "fixed_test_size": int(args.fixed_test_size),
        "result": result,
    }
    save_json(output_root / f"individual_layer_{layer:02d}.json", summary)
    return summary


def run_layers_at_output_root(args: argparse.Namespace, output_root: Path) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    split_override = prepare_tree_splits(args)
    n_layers = infer_num_layers(args, args.model_path)
    for layer in range(1, n_layers + 1):
        rows.append(
            compare_for_trained_probe(
                args,
                model_path=args.model_path,
                tokenizer_path=args.model_path,
                layer=layer,
                checkpoint_label="final",
                output_root=output_root,
                split_override=split_override,
            )
        )
    summary = {
        "run_mode": "layers",
        "model_source": model_source_label(args),
        "model_path": str(Path(args.model_path).resolve()),
        "output_root": str(output_root),
        "num_layers": n_layers,
        "train_trees": int(args.train_trees),
        "dev_trees": int(args.dev_trees),
        "fixed_test_size": int(args.fixed_test_size),
        "results": rows,
    }
    save_json(output_root / "layers_summary.json", summary)
    return summary


def run_checkpoints_at_output_root(args: argparse.Namespace, output_root: Path) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    split_override = prepare_tree_splits(args)
    for model_path, checkpoint_label in discover_checkpoints(args.model_path):
        rows.append(
            compare_for_trained_probe(
                args,
                model_path=model_path,
                tokenizer_path=args.model_path,
                layer=int(args.layer),
                checkpoint_label=checkpoint_label,
                output_root=output_root,
                split_override=split_override,
            )
        )
    summary = {
        "run_mode": "checkpoints",
        "model_source": model_source_label(args),
        "model_root": str(Path(args.model_path).resolve()),
        "output_root": str(output_root),
        "layer": int(args.layer),
        "train_trees": int(args.train_trees),
        "dev_trees": int(args.dev_trees),
        "fixed_test_size": int(args.fixed_test_size),
        "results": rows,
    }
    save_json(output_root / f"checkpoints_layer_{int(args.layer):02d}.json", summary)
    return summary


def print_summary(summary: Dict[str, object]) -> None:
    run_mode = str(summary["run_mode"])
    print(f"run_mode={run_mode}")
    if run_mode == "size_sweep":
        print(f"model_source={summary['model_source']}")
        print(f"num_runs={len(summary['results'])}")
        for row in summary["results"]:
            run_summary = row["summary"]
            result = run_summary.get("result")
            if result is None:
                print(
                    f"train={row['train_trees']} dev={row['dev_trees']} "
                    f"test={row['fixed_test_size']} output_root={row['output_root']}"
                )
                continue
            overlap = result["grammar_comparison"]["rule_overlap"]
            print(
                f"train={row['train_trees']} dev={row['dev_trees']} "
                f"test={row['fixed_test_size']} "
                f"probe_test_f1={float(result['probe']['test_metrics']['f1']):.4f} "
                f"grammar_rule_f1_model_vs_gold_original={float(overlap['f1']):.4f} "
                f"results_dir={result['artifacts']['run_dir']}"
            )
        return
    if run_mode == "individual":
        result = summary["result"]
        overlap = result["grammar_comparison"]["rule_overlap"]
        print(f"layer={int(result['layer'])}")
        print(f"probe_test_f1={float(result['probe']['test_metrics']['f1']):.4f}")
        print(f"grammar_rule_f1_model_vs_gold_original={float(overlap['f1']):.4f}")
        print(f"analysis_csv={result['artifacts']['analysis_csv_path']}")
        print(f"results_dir={result['artifacts']['run_dir']}")
        return
    if run_mode == "layers":
        for row in summary["results"]:
            overlap = row["grammar_comparison"]["rule_overlap"]
            print(
                f"layer={int(row['layer'])} "
                f"probe_test_f1={float(row['probe']['test_metrics']['f1']):.4f} "
                f"grammar_rule_f1_model_vs_gold_original={float(overlap['f1']):.4f} "
                f"analysis_csv={row['artifacts']['analysis_csv_path']}"
            )
        return
    if run_mode == "checkpoints":
        for row in summary["results"]:
            overlap = row["grammar_comparison"]["rule_overlap"]
            print(
                f"checkpoint={row['checkpoint_label']} "
                f"probe_test_f1={float(row['probe']['test_metrics']['f1']):.4f} "
                f"grammar_rule_f1_model_vs_gold_original={float(overlap['f1']):.4f} "
                f"analysis_csv={row['artifacts']['analysis_csv_path']}"
            )





def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare grammars induced from original gold, Vilares-decoded gold, and probe-predicted trees."
    )
    parser.add_argument("--run-mode", choices=["individual", "layers", "checkpoints"], default="individual")
    parser.add_argument("--layer", type=int, default=12, help="Used by individual/checkpoints modes.")
    parser.add_argument("--train-trees", type=int, default=100)
    parser.add_argument("--dev-trees", type=int, default=500)
    parser.add_argument("--fixed-test-size", type=int, default=10000)
    parser.add_argument("--train-tree-sizes", default=None, help="Comma-separated train sizes for a size sweep.")
    parser.add_argument("--dev-tree-sizes", default=None, help="Comma-separated dev sizes for a size sweep.")
    parser.add_argument("--eval-tree-sizes", default=None, help="Comma-separated test/eval sizes for a size sweep.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--extract-batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--rule-support-thresholds",
        default="1,2,3,5,10,20,50",
        help="Comma-separated support thresholds for compact grammar overlap metrics.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--train-trees-path", default=None)
    parser.add_argument("--dev-trees-path", default=None)
    parser.add_argument("--test-trees-path", default=None)
    parser.add_argument(
        "--supergrammar-path",
        default=None,
        help="Optional larger grammar.txt/grammar_mle.txt for model-only rule coverage checks.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Base output directory. Defaults to MODEL_PARENT/probing_grammar_induction/seed_X.",
    )
    parser.add_argument("--model-init-seed", type=int, default=1)
    parser.add_argument("--run-baseline", "--run_baseline", dest="run_baseline", nargs="?", const=True, default=False, type=parse_bool)
    parser.add_argument("--trained-model", dest="run_baseline", action="store_false")

    args = parser.parse_args()
    args = apply_default_probe_paths(args)

    if args.train_tree_sizes or args.dev_tree_sizes or args.eval_tree_sizes:
        summary = run_size_sweep(args)
    else:
        summary = run_one(args)
    print_summary(summary)


if __name__ == "__main__":
    main()
