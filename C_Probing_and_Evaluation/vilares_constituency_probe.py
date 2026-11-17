"""Vilares-style constituency probe implementation.

This module implements the tree-to-sequence-label encoding, decoding back to
constituency trees, hidden-state extraction, linear token-level probe training,
tree-level metrics, random-initialization baselines, and diagnostic grouped
evaluations.
"""

from __future__ import annotations
import argparse
import json
import math
import random
import re
import tempfile
import unittest
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset



END_C_LABEL = "<END>"
UNKNOWN_LABEL = "UNK"
ATOMIC_DELIM = "|||"
UNARY_DELIM = ">"

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_TREES_PATH = None
DEFAULT_MODEL_PATH = None
DEFAULT_TRAIN_TREES_PATH = None
DEFAULT_TEST_TREES_PATH = None
DEFAULT_DEV_TREES_PATH = None



@dataclass
class SequenceLabel:
    """Word-level Vilares label."""

    n: int
    c: str
    u: Tuple[str, ...] = ()

    def atomic_string(self) -> str:
        unary_str = UNARY_DELIM.join(self.u)
        return f"{self.n}{ATOMIC_DELIM}{self.c}{ATOMIC_DELIM}{unary_str}"

    @staticmethod
    def from_atomic_string(value: str) -> "SequenceLabel":
        n_str, c_value, unary_str = value.split(ATOMIC_DELIM)
        unary = tuple(x for x in unary_str.split(UNARY_DELIM) if x)
        return SequenceLabel(n=int(n_str), c=c_value, u=unary)


@dataclass
class TreeNode:
    """Simple labeled constituency tree."""

    label: str
    children: List[Union["TreeNode", str]]
    start: int = 0
    end: int = 0
    level: int = 0

    def is_leaf(self) -> bool:
        return len(self.children) == 1 and isinstance(self.children[0], str)

    def word(self) -> str:
        if not self.is_leaf():
            raise ValueError("Only leaf/preterminal nodes directly dominate a word.")
        value = str(self.children[0])
        if value == "__LPAR__":
            return "("
        if value == "__RPAR__":
            return ")"
        return value

    def words(self) -> List[str]:
        out: List[str] = []
        for child in self.children:
            if isinstance(child, str):
                out.append(child)
            else:
                out.extend(child.words())
        return out

    def pretty(self) -> str:
        if self.is_leaf():
            return f"({self.label} {self.word()})"
        child_str = " ".join(child if isinstance(child, str) else child.pretty() for child in self.children)
        return f"({self.label} {child_str})"

    def clone(self) -> "TreeNode":
        children: List[Union[TreeNode, str]] = []
        for child in self.children:
            children.append(child if isinstance(child, str) else child.clone())
        return TreeNode(self.label, children, self.start, self.end, self.level)


@dataclass(frozen=True)
class GrammarRule:
    lhs: str
    rhs: Tuple[str, ...]
    rhs_is_terminal: Tuple[bool, ...]
    prob: float


@dataclass
class SortedTreeGroup:
    group_type: str
    group_label: str
    tree_strings: List[str]
    metadata: Dict[str, object]


def tokenize_bracket_tree(text: str) -> List[str]:
    return re.findall(r"\(|\)|[^\s()]+", text)


def normalize_repo_tree_string(tree_line: str) -> str:
    normalized = re.sub(r"\(LPAR\s+\(\)", "(LPAR (__LPAR__)", tree_line)
    normalized = re.sub(r"\(RPAR\s+\)", "(RPAR (__RPAR__))", normalized)
    return normalized


def parse_bracket_tree(text: str) -> TreeNode:
    """Parse a Penn-style bracketed tree."""

    toks = tokenize_bracket_tree(normalize_repo_tree_string(text.strip()))

    def parse_at(idx: int) -> Tuple[TreeNode, int]:
        if idx >= len(toks) or toks[idx] != "(":
            raise ValueError("Expected '(' while parsing bracketed tree.")
        label = toks[idx + 1]
        idx += 2
        children: List[Union[TreeNode, str]] = []
        while idx < len(toks) and toks[idx] != ")":
            if toks[idx] == "(":
                child, idx = parse_at(idx)
                children.append(child)
            else:
                children.append(toks[idx])
                idx += 1
        if idx >= len(toks) or toks[idx] != ")":
            raise ValueError("Unbalanced bracket tree.")
        return TreeNode(label=label, children=children), idx + 1

    node, end = parse_at(0)
    if end != len(toks):
        raise ValueError("Tree parser did not consume full token stream.")
    node = prune_empty_nodes(node)
    if node is None:
        raise ValueError("Tree collapsed to empty after pruning empty nodes.")
    annotate_spans(node)
    return node


def prune_empty_nodes(node: TreeNode) -> Optional[TreeNode]:
    """Drop empty nonterminal nodes created by epsilon productions."""

    kept_children: List[Union[TreeNode, str]] = []
    for child in node.children:
        if isinstance(child, str):
            kept_children.append(child)
            continue
        pruned = prune_empty_nodes(child)
        if pruned is not None:
            kept_children.append(pruned)
    node.children = kept_children
    if not node.children:
        return None
    return node


def load_probabilistic_grammar(grammar_path: Path) -> Dict[Tuple[str, Tuple[str, ...], Tuple[bool, ...]], float]:
    rule_probs: Dict[Tuple[str, Tuple[str, ...], Tuple[bool, ...]], float] = {}
    line_re = re.compile(r"^\s*(?P<lhs>\S+)\s*->\s*\[(?P<rhs>.*?)\]\s*:\s*(?P<prob>[0-9eE+\-.]+)\s*$")
    with grammar_path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            match = line_re.match(line)
            if not match:
                raise ValueError(f"Cannot parse grammar line {line_no} in {grammar_path}: {line}")
            lhs = match.group("lhs")
            rhs_items = [tok for tok in match.group("rhs").split() if tok]
            rhs: List[str] = []
            rhs_is_terminal: List[bool] = []
            for item in rhs_items:
                if item.startswith("_"):
                    rhs.append(item[1:])
                    rhs_is_terminal.append(True)
                else:
                    rhs.append(item)
                    rhs_is_terminal.append(False)
            key = (lhs, tuple(rhs), tuple(rhs_is_terminal))
            rule_probs[key] = float(match.group("prob"))
    if not rule_probs:
        raise ValueError(f"No grammar rules loaded from {grammar_path}")
    return rule_probs


def infer_grammar_path(args: argparse.Namespace) -> Path:
    explicit = getattr(args, "grammar_path", None)
    if explicit:
        path = Path(str(explicit)).expanduser().resolve()
        if not path.exists():
            raise ValueError(f"grammar file does not exist: {path}")
        return path

    ref_text = " ".join(
        [
            str(getattr(args, "model_path", "")),
            str(getattr(args, "train_trees_path", "")),
            str(getattr(args, "dev_trees_path", "")),
            str(getattr(args, "test_trees_path", "")),
        ]
    ).lower()
    candidate_names = ["grammar.txt"]

    ref_paths = [
        Path(str(getattr(args, "train_trees_path", ""))),
        Path(str(getattr(args, "dev_trees_path", ""))),
        Path(str(getattr(args, "test_trees_path", ""))),
        Path(str(getattr(args, "model_path", ""))),
    ]
    for ref_path in ref_paths:
        for parent in [ref_path.parent, *ref_path.parents]:
            for candidate_name in candidate_names:
                candidate = parent / candidate_name
                if candidate.exists():
                    return candidate.resolve()

    for candidate_name in candidate_names:
        candidate = REPO_ROOT / "data" / candidate_name
        if candidate.exists():
            return candidate.resolve()
    raise ValueError("Could not infer a grammar file. Pass --grammar-path explicitly.")


def annotate_spans(node: TreeNode, word_offset: int = 0, level: int = 0) -> int:
    """Populate span and level metadata in-place."""

    node.level = level
    cur = word_offset
    for child in node.children:
        if isinstance(child, str):
            cur += 1
        else:
            cur = annotate_spans(child, cur, level + 1)
    node.start = word_offset
    node.end = cur
    return cur


def iter_leaf_paths(node: TreeNode) -> List[Tuple[str, List[TreeNode]]]:
    """Return ``(word, path_from_root_to_leaf_parent)`` for each word."""

    result: List[Tuple[str, List[TreeNode]]] = []

    def visit(cur: TreeNode, path: List[TreeNode]) -> None:
        if cur.is_leaf():
            result.append((cur.word(), path + [cur]))
            return
        for child in cur.children:
            if isinstance(child, str):
                raise ValueError("Words must appear under a labeled preterminal.")
            visit(child, path + [cur])

    visit(node, [])
    return result


def unary_chain_for_path(path: Sequence[TreeNode]) -> Tuple[str, ...]:
    """Extract the leaf unary chain for a word path."""

    if not path:
        return ()

    chain: List[str] = []
    for node in reversed(path):
        n_children = len(node.children)
        if n_children == 1:
            chain.append(node.label)
        else:
            break
    chain.reverse()
    return tuple(chain)


def shared_ancestor_info(left_path: Sequence[TreeNode], right_path: Sequence[TreeNode]) -> Tuple[int, str]:
    """Return ``(shared_levels, lca_label)`` for adjacent words."""

    shared = 0
    lca_label = END_C_LABEL
    for left_node, right_node in zip(left_path, right_path):
        if left_node.label != right_node.label or left_node.start != right_node.start or left_node.end != right_node.end:
            break
        shared += 1
        lca_label = left_node.label
    return shared, lca_label


def tree_to_sequence_labels(tree: TreeNode) -> Tuple[List[str], List[SequenceLabel]]:
    """Encode a tree into Vilares sequence labels."""

    annotate_spans(tree)
    leaf_paths = iter_leaf_paths(tree)
    words = [word for word, _ in leaf_paths]
    if not words:
        return [], []

    labels: List[SequenceLabel] = []
    prev_shared = 0
    for idx, (_, path) in enumerate(leaf_paths):
        unary = unary_chain_for_path(path)
        if idx + 1 < len(leaf_paths):
            shared, lca_label = shared_ancestor_info(path, leaf_paths[idx + 1][1])
        else:
            shared, lca_label = 0, END_C_LABEL
        labels.append(SequenceLabel(n=shared - prev_shared, c=lca_label, u=unary))
        prev_shared = shared
    return words, labels


def labels_to_absolute_shared_levels(labels: Sequence[SequenceLabel]) -> List[int]:
    shared: List[int] = []
    cur = 0
    for label in labels:
        cur += int(label.n)
        if cur < 0:
            cur = 0
        shared.append(cur)
    if shared:
        shared[-1] = 0
    return shared


def build_word_leaf(word: str, unary_chain: Sequence[str]) -> TreeNode:
    """Wrap a word with its unary chain."""

    if unary_chain:
        cur = TreeNode(label=unary_chain[-1], children=[word])
        for label in reversed(unary_chain[:-1]):
            cur = TreeNode(label=label, children=[cur])
        annotate_spans(cur)
        return cur
    leaf = TreeNode(label=UNKNOWN_LABEL, children=[word])
    annotate_spans(leaf)
    return leaf


def build_unlabeled_skeleton(words: Sequence[str], labels: Sequence[SequenceLabel]) -> TreeNode:
    """Build an unlabeled tree skeleton from absolute shared depths."""

    shared = labels_to_absolute_shared_levels(labels)
    word_leaves = [build_word_leaf(word, label.u) for word, label in zip(words, labels)]

    def build_segment(left: int, right: int, level: int) -> TreeNode:
        node = TreeNode(label="", children=[], level=level)
        idx = left
        while idx <= right:
            end = idx
            while end < right and shared[end] >= level + 1:
                end += 1
            if idx == end:
                node.children.append(word_leaves[idx].clone())
            else:
                node.children.append(build_segment(idx, end, level + 1))
            idx = end + 1
        annotate_spans(node, left, level)
        return node

    if not words:
        raise ValueError("Cannot decode an empty sentence.")
    root = build_segment(0, len(words) - 1, 1)
    annotate_spans(root, 0, 1)
    return root


def iter_internal_nodes(node: TreeNode) -> Iterable[TreeNode]:
    for child in node.children:
        if isinstance(child, TreeNode):
            yield from iter_internal_nodes(child)
    if any(isinstance(child, TreeNode) for child in node.children):
        yield node


def assign_constituent_labels(
    root: TreeNode, labels: Sequence[SequenceLabel], shared: Sequence[int]
) -> None:
    """Assign constituent labels from ``c_i`` with first-conflict wins."""

    candidate_nodes: List[TreeNode] = [node for node in iter_internal_nodes(root) if node.label == ""]
    candidate_nodes.sort(key=lambda n: (n.level, n.start, n.end))

    for idx, label in enumerate(labels[:-1]):
        depth = shared[idx]
        if depth <= 0 or label.c == END_C_LABEL:
            continue
        selected: Optional[TreeNode] = None
        for node in candidate_nodes:
            if node.level != depth:
                continue
            if node.start <= idx and node.end >= idx + 2:
                selected = node
                break
        if selected is None:
            continue
        if selected.label == "":
            selected.label = label.c


def cleanup_decoded_tree(node: TreeNode) -> TreeNode:
    """Remove empty unary levels and ensure all remaining nodes are labeled."""

    cleaned_children: List[Union[TreeNode, str]] = []
    for child in node.children:
        if isinstance(child, str):
            cleaned_children.append(child)
        else:
            cleaned_children.append(cleanup_decoded_tree(child))
    node.children = cleaned_children

    if node.label == "" and len(node.children) == 1 and isinstance(node.children[0], TreeNode):
        return node.children[0]

    if node.label == "":
        node.label = UNKNOWN_LABEL
    annotate_spans(node)
    return node


def decode_sequence_labels(words: Sequence[str], labels: Sequence[SequenceLabel]) -> TreeNode:
    """Decode predicted labels back into a labeled constituency tree."""

    if len(words) != len(labels):
        raise ValueError("Need exactly one label per word.")
    shared = labels_to_absolute_shared_levels(labels)
    root = build_unlabeled_skeleton(words, labels)
    assign_constituent_labels(root, labels, shared)
    decoded = cleanup_decoded_tree(root)
    if decoded.label == UNKNOWN_LABEL and len(decoded.children) == 1 and isinstance(decoded.children[0], TreeNode):
        decoded = decoded.children[0]
    if decoded.label == UNKNOWN_LABEL:
        decoded.label = "ROOT"
    annotate_spans(decoded)
    return decoded


def sequence_labels_to_strings(labels: Sequence[SequenceLabel]) -> List[str]:
    return [label.atomic_string() for label in labels]


def extract_labeled_spans(
    node: TreeNode, *, ignore_full_span: bool = True, ignore_preterminals: bool = True
) -> Counter:
    """Extract labeled spans for bracketing evaluation."""

    annotate_spans(node)
    spans: Counter = Counter()

    def visit(cur: TreeNode, is_root: bool) -> None:
        child_nodes = [child for child in cur.children if isinstance(child, TreeNode)]
        if child_nodes:
            span_len = cur.end - cur.start
            if not (ignore_full_span and is_root) and not (ignore_preterminals and span_len <= 1):
                spans[(cur.start, cur.end, cur.label)] += 1
            for child in child_nodes:
                visit(child, False)

    visit(node, True)
    return spans


def labeled_bracketing_scores(
    gold_trees: Sequence[TreeNode],
    pred_trees: Sequence[TreeNode],
    *,
    ignore_full_span: bool = True,
    ignore_preterminals: bool = True,
) -> Dict[str, float]:
    if len(gold_trees) != len(pred_trees):
        raise ValueError("Gold and predicted tree lists must have equal length.")

    tp = 0
    pred_total = 0
    gold_total = 0
    for gold, pred in zip(gold_trees, pred_trees):
        gold_spans = extract_labeled_spans(gold, ignore_full_span=ignore_full_span, ignore_preterminals=ignore_preterminals)
        pred_spans = extract_labeled_spans(pred, ignore_full_span=ignore_full_span, ignore_preterminals=ignore_preterminals)
        gold_total += sum(gold_spans.values())
        pred_total += sum(pred_spans.values())
        for key, count in pred_spans.items():
            tp += min(count, gold_spans.get(key, 0))

    precision = tp / pred_total if pred_total else 0.0
    recall = tp / gold_total if gold_total else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def labeled_f1_by_span_length(
    gold_trees: Sequence[TreeNode], pred_trees: Sequence[TreeNode], max_span_len: Optional[int] = None
) -> Dict[int, Dict[str, float]]:
    """Break down labeled F1 by span length."""

    per_len_counts: Dict[int, Dict[str, int]] = defaultdict(lambda: {"tp": 0, "pred": 0, "gold": 0})
    for gold, pred in zip(gold_trees, pred_trees):
        gold_spans = extract_labeled_spans(gold)
        pred_spans = extract_labeled_spans(pred)
        for span, count in gold_spans.items():
            length = span[1] - span[0]
            if max_span_len is not None and length > max_span_len:
                continue
            per_len_counts[length]["gold"] += count
        for span, count in pred_spans.items():
            length = span[1] - span[0]
            if max_span_len is not None and length > max_span_len:
                continue
            per_len_counts[length]["pred"] += count
            per_len_counts[length]["tp"] += min(count, gold_spans.get(span, 0))

    result: Dict[int, Dict[str, float]] = {}
    for length, counts in sorted(per_len_counts.items()):
        precision = counts["tp"] / counts["pred"] if counts["pred"] else 0.0
        recall = counts["tp"] / counts["gold"] if counts["gold"] else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        result[length] = {"precision": precision, "recall": recall, "f1": f1}
    return result


def canonicalize_sentence(text: str) -> str:
    text = str(text).strip()
    if not text:
        return text
    toks = re.findall(r"\(|\)|\+|\*|[^\s()+*]+", text)
    return " ".join(toks)


def load_tree_file(tree_path: str) -> List[str]:
    path = Path(tree_path)
    if not path.exists():
        raise ValueError(f"tree file does not exist: {path}")
    trees: List[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("{"):
            obj = json.loads(line)
            trees.append(str(obj["tree"]).strip())
        else:
            trees.append(line)
    if not trees:
        raise ValueError(f"tree file is empty: {path}")
    return trees


def select_split_trees(
    trees: Sequence[str],
    *,
    requested_count: int,
    seed: int,
    split_name: str,
) -> List[str]:
    if requested_count <= 0:
        raise ValueError(f"{split_name} requested_count must be > 0 but got {requested_count}.")
    pool = list(trees)
    random.Random(seed).shuffle(pool)
    selected = pool[: min(requested_count, len(pool))]
    if not selected:
        raise ValueError(f"{split_name} split is empty after selection.")
    return selected


def prepare_tree_splits(args: argparse.Namespace) -> Tuple[List[str], List[str], List[str], Dict[str, object]]:
    train_tree_strings = select_split_trees(
        load_tree_file(args.train_trees_path),
        requested_count=int(args.train_trees),
        seed=int(args.seed),
        split_name="train",
    )
    dev_tree_strings = select_split_trees(
        load_tree_file(args.dev_trees_path),
        requested_count=int(args.dev_trees),
        seed=int(args.seed) + 1,
        split_name="dev",
    )
    test_tree_strings = select_split_trees(
        load_tree_file(args.test_trees_path),
        requested_count=int(args.fixed_test_size),
        seed=int(args.seed) + 2,
        split_name="test",
    )
    split_info = {
        "split_strategy": "explicit_tree_files",
        "n_train": len(train_tree_strings),
        "n_dev": len(dev_tree_strings),
        "n_test": len(test_tree_strings),
        "train_trees_path": str(Path(args.train_trees_path).resolve()),
        "dev_trees_path": str(Path(args.dev_trees_path).resolve()),
        "test_trees_path": str(Path(args.test_trees_path).resolve()),
    }
    return train_tree_strings, dev_tree_strings, test_tree_strings, split_info


def tree_max_depth(node: TreeNode) -> int:
    child_nodes = [child for child in node.children if isinstance(child, TreeNode)]
    if not child_nodes:
        return 1
    return 1 + max(tree_max_depth(child) for child in child_nodes)


def group_tree_strings_by_depth(tree_strings: Sequence[str]) -> Dict[int, List[str]]:
    groups: Dict[int, List[str]] = defaultdict(list)
    for tree_str in tree_strings:
        groups[tree_max_depth(parse_bracket_tree(tree_str))].append(tree_str)
    return dict(sorted(groups.items()))


def tree_rule_signature(node: TreeNode) -> Tuple[str, Tuple[str, ...], Tuple[bool, ...]]:
    rhs: List[str] = []
    rhs_is_terminal: List[bool] = []
    for child in node.children:
        if isinstance(child, TreeNode):
            rhs.append(child.label)
            rhs_is_terminal.append(False)
        else:
            rhs.append(child)
            rhs_is_terminal.append(True)
    return node.label, tuple(rhs), tuple(rhs_is_terminal)


def tree_log_probability(
    node: TreeNode,
    rule_probs: Dict[Tuple[str, Tuple[str, ...], Tuple[bool, ...]], float],
) -> float:
    key = tree_rule_signature(node)
    prob = rule_probs.get(key)
    if prob is None:
        raise ValueError(
            "Tree uses a grammar rule that was not found: "
            f"{key[0]} -> {list(key[1])} with terminal_mask={list(key[2])}"
        )
    log_prob = math.log(max(prob, 1e-300))
    for child in node.children:
        if isinstance(child, TreeNode):
            log_prob += tree_log_probability(child, rule_probs)
    return log_prob


def safe_exp(log_value: float) -> float:
    return math.exp(max(log_value, -745.0))


def make_depth_sorted_groups(tree_strings: Sequence[str]) -> List[SortedTreeGroup]:
    groups: List[SortedTreeGroup] = []
    for depth, members in group_tree_strings_by_depth(tree_strings).items():
        groups.append(
            SortedTreeGroup(
                group_type="byDepth",
                group_label=f"depth_{depth}",
                tree_strings=list(members),
                metadata={"depth_group": int(depth)},
            )
        )
    return groups


def make_probability_sorted_groups(
    tree_strings: Sequence[str],
    *,
    grammar_path: Path,
    n_groups: int,
) -> List[SortedTreeGroup]:
    if n_groups <= 0:
        raise ValueError(f"probability group count must be > 0 but got {n_groups}")
    rule_probs = load_probabilistic_grammar(grammar_path)
    scored: List[Tuple[str, float]] = []
    for tree_str in tree_strings:
        log_prob = tree_log_probability(parse_bracket_tree(tree_str), rule_probs)
        scored.append((tree_str, log_prob))
    scored.sort(key=lambda item: item[1], reverse=True)

    actual_groups = min(int(n_groups), len(scored))
    if actual_groups <= 0:
        return []

    groups: List[SortedTreeGroup] = []
    start = 0
    for group_idx in range(actual_groups):
        remaining_items = len(scored) - start
        remaining_groups = actual_groups - group_idx
        take = max(1, math.ceil(remaining_items / remaining_groups))
        chunk = scored[start : start + take]
        start += take
        log_values = [value for _, value in chunk]
        groups.append(
            SortedTreeGroup(
                group_type="byProb",
                group_label=f"prob_group_{group_idx + 1}",
                tree_strings=[tree_str for tree_str, _ in chunk],
                metadata={
                    "prob_group": f"prob_group_{group_idx + 1}",
                    "rank_group": int(group_idx + 1),
                    "min_log_probability": float(min(log_values)),
                    "max_log_probability": float(max(log_values)),
                    "mean_log_probability": float(sum(log_values) / len(log_values)),
                    "min_probability": float(safe_exp(min(log_values))),
                    "max_probability": float(safe_exp(max(log_values))),
                    "mean_probability": float(safe_exp(sum(log_values) / len(log_values))),
                },
            )
        )
    return groups


def build_sorted_tree_groups(args: argparse.Namespace, tree_strings: Sequence[str]) -> Tuple[List[SortedTreeGroup], Dict[str, object]]:
    sorted_mode = str(getattr(args, "sorted_tree_groups", "byDepth"))
    if sorted_mode == "byDepth":
        groups = make_depth_sorted_groups(tree_strings)
        return groups, {"sorted_tree_groups": "byDepth"}
    if sorted_mode == "byProb":
        grammar_path = infer_grammar_path(args)
        groups = make_probability_sorted_groups(
            tree_strings,
            grammar_path=grammar_path,
            n_groups=int(getattr(args, "prob_group_count", 4)),
        )
        return groups, {
            "sorted_tree_groups": "byProb",
            "grammar_path": str(grammar_path),
            "prob_group_count": int(getattr(args, "prob_group_count", 4)),
        }
    raise ValueError(f"Unsupported sorted_tree_groups={sorted_mode!r}. Expected 'byDepth' or 'byProb'.")


def evaluate_grouped_test_sets(
    tree_groups: Dict[int, List[str]],
    *,
    model_for_hidden_states: nn.Module,
    tokenizer,
    probe_model: nn.Module,
    vocab: "AtomicLabelVocab",
    layer: int,
    config: "ProbeTrainingConfig",
    args: argparse.Namespace,
    results_dir: Path,
    run_stem: str,
) -> List[Dict[str, object]]:
    grouped_results: List[Dict[str, object]] = []
    for depth, tree_strings in tree_groups.items():
        test_examples = build_probe_examples_from_trees(
            tree_strings,
            model_for_hidden_states,
            layer,
            local_files_only=args.local_files_only,
            device=config.device,
            batch_size=int(args.extract_batch_size),
            tokenizer=tokenizer,
        )
        collate_fn = lambda batch: collate_atomic(batch, vocab)
        test_loader = DataLoader(ProbeDataset(test_examples), batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
        test_metrics = evaluate_probe(probe_model, test_loader, vocab, config)
        oracle_metrics = oracle_roundtrip_metrics(test_examples)
        grouped_results.append(
            {
                "depth_group": depth,
                "n_test": len(test_examples),
                "oracle_roundtrip": compact_metrics(oracle_metrics),
                "model": compact_metrics(test_metrics),
            }
        )
    return grouped_results


@torch.no_grad()
def build_probe_examples_from_trees(
    tree_strings: Sequence[str],
    model_or_path: Union[str, nn.Module],
    layer: int,
    *,
    local_files_only: bool,
    device: str,
    batch_size: int,
    tokenizer=None,
) -> List["ProbeExample"]:
    """Turn tree strings into probing examples using model hidden states."""

    from transformers import AutoModelForCausalLM, AutoTokenizer

    parsed_trees = [parse_bracket_tree(tree_str) for tree_str in tree_strings]
    encoded = [tree_to_sequence_labels(tree) for tree in parsed_trees]
    words_per_tree = [words for words, _ in encoded]
    labels_per_tree = [labels for _, labels in encoded]

    if isinstance(model_or_path, str):
        tokenizer = AutoTokenizer.from_pretrained(model_or_path, local_files_only=local_files_only)
        model = AutoModelForCausalLM.from_pretrained(model_or_path, local_files_only=local_files_only)
        model.eval()
        model.to(device)
    else:
        model = model_or_path
        if tokenizer is None:
            raise ValueError("tokenizer must be provided when reusing a preloaded model.")
        model.eval()
        model.to(device)

    examples: List[ProbeExample] = []
    for start in range(0, len(parsed_trees), batch_size):
        batch_words = words_per_tree[start : start + batch_size]
        enc = tokenizer(
            batch_words,
            is_split_into_words=True,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            return_attention_mask=True,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        output = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        layer_states = output.hidden_states[layer]

        for row, words in enumerate(batch_words):
            seq_len = int(attention_mask[row].sum().item())
            if hasattr(enc, "word_ids"):
                word_ids = enc.word_ids(batch_index=row)
            else:
                word_ids = list(range(seq_len))
            token_word_ids = [word_id for word_id in word_ids[:seq_len] if word_id is not None]
            expected_word_ids = list(range(len(words)))
            if token_word_ids != expected_word_ids:
                raise ValueError(
                    "Tokenizer alignment is not one-token-per-word. "
                    f"Expected word_ids={expected_word_ids} but got {token_word_ids} for words={words!r}."
                )
            examples.append(
                ProbeExample(
                    words=list(words),
                    gold_tree=parsed_trees[start + row],
                    hidden_states=layer_states[row, :seq_len].detach().cpu(),
                    labels=labels_per_tree[start + row],
                )
            )
    return examples


@dataclass
class ProbeExample:
    words: List[str]
    gold_tree: TreeNode
    hidden_states: torch.Tensor
    labels: List[SequenceLabel]


class AtomicLabelVocab:
    def __init__(self, labels: Sequence[SequenceLabel]):
        values = sorted({label.atomic_string() for label in labels})
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.itos = [self.pad_token, self.unk_token] + values
        self.stoi = {value: idx for idx, value in enumerate(self.itos)}
        self.pad_id = 0
        self.unk_id = 1

    def encode(self, labels: Sequence[SequenceLabel]) -> List[int]:
        return [self.stoi.get(label.atomic_string(), self.unk_id) for label in labels]

    def decode(self, ids: Sequence[int]) -> List[SequenceLabel]:
        decoded: List[SequenceLabel] = []
        for idx in ids:
            value = self.itos[int(idx)]
            if value == self.pad_token:
                decoded.append(SequenceLabel(n=0, c=END_C_LABEL, u=()))
            elif value == self.unk_token:
                decoded.append(SequenceLabel(n=0, c=UNKNOWN_LABEL, u=()))
            else:
                decoded.append(SequenceLabel.from_atomic_string(value))
        return decoded


class ProbeDataset(Dataset):
    def __init__(self, examples: Sequence[ProbeExample]):
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> ProbeExample:
        return self.examples[idx]


@dataclass
class Batch:
    hidden_states: torch.Tensor
    mask: torch.Tensor
    words: List[List[str]]
    gold_trees: List[TreeNode]
    gold_labels: List[List[SequenceLabel]]
    atomic_targets: Optional[torch.Tensor] = None


def collate_atomic(batch: Sequence[ProbeExample], vocab: AtomicLabelVocab) -> Batch:
    max_len = max(example.hidden_states.shape[0] for example in batch)
    hidden_dim = batch[0].hidden_states.shape[1]
    hidden = torch.zeros(len(batch), max_len, hidden_dim, dtype=batch[0].hidden_states.dtype)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    targets = torch.full((len(batch), max_len), fill_value=vocab.pad_id, dtype=torch.long)
    words: List[List[str]] = []
    trees: List[TreeNode] = []
    gold_labels: List[List[SequenceLabel]] = []

    for row, example in enumerate(batch):
        seq_len = example.hidden_states.shape[0]
        hidden[row, :seq_len] = example.hidden_states
        mask[row, :seq_len] = True
        targets[row, :seq_len] = torch.tensor(vocab.encode(example.labels), dtype=torch.long)
        words.append(example.words)
        trees.append(example.gold_tree)
        gold_labels.append(list(example.labels))
    return Batch(hidden_states=hidden, mask=mask, words=words, gold_trees=trees, gold_labels=gold_labels, atomic_targets=targets)


class LinearTokenProbe(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.classifier = nn.Linear(input_dim, output_dim)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.classifier(hidden_states)


@dataclass
class ProbeTrainingConfig:
    lr: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 16
    epochs: int = 20
    patience: int = 3
    device: str = "cpu"
    seed: int = 13


def build_probe(
    input_dim: int,
    labels: Sequence[SequenceLabel],
    config: ProbeTrainingConfig,
) -> Tuple[nn.Module, AtomicLabelVocab]:
    vocab = AtomicLabelVocab(labels)
    model = LinearTokenProbe(input_dim, len(vocab.itos))
    return model, vocab


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cross_entropy_masked(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor, pad_id: int) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_targets = targets.reshape(-1)
    flat_mask = mask.reshape(-1)
    return F.cross_entropy(flat_logits[flat_mask], flat_targets[flat_mask], ignore_index=pad_id)


def compute_macro_f1(gold: Sequence[str], pred: Sequence[str]) -> float:
    labels = sorted(set(gold) | set(pred))
    if not labels:
        return 0.0
    f1_values: List[float] = []
    for label in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_values.append(0.0 if precision + recall == 0.0 else (2.0 * precision * recall) / (precision + recall))
    return float(sum(f1_values) / len(f1_values))


def token_label_metrics(
    gold_sequences: Sequence[Sequence[SequenceLabel]],
    pred_sequences: Sequence[Sequence[SequenceLabel]],
) -> Dict[str, float]:
    if len(gold_sequences) != len(pred_sequences):
        raise ValueError("Gold and predicted label sequences must have equal length.")
    gold_flat: List[str] = []
    pred_flat: List[str] = []
    for gold_seq, pred_seq in zip(gold_sequences, pred_sequences):
        if len(gold_seq) != len(pred_seq):
            raise ValueError("Gold and predicted label sequences must have matching token lengths.")
        gold_flat.extend(label.atomic_string() for label in gold_seq)
        pred_flat.extend(label.atomic_string() for label in pred_seq)
    accuracy = float(sum(int(g == p) for g, p in zip(gold_flat, pred_flat)) / max(1, len(gold_flat)))
    macro_f1 = compute_macro_f1(gold_flat, pred_flat)
    return {"accuracy": accuracy, "macro_f1": macro_f1}


def reencode_label_sequences(words_per_tree: Sequence[Sequence[str]], trees: Sequence[TreeNode]) -> List[List[SequenceLabel]]:
    if len(words_per_tree) != len(trees):
        raise ValueError("Word sequences and trees must have equal length.")
    reencoded: List[List[SequenceLabel]] = []
    for words, tree in zip(words_per_tree, trees):
        decoded_words, labels = tree_to_sequence_labels(tree)
        if list(decoded_words) != list(words):
            raise ValueError(
                f"Re-encoded tree words do not match the original sentence. Expected {list(words)!r} but got {list(decoded_words)!r}."
            )
        reencoded.append(list(labels))
    return reencoded


def decode_batch_predictions(
    model_output: torch.Tensor,
    batch: Batch,
    vocab: AtomicLabelVocab,
) -> List[List[SequenceLabel]]:
    decoded: List[List[SequenceLabel]] = []
    lengths = batch.mask.sum(dim=1).tolist()
    pred_ids = torch.argmax(model_output, dim=-1).detach().cpu()
    for row, length in enumerate(lengths):
        decoded.append(vocab.decode(pred_ids[row, :length].tolist()))
    return decoded


@torch.no_grad()
def evaluate_probe(
    model: nn.Module,
    dataloader: DataLoader,
    vocab: AtomicLabelVocab,
    config: ProbeTrainingConfig,
) -> Dict[str, object]:
    model.eval()
    gold_trees: List[TreeNode] = []
    pred_trees: List[TreeNode] = []
    gold_label_sequences: List[List[SequenceLabel]] = []
    pred_label_sequences: List[List[SequenceLabel]] = []
    total_loss = 0.0
    total_batches = 0

    for batch in dataloader:
        batch.hidden_states = batch.hidden_states.to(config.device)
        batch.mask = batch.mask.to(config.device)
        output = model(batch.hidden_states)
        assert batch.atomic_targets is not None
        targets = batch.atomic_targets.to(config.device)
        loss = cross_entropy_masked(output, targets, batch.mask, vocab.pad_id)
        total_loss += float(loss.item())
        total_batches += 1

        decoded_labels = decode_batch_predictions(output, batch, vocab)
        for words, gold_tree, gold_labels, pred_labels in zip(batch.words, batch.gold_trees, batch.gold_labels, decoded_labels):
            gold_trees.append(gold_tree)
            gold_label_sequences.append(list(gold_labels))
            pred_label_sequences.append(list(pred_labels))
            pred_trees.append(decode_sequence_labels(words, pred_labels))

    metrics = labeled_bracketing_scores(gold_trees, pred_trees)
    metrics.update(token_label_metrics(gold_label_sequences, pred_label_sequences))
    metrics["loss"] = total_loss / max(1, total_batches)
    metrics["span_length_breakdown"] = labeled_f1_by_span_length(gold_trees, pred_trees)
    metrics["pred_trees"] = pred_trees
    metrics["gold_trees"] = gold_trees
    return metrics


def train_probe(
    train_examples: Sequence[ProbeExample],
    dev_examples: Sequence[ProbeExample],
    test_examples: Sequence[ProbeExample],
    config: ProbeTrainingConfig,
) -> Dict[str, object]:
    """Train a probe with early stopping on dev labeled F1."""

    if not train_examples:
        raise ValueError("Need at least one training example.")

    set_random_seed(config.seed)
    all_labels = [label for example in train_examples for label in example.labels]
    input_dim = int(train_examples[0].hidden_states.shape[-1])
    model, vocab = build_probe(input_dim, all_labels, config)
    model.to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    collate_fn = lambda batch: collate_atomic(batch, vocab)

    train_loader = DataLoader(ProbeDataset(train_examples), batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(ProbeDataset(dev_examples), batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(ProbeDataset(test_examples), batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)

    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_dev_f1 = -1.0
    best_epoch = 0
    bad_epochs = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, config.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch.hidden_states = batch.hidden_states.to(config.device)
            batch.mask = batch.mask.to(config.device)
            optimizer.zero_grad()
            output = model(batch.hidden_states)
            assert batch.atomic_targets is not None
            targets = batch.atomic_targets.to(config.device)
            loss = cross_entropy_masked(output, targets, batch.mask, vocab.pad_id)

            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            n_batches += 1

        dev_metrics = evaluate_probe(model, dev_loader, vocab, config)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": running_loss / max(1, n_batches),
                "dev_loss": float(dev_metrics["loss"]),
                "dev_f1": float(dev_metrics["f1"]),
            }
        )
        if float(dev_metrics["f1"]) > best_dev_f1 + 1e-12:
            best_dev_f1 = float(dev_metrics["f1"])
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_metrics = evaluate_probe(model, test_loader, vocab, config)
    return {
        "best_epoch": best_epoch,
        "best_dev_f1": best_dev_f1,
        "history": history,
        "test_metrics": test_metrics,
        "model": model,
        "vocab": vocab,
    }


def print_gold_vs_reconstructed(tree: TreeNode) -> None:
    words, labels = tree_to_sequence_labels(tree)
    reconstructed = decode_sequence_labels(words, labels)
    print("Words:", words)
    print("Labels:", [label.atomic_string() for label in labels])
    print("Gold:", tree.pretty())
    print("Reconstructed:", reconstructed.pretty())


def verify_roundtrip_identity(trees: Sequence[TreeNode]) -> Dict[str, object]:
    exact = 0
    mismatches: List[Tuple[str, str]] = []
    for tree in trees:
        words, labels = tree_to_sequence_labels(tree)
        reconstructed = decode_sequence_labels(words, labels)
        if tree.pretty() == reconstructed.pretty():
            exact += 1
        else:
            mismatches.append((tree.pretty(), reconstructed.pretty()))
    return {"n_trees": len(trees), "exact_matches": exact, "mismatches": mismatches[:5]}


def verify_relative_n_values(tree: TreeNode) -> Dict[str, object]:
    words, labels = tree_to_sequence_labels(tree)
    shared = labels_to_absolute_shared_levels(labels)
    observed = [label.n for label in labels]
    expected = [shared[0]] + [shared[i] - shared[i - 1] for i in range(1, len(shared))]
    return {
        "words": words,
        "absolute_shared_levels": shared,
        "observed_n": observed,
        "expected_n_from_shared_levels": expected,
        "matches": observed == expected,
    }


def inspect_common_decoding_failures(gold_trees: Sequence[TreeNode], pred_trees: Sequence[TreeNode], limit: int = 5) -> List[Dict[str, str]]:
    failures: List[Dict[str, str]] = []
    for gold, pred in zip(gold_trees, pred_trees):
        if gold.pretty() == pred.pretty():
            continue
        failures.append({"gold": gold.pretty(), "pred": pred.pretty()})
        if len(failures) >= limit:
            break
    return failures


def oracle_roundtrip_metrics(examples: Sequence[ProbeExample]) -> Dict[str, object]:
    gold_trees = [example.gold_tree for example in examples]
    words_per_tree = [list(example.words) for example in examples]
    gold_label_sequences = [list(example.labels) for example in examples]
    reconstructed_trees = [decode_sequence_labels(example.words, example.labels) for example in examples]
    reencoded_label_sequences = reencode_label_sequences(words_per_tree, reconstructed_trees)
    metrics = labeled_bracketing_scores(gold_trees, reconstructed_trees)

    #metrics.update(token_label_metrics(gold_label_sequences, gold_label_sequences))

    reencoded_metrics = token_label_metrics(gold_label_sequences, reencoded_label_sequences)
    #metrics["reencoded_accuracy"] = float(reencoded_metrics["accuracy"])
    #metrics["reencoded_macro_f1"] = float(reencoded_metrics["macro_f1"])

    metrics["accuracy"] = float(reencoded_metrics["accuracy"])
    metrics["macro_f1"] = float(reencoded_metrics["macro_f1"])
    metrics["span_length_breakdown"] = labeled_f1_by_span_length(gold_trees,reconstructed_trees,)

    metrics["gold_trees"] = gold_trees
    metrics["pred_trees"] = reconstructed_trees
    metrics["gold_label_sequences"] = gold_label_sequences
    metrics["reencoded_label_sequences"] = reencoded_label_sequences
    return metrics


def get_results_dir(reference_tree_path: str) -> Path:
    tree_path = Path(reference_tree_path).resolve()
    anchor: Optional[Path] = None
    for parent in tree_path.parents:
        if parent.name == "data":
            anchor = parent.parent
            break
    if anchor is None:
        anchor = tree_path.parent.parent if tree_path.parent.parent != tree_path.parent else tree_path.parent
    results_dir = anchor / "probing_results_data_from_llk_macro"
    #results_dir.mkdir(parents=True, exist_ok=True)
    model_dir = DEFAULT_MODEL_PATH.resolve().parent
    results_dir = model_dir.parent / "probing_results_data_from_llk_macro"
    results_dir.mkdir(parents=True, exist_ok=True)

    return results_dir


def sanitize_filename_part(value: str, max_len: int = 24) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", value).strip("-_")
    if not cleaned:
        cleaned = "run"
    return cleaned[:max_len]


def infer_model_tag(model_path: str) -> str:
    path_name = Path(model_path).name
    match = re.search(r"(gpt2(?:_[A-Za-z0-9]+)+)", path_name)
    if match:
        return sanitize_filename_part(match.group(1), max_len=20)
    return sanitize_filename_part(path_name, max_len=20)


def build_run_stem(
    args: argparse.Namespace,
    *,
    model_path: Optional[str] = None,
    layer: Optional[int] = None,
    run_mode: Optional[str] = None,
    checkpoint_label: Optional[str] = None,
) -> str:
    parts = [
        f"vilares_{infer_model_tag(model_path or args.model_path)}",
        sanitize_filename_part(run_mode or getattr(args, "run_mode", "individual"), max_len=16),
        f"l{int(layer if layer is not None else args.layer)}",
        f"tr{int(args.train_trees)}",
        f"dv{int(args.dev_trees)}",
        f"te{int(args.fixed_test_size)}",
        f"s{int(args.seed)}",
        f"m{int(args.model_init_seed)}",
    ]
    if "sorted_trees" in str(run_mode or getattr(args, "run_mode", "individual")):
        parts.append(sanitize_filename_part(str(getattr(args, "sorted_tree_groups", "byDepth")), max_len=12))
    if checkpoint_label:
        parts.append(sanitize_filename_part(checkpoint_label, max_len=20))
    return "_".join(parts)


def compact_metrics(metrics: Dict[str, object]) -> Dict[str, object]:
    compact = {
        "f1": float(metrics["f1"]),
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
    }
    if "reencoded_accuracy" in metrics:
        compact["reencoded_accuracy"] = float(metrics["reencoded_accuracy"])
    if "reencoded_macro_f1" in metrics:
        compact["reencoded_macro_f1"] = float(metrics["reencoded_macro_f1"])
    return compact


def concise_result_row(output: Dict[str, object], group_key: str, group_value: object) -> Dict[str, object]:
    return {
        group_key: group_value,
        "oracle_roundtrip": output["oracle_roundtrip_test_metrics"],
        "model": output["trained_probe"]["test_metrics"],
        "baseline": output["random_init_baseline"]["test_metrics"],
    }


def print_first_model_sample(split_name: str, examples: Sequence["ProbeExample"]) -> None:
    if not examples:
        print(f"{split_name}_first_model_input: <empty>")
        return
    example = examples[0]
    sentence = " ".join(example.words)
    print(f"{split_name}_first_model_input_words: {example.words}")
    print(f"{split_name}_first_model_input_sentence: {sentence}")
    print(f"{split_name}_first_gold_tree: {example.gold_tree.pretty()}")


def predict_labels_for_example(
    model: nn.Module,
    vocab: AtomicLabelVocab,
    example: ProbeExample,
    config: ProbeTrainingConfig,
) -> List[SequenceLabel]:
    collate_fn = lambda batch: collate_atomic(batch, vocab)
    batch = collate_fn([example])
    batch.hidden_states = batch.hidden_states.to(config.device)
    batch.mask = batch.mask.to(config.device)
    model.eval()
    with torch.no_grad():
        output = model(batch.hidden_states)
    return decode_batch_predictions(output, batch, vocab)[0]


def save_tree_comparison_plot(
    gold_tree: TreeNode,
    pred_tree: TreeNode,
    output_path: Path,
    *,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    def build_plot_data(tree: TreeNode) -> Tuple[Dict[str, Tuple[float, float]], List[Tuple[str, str]], Dict[str, str]]:
        positions: Dict[str, Tuple[float, float]] = {}
        edges: List[Tuple[str, str]] = []
        labels: Dict[str, str] = {}
        next_leaf_x = 0
        next_node_id = 0

        def new_id() -> str:
            nonlocal next_node_id
            next_node_id += 1
            return f"n{next_node_id}"

        def visit(node: TreeNode, depth: int) -> str:
            nonlocal next_leaf_x
            node_id = new_id()
            labels[node_id] = node.label
            child_xs: List[float] = []
            for child in node.children:
                if isinstance(child, str):
                    word_id = new_id()
                    labels[word_id] = child
                    positions[word_id] = (float(next_leaf_x), -float(depth + 1))
                    edges.append((node_id, word_id))
                    child_xs.append(float(next_leaf_x))
                    next_leaf_x += 1
                else:
                    child_id = visit(child, depth + 1)
                    edges.append((node_id, child_id))
                    child_xs.append(positions[child_id][0])
            if not child_xs:
                child_xs.append(float(next_leaf_x))
                next_leaf_x += 1
            positions[node_id] = (sum(child_xs) / len(child_xs), -float(depth))
            return node_id

        visit(tree, 0)
        return positions, edges, labels

    def draw_tree(ax, tree: TreeNode, heading: str) -> None:
        positions, edges, labels = build_plot_data(tree)
        ax.axis("off")
        ax.set_title(heading)
        for parent_id, child_id in edges:
            x0, y0 = positions[parent_id]
            x1, y1 = positions[child_id]
            ax.plot([x0, x1], [y0, y1], color="black", linewidth=1.0, zorder=1)
        for node_id, (x, y) in positions.items():
            is_terminal = not any(parent_id == node_id for parent_id, _ in edges)
            ax.text(
                x,
                y,
                labels[node_id],
                ha="center",
                va="center",
                fontsize=9 if is_terminal else 10,
                family="monospace",
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "facecolor": "#f5efe6" if is_terminal else "#d9e6f2",
                    "edgecolor": "#444444",
                    "linewidth": 0.8,
                },
                zorder=2,
            )
        xs = [x for x, _ in positions.values()]
        ys = [y for _, y in positions.values()]
        ax.set_xlim(min(xs) - 1.0, max(xs) + 1.0)
        ax.set_ylim(min(ys) - 1.0, max(ys) + 1.0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(gold_tree.words()) * 1.4), 8))
    draw_tree(axes[0], gold_tree, "Gold Tree")
    draw_tree(axes[1], pred_tree, "Reconstructed Tree")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def print_example_debug_report(
    *,
    prefix: str,
    example: ProbeExample,
    predicted_labels: Sequence[SequenceLabel],
    predicted_tree: TreeNode,
) -> None:
    gold_roundtrip = decode_sequence_labels(example.words, example.labels)
    gold_n_info = verify_relative_n_values(example.gold_tree)
    example_scores = labeled_bracketing_scores([example.gold_tree], [predicted_tree])
    print(f"{prefix}_words:", example.words)
    print(f"{prefix}_gold_labels:", [label.atomic_string() for label in example.labels])
    print(f"{prefix}_pred_labels:", [label.atomic_string() for label in predicted_labels])
    print(f"{prefix}_gold_shared_levels:", gold_n_info["absolute_shared_levels"])
    print(f"{prefix}_gold_n:", gold_n_info["observed_n"])
    print(f"{prefix}_pred_n:", [label.n for label in predicted_labels])
    print(f"{prefix}_gold_roundtrip_exact:", example.gold_tree.pretty() == gold_roundtrip.pretty())
    print(
        f"{prefix}_span_scores:"
        f" P={float(example_scores['precision']):.4f}"
        f" R={float(example_scores['recall']):.4f}"
        f" F1={float(example_scores['f1']):.4f}"
    )

def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def load_tokenizer(tokenizer_path: str, *, local_files_only: bool):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=local_files_only)


def load_model(model_path: str, *, local_files_only: bool, device: str):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=local_files_only)
    model.eval()
    model.to(device)
    return model


def load_random_baseline_model(
    model_path: str,
    *,
    local_files_only: bool,
    device: str,
    model_init_seed: int,
) -> nn.Module:
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_path, local_files_only=local_files_only)
    torch.manual_seed(model_init_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_init_seed)
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    model.to(device)
    return model


def validate_requested_layer(model: nn.Module, layer_1_based: int) -> int:
    n_layers = getattr(model.config, "n_layer", None)
    if n_layers is None:
        raise ValueError("Could not infer n_layer from the model config.")
    if not (1 <= int(layer_1_based) <= int(n_layers)):
        raise ValueError(f"layer must be in [1, {int(n_layers)}] but got {layer_1_based}.")
    return int(layer_1_based)


def discover_checkpoints(model_path: str) -> List[Tuple[str, str]]:
    model_root = Path(model_path)
    checkpoint_dirs = [p for p in model_root.glob("checkpoint-*") if p.is_dir()]

    def sort_key(path: Path) -> Tuple[int, str]:
        match = re.search(r"checkpoint-(\d+)$", path.name)
        return (int(match.group(1)) if match else 10**12, path.name)

    ordered = sorted(checkpoint_dirs, key=sort_key)
    discovered = [(str(path), path.name) for path in ordered]
    discovered.append((str(model_root), "final"))
    return discovered


def run_single_probe(
    args: argparse.Namespace,
    *,
    model_path: str,
    tokenizer_path: str,
    layer: int,
    run_mode_label: str,
    checkpoint_label: Optional[str] = None,
    split_override: Optional[Tuple[List[str], List[str], List[str], Dict[str, object]]] = None,
    write_results_file: bool = True,
) -> Dict[str, object]:
    device = resolve_device(args.device)
    tokenizer = load_tokenizer(tokenizer_path, local_files_only=args.local_files_only)
    model = load_model(model_path, local_files_only=args.local_files_only, device=device)
    layer = validate_requested_layer(model, int(layer))

    if split_override is None:
        train_tree_strings, dev_tree_strings, test_tree_strings, split_info = prepare_tree_splits(args)
    else:
        train_tree_strings, dev_tree_strings, test_tree_strings, split_info = split_override
    data_source_note = str(split_info.get("split_strategy", "explicit_tree_files"))

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
    print_first_model_sample("train", train_examples)
    print_first_model_sample("dev", dev_examples)
    print_first_model_sample("test", test_examples)

    config = ProbeTrainingConfig(
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        patience=int(args.patience),
        device=device,
        seed=int(args.seed),
    )
    result = train_probe(train_examples, dev_examples, test_examples, config)
    test_metrics = dict(result["test_metrics"])
    oracle_test_metrics = oracle_roundtrip_metrics(test_examples)

    baseline_model = load_random_baseline_model(
        model_path,
        local_files_only=args.local_files_only,
        device=device,
        model_init_seed=int(args.model_init_seed),
    )
    validate_requested_layer(baseline_model, int(layer))
    baseline_train_examples = build_probe_examples_from_trees(
        train_tree_strings,
        baseline_model,
        layer,
        local_files_only=args.local_files_only,
        device=device,
        batch_size=int(args.extract_batch_size),
        tokenizer=tokenizer,
    )
    baseline_dev_examples = build_probe_examples_from_trees(
        dev_tree_strings,
        baseline_model,
        layer,
        local_files_only=args.local_files_only,
        device=device,
        batch_size=int(args.extract_batch_size),
        tokenizer=tokenizer,
    )
    baseline_test_examples = build_probe_examples_from_trees(
        test_tree_strings,
        baseline_model,
        layer,
        local_files_only=args.local_files_only,
        device=device,
        batch_size=int(args.extract_batch_size),
        tokenizer=tokenizer,
    )
    baseline_result = train_probe(baseline_train_examples, baseline_dev_examples, baseline_test_examples, config)
    baseline_test_metrics = dict(baseline_result["test_metrics"])
    results_dir = get_results_dir(args.model_path)
    run_stem = build_run_stem(
        args,
        model_path=model_path,
        layer=layer,
        run_mode=run_mode_label,
        checkpoint_label=checkpoint_label,
    )
    results_json_path = results_dir / f"{run_stem}.json"

    output = {
        "model_path": str(Path(model_path).resolve()),
        "run_mode": run_mode_label,
        "checkpoint_label": checkpoint_label,
        "layer_1_based": layer,
        "oracle_roundtrip_test_metrics": compact_metrics(oracle_test_metrics),
        "trained_probe": {"test_metrics": compact_metrics(test_metrics)},
        "random_init_baseline": {"test_metrics": compact_metrics(baseline_test_metrics)},
    }

    print(f"model_path={model_path}")
    print(f"tree_data_source=data_trees_lookup ({data_source_note})")
    print(
        f"run_mode={run_mode_label} layer={layer} probe=linear "
        f"device={device}"
    )
    if checkpoint_label is not None:
        print(f"checkpoint_label={checkpoint_label}")
    print(
        f"splits train/dev/test={len(train_examples)}/{len(dev_examples)}/{len(test_examples)} "
        f"(requested train_trees={args.train_trees}, dev_trees={args.dev_trees}, fixed_test_size={args.fixed_test_size})"
    )
    print(
        "oracle roundtrip test labeled bracketing:"
        f" P={float(oracle_test_metrics['precision']):.4f}"
        f" R={float(oracle_test_metrics['recall']):.4f}"
        f" F1={float(oracle_test_metrics['f1']):.4f}"
        f" Acc={float(oracle_test_metrics['accuracy']):.4f}"
        f" MacroF1={float(oracle_test_metrics['macro_f1']):.4f}"
        f" ReEncAcc={float(oracle_test_metrics['reencoded_accuracy']):.4f}"
        f" ReEncMacroF1={float(oracle_test_metrics['reencoded_macro_f1']):.4f}"
    )
    print(
        "trained model test labeled bracketing:"
        f" P={float(test_metrics['precision']):.4f}"
        f" R={float(test_metrics['recall']):.4f}"
        f" F1={float(test_metrics['f1']):.4f}"
        f" Acc={float(test_metrics['accuracy']):.4f}"
        f" MacroF1={float(test_metrics['macro_f1']):.4f}"
    )
    print(
        "random-init baseline test labeled bracketing:"
        f" P={float(baseline_test_metrics['precision']):.4f}"
        f" R={float(baseline_test_metrics['recall']):.4f}"
        f" F1={float(baseline_test_metrics['f1']):.4f}"
        f" Acc={float(baseline_test_metrics['accuracy']):.4f}"
        f" MacroF1={float(baseline_test_metrics['macro_f1']):.4f}"
        f" (model_init_seed={int(args.model_init_seed)})"
    )
    if args.show_example and test_examples:
        example_idx = min(max(0, int(args.debug_example_index)), len(test_examples) - 1)
        example = test_examples[example_idx]
        baseline_example = baseline_test_examples[example_idx]
        trained_pred_labels = predict_labels_for_example(result["model"], result["vocab"], example, config)
        trained_pred_tree = decode_sequence_labels(example.words, trained_pred_labels)
        baseline_pred_labels = predict_labels_for_example(baseline_result["model"], baseline_result["vocab"], baseline_example, config)
        baseline_pred_tree = decode_sequence_labels(baseline_example.words, baseline_pred_labels)

        trained_plot_path = results_dir / f"{run_stem}_ex{example_idx}_trained.png"
        baseline_plot_path = results_dir / f"{run_stem}_ex{example_idx}_random_init.png"
        save_tree_comparison_plot(
            example.gold_tree,
            trained_pred_tree,
            trained_plot_path,
            title=f"Trained Model Example {example_idx}",
        )
        save_tree_comparison_plot(
            baseline_example.gold_tree,
            baseline_pred_tree,
            baseline_plot_path,
            title=f"Random Init Baseline Example {example_idx}",
        )

        print(f"trained_tree_plot={trained_plot_path}")
        print(f"random_init_tree_plot={baseline_plot_path}")
        print(f"debug_example_index={example_idx}")
        print_example_debug_report(
            prefix="trained_example",
            example=example,
            predicted_labels=trained_pred_labels,
            predicted_tree=trained_pred_tree,
        )
        print_example_debug_report(
            prefix="random_init_example",
            example=baseline_example,
            predicted_labels=baseline_pred_labels,
            predicted_tree=baseline_pred_tree,
        )

    if args.show_example and int(args.debug_failure_limit) > 0:
        trained_failures = inspect_common_decoding_failures(test_metrics["gold_trees"], test_metrics["pred_trees"], limit=int(args.debug_failure_limit))
        baseline_failures = inspect_common_decoding_failures(
            baseline_test_metrics["gold_trees"],
            baseline_test_metrics["pred_trees"],
            limit=int(args.debug_failure_limit),
        )
        print("trained_decoding_failures:", trained_failures)
        print("random_init_decoding_failures:", baseline_failures)
    output["results_json"] = str(results_json_path)
    if write_results_file:
        results_json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"results_json={results_json_path}")
    return output


def run_repo_probe(args: argparse.Namespace) -> Dict[str, object]:
    run_mode = str(args.run_mode).strip().lower()
    if run_mode == "individual":
        return run_single_probe(
            args,
            model_path=args.model_path,
            tokenizer_path=args.model_path,
            layer=int(args.layer),
            run_mode_label="individual",
        )

    if run_mode == "checkpoints":
        results_dir = get_results_dir(args.train_trees_path)
        all_outputs: List[Dict[str, object]] = []
        for checkpoint_path, checkpoint_label in discover_checkpoints(args.model_path):
            all_outputs.append(
                run_single_probe(
                    args,
                    model_path=checkpoint_path,
                    tokenizer_path=args.model_path,
                    layer=int(args.layer),
                    run_mode_label="checkpoints",
                    checkpoint_label=checkpoint_label,
                    write_results_file=False,
                )
            )
        checkpoint_results = [
            concise_result_row(out, "checkpoint", str(out.get("checkpoint_label") or "final"))
            for out in all_outputs
        ]
        aggregate = {
            "run_mode": "checkpoints",
            "model_path": str(Path(args.model_path).resolve()),
            "layer_1_based": int(args.layer),
            "results": checkpoint_results,
        }
        aggregate_path = results_dir / f"{build_run_stem(args, run_mode='checkpoints', layer=int(args.layer))}.json"
        aggregate["results_json"] = str(aggregate_path)
        aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        print(f"results_json={aggregate_path}")
        return aggregate

    if run_mode in {"layers_sorted_trees", "checkpoints_sorted_trees"}:
        base_train, base_dev, base_test, base_split_info = prepare_tree_splits(args)
        test_groups, sorted_group_info = build_sorted_tree_groups(args, base_test)
        results_dir = get_results_dir(args.train_trees_path)

        if run_mode == "layers_sorted_trees":
            device = resolve_device(args.device)
            tokenizer = load_tokenizer(args.model_path, local_files_only=args.local_files_only)
            model = load_model(args.model_path, local_files_only=args.local_files_only, device=device)
            n_layers = validate_requested_layer(model, int(getattr(model.config, "n_layer")))
            baseline_model = load_random_baseline_model(
                args.model_path,
                local_files_only=args.local_files_only,
                device=device,
                model_init_seed=int(args.model_init_seed),
            )
            aggregate_groups: List[Dict[str, object]] = []
            for layer in range(1, n_layers + 1):
                train_examples = build_probe_examples_from_trees(
                    base_train, model, layer, local_files_only=args.local_files_only, device=device, batch_size=int(args.extract_batch_size), tokenizer=tokenizer
                )
                dev_examples = build_probe_examples_from_trees(
                    base_dev, model, layer, local_files_only=args.local_files_only, device=device, batch_size=int(args.extract_batch_size), tokenizer=tokenizer
                )
                full_test_examples = build_probe_examples_from_trees(
                    base_test, model, layer, local_files_only=args.local_files_only, device=device, batch_size=int(args.extract_batch_size), tokenizer=tokenizer
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
                trained_probe = train_probe(train_examples, dev_examples, full_test_examples, config)
                baseline_train_examples = build_probe_examples_from_trees(
                    base_train, baseline_model, layer, local_files_only=args.local_files_only, device=device, batch_size=int(args.extract_batch_size), tokenizer=tokenizer
                )
                baseline_dev_examples = build_probe_examples_from_trees(
                    base_dev, baseline_model, layer, local_files_only=args.local_files_only, device=device, batch_size=int(args.extract_batch_size), tokenizer=tokenizer
                )
                baseline_full_test_examples = build_probe_examples_from_trees(
                    base_test, baseline_model, layer, local_files_only=args.local_files_only, device=device, batch_size=int(args.extract_batch_size), tokenizer=tokenizer
                )
                baseline_probe = train_probe(baseline_train_examples, baseline_dev_examples, baseline_full_test_examples, config)

                depth_rows: List[Dict[str, object]] = []
                for group in test_groups:
                    tree_strings = group.tree_strings
                    test_examples = build_probe_examples_from_trees(
                        tree_strings, model, layer, local_files_only=args.local_files_only, device=device, batch_size=int(args.extract_batch_size), tokenizer=tokenizer
                    )
                    baseline_test_examples = build_probe_examples_from_trees(
                        tree_strings, baseline_model, layer, local_files_only=args.local_files_only, device=device, batch_size=int(args.extract_batch_size), tokenizer=tokenizer
                    )
                    collate_trained = lambda batch: collate_atomic(batch, trained_probe["vocab"])
                    collate_baseline = lambda batch: collate_atomic(batch, baseline_probe["vocab"])
                    trained_metrics = evaluate_probe(
                        trained_probe["model"],
                        DataLoader(ProbeDataset(test_examples), batch_size=config.batch_size, shuffle=False, collate_fn=collate_trained),
                        trained_probe["vocab"],
                        config,
                    )
                    baseline_metrics = evaluate_probe(
                        baseline_probe["model"],
                        DataLoader(ProbeDataset(baseline_test_examples), batch_size=config.batch_size, shuffle=False, collate_fn=collate_baseline),
                        baseline_probe["vocab"],
                        config,
                    )
                    row = {
                        "group_type": group.group_type,
                        "group_label": group.group_label,
                        "n_test": len(test_examples),
                        "oracle_roundtrip": compact_metrics(oracle_roundtrip_metrics(test_examples)),
                        "model": compact_metrics(trained_metrics),
                        "baseline": compact_metrics(baseline_metrics),
                    }
                    row.update(group.metadata)
                    depth_rows.append(row)
                aggregate_groups.append({"layer": layer, "results": depth_rows})
            aggregate = {
                "run_mode": "layers_sorted_trees",
                "model_path": str(Path(args.model_path).resolve()),
                "sorted_grouping": sorted_group_info,
                "results": aggregate_groups,
            }
            aggregate_path = results_dir / f"{build_run_stem(args, run_mode='layers_sorted_trees', layer=0)}.json"
            aggregate["results_json"] = str(aggregate_path)
            aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
            print(f"results_json={aggregate_path}")
            return aggregate

        device = resolve_device(args.device)
        tokenizer = load_tokenizer(args.model_path, local_files_only=args.local_files_only)
        checkpoint_groups: List[Dict[str, object]] = []
        for checkpoint_path, checkpoint_label in discover_checkpoints(args.model_path):
            model = load_model(checkpoint_path, local_files_only=args.local_files_only, device=device)
            layer = validate_requested_layer(model, int(args.layer))
            baseline_model = load_random_baseline_model(
                args.model_path,
                local_files_only=args.local_files_only,
                device=device,
                model_init_seed=int(args.model_init_seed),
            )
            train_examples = build_probe_examples_from_trees(
                base_train, model, layer, local_files_only=args.local_files_only, device=device, batch_size=int(args.extract_batch_size), tokenizer=tokenizer
            )
            dev_examples = build_probe_examples_from_trees(
                base_dev, model, layer, local_files_only=args.local_files_only, device=device, batch_size=int(args.extract_batch_size), tokenizer=tokenizer
            )
            full_test_examples = build_probe_examples_from_trees(
                base_test, model, layer, local_files_only=args.local_files_only, device=device, batch_size=int(args.extract_batch_size), tokenizer=tokenizer
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
            trained_probe = train_probe(train_examples, dev_examples, full_test_examples, config)
            baseline_train_examples = build_probe_examples_from_trees(
                base_train, baseline_model, layer, local_files_only=args.local_files_only, device=device, batch_size=int(args.extract_batch_size), tokenizer=tokenizer
            )
            baseline_dev_examples = build_probe_examples_from_trees(
                base_dev, baseline_model, layer, local_files_only=args.local_files_only, device=device, batch_size=int(args.extract_batch_size), tokenizer=tokenizer
            )
            baseline_full_test_examples = build_probe_examples_from_trees(
                base_test, baseline_model, layer, local_files_only=args.local_files_only, device=device, batch_size=int(args.extract_batch_size), tokenizer=tokenizer
            )
            baseline_probe = train_probe(baseline_train_examples, baseline_dev_examples, baseline_full_test_examples, config)
            depth_rows: List[Dict[str, object]] = []
            for group in test_groups:
                tree_strings = group.tree_strings
                test_examples = build_probe_examples_from_trees(
                    tree_strings, model, layer, local_files_only=args.local_files_only, device=device, batch_size=int(args.extract_batch_size), tokenizer=tokenizer
                )
                baseline_test_examples = build_probe_examples_from_trees(
                    tree_strings, baseline_model, layer, local_files_only=args.local_files_only, device=device, batch_size=int(args.extract_batch_size), tokenizer=tokenizer
                )
                collate_trained = lambda batch: collate_atomic(batch, trained_probe["vocab"])
                collate_baseline = lambda batch: collate_atomic(batch, baseline_probe["vocab"])
                trained_metrics = evaluate_probe(
                    trained_probe["model"],
                    DataLoader(ProbeDataset(test_examples), batch_size=config.batch_size, shuffle=False, collate_fn=collate_trained),
                    trained_probe["vocab"],
                    config,
                )
                baseline_metrics = evaluate_probe(
                    baseline_probe["model"],
                    DataLoader(ProbeDataset(baseline_test_examples), batch_size=config.batch_size, shuffle=False, collate_fn=collate_baseline),
                    baseline_probe["vocab"],
                    config,
                )
                row = {
                    "group_type": group.group_type,
                    "group_label": group.group_label,
                    "n_test": len(test_examples),
                    "oracle_roundtrip": compact_metrics(oracle_roundtrip_metrics(test_examples)),
                    "model": compact_metrics(trained_metrics),
                    "baseline": compact_metrics(baseline_metrics),
                }
                row.update(group.metadata)
                depth_rows.append(row)
            checkpoint_groups.append({"checkpoint": checkpoint_label, "results": depth_rows})
        aggregate = {
            "run_mode": "checkpoints_sorted_trees",
            "model_path": str(Path(args.model_path).resolve()),
            "layer_1_based": int(args.layer),
            "sorted_grouping": sorted_group_info,
            "results": checkpoint_groups,
        }
        aggregate_path = results_dir / f"{build_run_stem(args, run_mode='checkpoints_sorted_trees', layer=int(args.layer))}.json"
        aggregate["results_json"] = str(aggregate_path)
        aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        print(f"results_json={aggregate_path}")
        return aggregate

    if run_mode == "layers":
        device = resolve_device(args.device)
        model = load_model(args.model_path, local_files_only=args.local_files_only, device=device)
        n_layers = validate_requested_layer(model, int(getattr(model.config, "n_layer")))
        del model
        results_dir = get_results_dir(args.train_trees_path)
        all_outputs = []
        for layer in range(1, n_layers + 1):
            all_outputs.append(
                run_single_probe(
                    args,
                    model_path=args.model_path,
                    tokenizer_path=args.model_path,
                    layer=layer,
                    run_mode_label="layers",
                    write_results_file=False,
                )
            )
        layer_results = [
            concise_result_row(out, "layer", int(out["layer_1_based"]))
            for out in all_outputs
        ]
        aggregate = {
            "run_mode": "layers",
            "model_path": str(Path(args.model_path).resolve()),
            "results": layer_results,
        }
        aggregate_path = results_dir / f"{build_run_stem(args, run_mode='layers', layer=0)}.json"
        aggregate["results_json"] = str(aggregate_path)
        aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
        print(f"results_json={aggregate_path}")
        return aggregate

    raise ValueError(
        f"Unsupported run_mode={args.run_mode!r}. Expected one of: "
        "individual, checkpoints, layers, layers_sorted_trees, checkpoints_sorted_trees."
    )


class EncodingTests(unittest.TestCase):
    def test_gold_label_extraction(self) -> None:
        tree = parse_bracket_tree("(S (NP (DT The) (NN cat)) (VP (VBZ sleeps)))")
        words, labels = tree_to_sequence_labels(tree)
        self.assertEqual(words, ["The", "cat", "sleeps"])
        self.assertEqual([label.n for label in labels], [2, -1, -1])
        self.assertEqual(labels[0].c, "NP")
        self.assertEqual(labels[1].c, "S")
        self.assertEqual(labels[2].c, END_C_LABEL)

    def test_unary_chain_extraction(self) -> None:
        tree = parse_bracket_tree("(S (NP (N (NNP John))) (VP (VBZ runs)))")
        _, labels = tree_to_sequence_labels(tree)
        self.assertEqual(labels[0].u, ("NP", "N", "NNP"))

    def test_roundtrip_tree_identity(self) -> None:
        tree = parse_bracket_tree("(S (NP (DT The) (NN cat)) (VP (VBZ sleeps)))")
        words, labels = tree_to_sequence_labels(tree)
        reconstructed = decode_sequence_labels(words, labels)
        self.assertEqual(tree.pretty(), reconstructed.pretty())

    def test_relative_n_consistency(self) -> None:
        tree = parse_bracket_tree("(S (NP (DT The) (NN cat)) (VP (VBZ sleeps)))")
        info = verify_relative_n_values(tree)
        self.assertTrue(bool(info["matches"]))

    def test_unknown_label_vocab_fallback(self) -> None:
        train_labels = [SequenceLabel(n=1, c="NP", u=()), SequenceLabel(n=-1, c=END_C_LABEL, u=())]
        unseen = [SequenceLabel(n=-2, c="S", u=(".",))]
        atomic_vocab = AtomicLabelVocab(train_labels)
        self.assertEqual(atomic_vocab.encode(unseen), [atomic_vocab.unk_id])
        self.assertEqual(atomic_vocab.decode([atomic_vocab.unk_id])[0].c, UNKNOWN_LABEL)

    def test_tree_log_probability_matches_exact_derivation(self) -> None:
        rule_probs = {
            ("ROOT", ("S",), (False,)): 1.0,
            ("S", ("NP", "VP"), (False, False)): 0.5,
            ("NP", ("dog",), (True,)): 0.25,
            ("VP", ("barks",), (True,)): 0.125,
        }
        tree = parse_bracket_tree("(ROOT (S (NP dog) (VP barks)))")
        observed = tree_log_probability(tree, rule_probs)
        expected = math.log(1.0) + math.log(0.5) + math.log(0.25) + math.log(0.125)
        self.assertAlmostEqual(observed, expected, places=8)

    def test_probability_grouping_orders_trees_by_log_probability(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            grammar_path = Path(tmp_dir) / "toy_grammar_for_prob_groups.txt"
            grammar_path.write_text(
                "\n".join(
                    [
                        "ROOT -> [S] : 1.0",
                        "S -> [NP VP] : 0.6",
                        "S -> [VP] : 0.4",
                        "NP -> [_dog] : 0.5",
                        "VP -> [_runs] : 0.3",
                        "VP -> [_sleeps] : 0.7",
                    ]
                ),
                encoding="utf-8",
            )
            groups = make_probability_sorted_groups(
                [
                    "(ROOT (S (NP dog) (VP sleeps)))",
                    "(ROOT (S (NP dog) (VP runs)))",
                    "(ROOT (S (VP sleeps)))",
                    "(ROOT (S (VP runs)))",
                ],
                grammar_path=grammar_path,
                n_groups=2,
            )
        self.assertEqual(len(groups), 2)
        self.assertGreaterEqual(
            float(groups[0].metadata["max_log_probability"]),
            float(groups[1].metadata["max_log_probability"]),
        )
        self.assertEqual(groups[0].group_type, "byProb")



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-tests", action="store_true", help="Run the built-in toy unit tests.")
    parser.add_argument(
        "--run-mode",
        default="individual",
        choices=["individual", "checkpoints", "layers", "layers_sorted_trees", "checkpoints_sorted_trees"],
    )
    parser.add_argument("--train-trees-path", default=str(DEFAULT_TRAIN_TREES_PATH))
    parser.add_argument("--dev-trees-path", default=str(DEFAULT_DEV_TREES_PATH))
    parser.add_argument("--test-trees-path", default=str(DEFAULT_TEST_TREES_PATH))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--layer", type=int, default=12, help="1-based transformer layer index, matching probe_core.py.")
    parser.add_argument("--train-trees", type=int, default=5)
    parser.add_argument("--dev-trees", type=int, default=5)
    parser.add_argument("--fixed-test-size", type=int, default=5)
    parser.add_argument(
        "--sorted-tree-groups",
        default="byProb",
        choices=["byDepth", "byProb"],
        help="How sorted-tree run modes partition the test trees.",
    )
    parser.add_argument(
        "--prob-group-count",
        type=int,
        default=10,
        help="Number of probability buckets used when --sorted-tree-groups=byProb.",
    )
    parser.add_argument(
        "--grammar-path",
        default="../data/grammar_mle.txt",
        help="Optional grammar file for byProb grouping. If omitted, the code tries to infer a matching grammar.",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Probe training batch size.")
    parser.add_argument("--extract-batch-size", type=int, default=8, help="LM hidden-state extraction batch size.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--model-init-seed", type=int, default=1, help="Seed for the random-init baseline model.")
    parser.add_argument("--device", default="auto", help="'auto', 'cpu', or 'cuda'.")
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--show-example", action="store_true", help="Save tree plots and print detailed debug info for one test example.")
    parser.add_argument("--debug-example-index", type=int, default=0, help="Test example index used for plotting and debug output.")
    parser.add_argument("--debug-failure-limit", type=int, default=3, help="How many decoding failures to print per model.")
    args = parser.parse_args()

    if args.run_tests:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(EncodingTests)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():
            raise SystemExit(1)
        return


    run_repo_probe(args)


if __name__ == "__main__":
    main()
