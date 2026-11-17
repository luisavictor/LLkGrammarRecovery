"""Build deterministic LL(k) grammars from ambiguous treebanks.

This is the main implementation for block A after a source PCFG/treebank exists.
It samples stored derivation trees, admits structural rules only when they do not
create LL(k) conflicts, prunes and renormalizes the resulting grammar, attaches
the source lexicon, and optionally generates Gaussian length-filtered
sentence/tree data for downstream LM training and probing.
"""

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple
from nltk import Nonterminal, PCFG, ProbabilisticProduction, Tree

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from A_PCFG.llk_parser import load_grammar, ROOT_SYMBOL

from A_PCFG.LLk_core import (
    count_conflicts,
    grammar_dict_from_pcfg,
    has_conflicts,
    index_productions,
    llk_conflict_map,
    prune_nonproductive_structural_backbone,
    prune_unreachable_and_renormalize,
    prune_unreachable_structural_backbone,
    save_sentence_tree_pairs,
    terminal_rules_from_grammar
)

Rule = Tuple[str, Tuple[str, ...]]
LexRule = Tuple[str, str]


def log(message: str) -> None:
    print(f"[llk-atomic] {message}", flush=True)





def structural_rule_weights_from_grammar(grammar: PCFG) -> Dict[Rule, float]:
    out: Dict[Rule, float] = {}
    for prod in grammar.productions():
        if any(isinstance(sym, Nonterminal) for sym in prod.rhs()):
            out[(str(prod.lhs()), tuple(str(sym) for sym in prod.rhs()))] = float(prod.prob())
    return out


def lexicon_from_grammar(grammar: PCFG) -> Dict[str, Dict[str, float]]:
    lexicon: Dict[str, Dict[str, float]] = defaultdict(dict)
    for (lhs, word), prob in terminal_rules_from_grammar(grammar).items():
        lexicon[lhs][word] = prob
    return dict(lexicon)


def merge_protected_lexicon(
    lexicon: Dict[str, Dict[str, float]],
    protected_lexicon: Optional[Dict[str, Dict[str, float]]] = None,
) -> Dict[str, Dict[str, float]]:
    if not protected_lexicon:
        return {
            lhs: dict(word_probs)
            for lhs, word_probs in lexicon.items()
            if word_probs
        }

    merged: Dict[str, Dict[str, float]] = {}
    protected_owner_by_word: Dict[str, str] = {}

    for lhs, word_probs in protected_lexicon.items():
        for word, weight in word_probs.items():
            if weight <= 0.0:
                raise RuntimeError(
                    f"Protected lexical rule {lhs!r} -> {word!r} has non-positive weight {weight}."
                )
            previous_owner = protected_owner_by_word.get(word)
            if previous_owner is not None and previous_owner != lhs:
                raise RuntimeError(
                    f"Protected terminal {word!r} is assigned to both {previous_owner!r} and {lhs!r}."
                )
            protected_owner_by_word[word] = lhs
            merged.setdefault(lhs, {})[word] = float(weight)

    for lhs, word_probs in lexicon.items():
        for word, weight in word_probs.items():
            protected_owner = protected_owner_by_word.get(word)
            if protected_owner is None:
                merged.setdefault(lhs, {})[word] = float(weight)
            elif protected_owner == lhs:
                merged.setdefault(lhs, {})

    return {
        lhs: word_probs
        for lhs, word_probs in merged.items()
        if word_probs
    }


def save_llk_grammar(grammar: PCFG, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for prod in grammar.productions():
            rhs_items: List[str] = []
            for sym in prod.rhs():
                rhs_items.append(f"_{sym}" if isinstance(sym, str) else str(sym))
            f.write(f"{prod.lhs()}->[{ ' '.join(rhs_items) }] : {prod.prob():.8f}\n")
    tmp_path.replace(path)


def sample_tree_lines_random(path: Path, num_trees: Optional[int], seed: int) -> List[str]:
    """Reservoir-sample tree lines uniformly without loading the whole file when num_trees is set."""
    rng = random.Random(seed)
    reservoir: List[str] = []
    seen = 0

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            seen += 1

            if num_trees is None or len(reservoir) < num_trees:
                reservoir.append(line)
                continue

            slot = rng.randrange(seen)
            if slot < num_trees:
                reservoir[slot] = line

    if not reservoir:
        raise RuntimeError(f"No tree lines found in {path}.")
    return reservoir


def sample_tree_lines_original(path: Path, num_trees: Optional[int]) -> List[str]:
    """Read tree lines in original file order, optionally stopping after num_trees non-empty lines."""
    lines: List[str] = []

    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            lines.append(line)
            if num_trees is not None and len(lines) >= num_trees:
                break

    if not lines:
        raise RuntimeError(f"No tree lines found in {path}.")
    return lines


def sample_tree_lines(
    path: Path,
    num_trees: Optional[int],
    seed: int,
    sample_mode: str,
) -> List[str]:
    if sample_mode == "random":
        return sample_tree_lines_random(path, num_trees, seed)
    if sample_mode == "original":
        return sample_tree_lines_original(path, num_trees)
    raise ValueError(f"Unknown tree_sample_mode: {sample_mode}")


def iter_trees_from_lines(lines: Iterable[str]) -> Iterator[Tree]:
    skipped = 0
    for index, line in enumerate(lines, start=1):
        try:
            yield Tree.fromstring(json.loads(line)["tree"])
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            skipped += 1
            if skipped <= 10:
                log(f"Skipping malformed sampled tree line {index}: {exc}")
            elif skipped == 11:
                log("Further malformed tree warnings suppressed.")


def tree_structural_rules_top_down(tree: Tree) -> List[Rule]:
    rules: List[Rule] = []

    def visit(node: Tree) -> None:
        if not isinstance(node, Tree) or len(node) == 0:
            return
        if any(isinstance(child, Tree) for child in node):
            lhs = str(node.label())
            rhs = tuple(str(child.label()) if isinstance(child, Tree) else child for child in node)
            rules.append((lhs, rhs))
            for child in node:
                if isinstance(child, Tree):
                    visit(child)

    visit(tree)
    return rules


def sample_tree_from_pcfg(
    grammar: PCFG,
    rng: random.Random,
    prod_table: Dict[Nonterminal, List[Tuple[Tuple[object, ...], float]]],
    max_depth: int,
) -> Tree:
    def expand(sym: object, depth: int = 0) -> object:
        if isinstance(sym, str):
            return sym
        if depth >= max_depth:
            raise RuntimeError(f"Maximum derivation depth {max_depth} reached while sampling.")
        rules = prod_table.get(sym, [])
        if not rules:
            raise RuntimeError(f"No productions available for symbol {sym}.")
        rhs, _ = rng.choices(rules, weights=[prob for _, prob in rules], k=1)[0]
        children = [expand(child, depth + 1) for child in rhs]
        return Tree(str(sym), children)

    sampled = expand(grammar.start())
    if not isinstance(sampled, Tree):
        raise RuntimeError("Sampled derivation did not produce a tree.")
    return sampled



def accept_sentence_by_length(sentence: Sequence[str], mean_len: float, stdev_len: float, rng: random.Random) -> bool:
    if stdev_len <= 0:
        raise ValueError("stdev_len must be positive for Gaussian length filtering.")
    length = len(sentence)
    prob = math.exp(-0.5 * ((length - mean_len) / stdev_len) ** 2)
    prob /= (stdev_len * math.sqrt(2 * math.pi))
    max_pdf = 1 / (stdev_len * math.sqrt(2 * math.pi))
    acceptance = prob / max_pdf
    return rng.random() < acceptance


def numeric_summary(values: Sequence[float]) -> Dict[str, object]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "stdev": None,
            "min": None,
            "median": None,
            "max": None,
        }

    sorted_values = sorted(values)
    count = len(sorted_values)
    mean_value = sum(sorted_values) / count
    variance = sum((value - mean_value) ** 2 for value in sorted_values) / count
    middle = count // 2
    if count % 2:
        median = sorted_values[middle]
    else:
        median = (sorted_values[middle - 1] + sorted_values[middle]) / 2
    return {
        "count": count,
        "mean": mean_value,
        "stdev": math.sqrt(variance),
        "min": sorted_values[0],
        "median": median,
        "max": sorted_values[-1],
    }


def write_json_file(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


def sample_sentence_length_summary(
    grammar: PCFG,
    *,
    requested_samples: int,
    seed: int,
    max_depth: int,
    max_attempts_per_sentence: int,
    max_total_attempts: int,
    generation_mode: str,
    mean_len: float,
    stdev_len: float,
    unique_sentences: bool,
) -> Dict[str, object]:
    sample_data = sample_sentences_from_grammar(
        grammar=grammar,
        n_sentences=requested_samples,
        seed=seed,
        max_depth=max_depth,
        max_attempts_per_sentence=max_attempts_per_sentence,
        max_total_attempts=max_total_attempts,
        mean_len=mean_len,
        stdev_len=stdev_len,
        progress_every_sentences=0,
        generation_mode=generation_mode,
        unique_sentences=unique_sentences,
    )
    lengths = [len(tree.leaves()) for tree in sample_data["trees"]]

    return {
        "requested_samples": requested_samples,
        "sampled_sentences": len(lengths),
        "sample_seed": seed,
        "generation_mode": generation_mode,
        "unique_sentences": unique_sentences,
        "max_depth": max_depth,
        "max_attempts_per_sentence": max_attempts_per_sentence,
        "max_total_attempts": max_total_attempts,
        "target_mean_len": mean_len,
        "target_stdev_len": stdev_len,
        "total_attempts": sample_data["total_attempts"],
        "depth_limited_attempts": sample_data["depth_limited_attempts"],
        "stopped_early": len(lengths) < requested_samples,
        "sentence_length_tokens": numeric_summary(lengths),
    }


def format_optional_float(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return str(value)


def log_sentence_length_summary(label: str, summary: Dict[str, object]) -> None:
    length_summary = summary["sentence_length_tokens"]
    if not isinstance(length_summary, dict):
        raise RuntimeError("Sentence length summary has an unexpected shape.")
    log(
        f"{label}: sampled "
        f"{summary['sampled_sentences']}/{summary['requested_samples']} current-grammar sentences "
        "without storing them; "
        f"length mean={format_optional_float(length_summary['mean'])}, "
        f"median={format_optional_float(length_summary['median'])}, "
        f"stdev={format_optional_float(length_summary['stdev'])}."
    )


def log_sentence_length_checkpoint(pruning_round: int, summary: Dict[str, object]) -> None:
    log_sentence_length_summary(f"Pruning round {pruning_round}", summary)


def build_generation_metadata(
    sampled_sentences: Sequence[str],
    sampled_trees: Sequence[Tree],
    *,
    requested_sentences: int,
    unique_sentences: bool,
    generation_mode: str,
    seed: int,
    max_depth: int,
    max_attempts_per_sentence: int,
    max_total_attempts: int,
    mean_len: float,
    stdev_len: float,
    total_attempts: int,
    stopped_early: bool,
    data_path: Path,
    data_trees_path: Path,
) -> Dict[str, object]:
    sentence_lengths = [len(tree.leaves()) for tree in sampled_trees]
    tree_heights = [tree.height() for tree in sampled_trees]
    tree_depths = [max(height - 1, 0) for height in tree_heights]
    tree_node_counts = [sum(1 for _subtree in tree.subtrees()) for tree in sampled_trees]
    structural_rule_counts = [
        len(tree_structural_rules_top_down(tree))
        for tree in sampled_trees
    ]
    lexical_rule_counts = [len(tree_terminal_rules(tree)) for tree in sampled_trees]

    return {
        "requested_generated_sentences": requested_sentences,
        "num_generated_sentences": len(sampled_sentences),
        "unique_sentences": unique_sentences,
        "num_unique_sentences": len(set(sampled_sentences)),
        "generation_mode": generation_mode,
        "generation_seed": seed,
        "generation_max_depth": max_depth,
        "generation_max_attempts_per_sentence": max_attempts_per_sentence,
        "generation_max_total_attempts": max_total_attempts,
        "generation_total_attempts": total_attempts,
        "generation_stopped_early": stopped_early,
        "target_mean_len": mean_len,
        "target_stdev_len": stdev_len,
        "data_path": str(data_path),
        "data_trees_path": str(data_trees_path),
        "sentence_length_tokens": numeric_summary(sentence_lengths),
        "tree_depth_edges": numeric_summary(tree_depths),
        "tree_height_nodes": numeric_summary(tree_heights),
        "tree_nodes": numeric_summary(tree_node_counts),
        "structural_rules_per_tree": numeric_summary(structural_rule_counts),
        "lexical_rules_per_tree": numeric_summary(lexical_rule_counts),
    }



def sample_sentences_from_grammar(
    *,
    grammar: PCFG,
    n_sentences: int,
    seed: int,
    max_depth: int,
    max_attempts_per_sentence: int,
    max_total_attempts: int,
    mean_len: float,
    stdev_len: float,
    progress_every_sentences: int,
    generation_mode: str = "gaussian",
    unique_sentences: bool = False,
) -> Dict[str, object]:
    rng = random.Random(seed)
    prod_table = index_productions(grammar)
    sampled_trees: List[Tree] = []
    sampled_sentences: List[str] = []
    seen_sentences: Set[str] = set()
    total_attempts = 0
    depth_limited_attempts = 0
    stopped_early = False

    while len(sampled_sentences) < n_sentences:
        if total_attempts >= max_total_attempts:
            stopped_early = True
            break
        tree = None
        for _attempt in range(max_attempts_per_sentence):
            if total_attempts >= max_total_attempts:
                break
            total_attempts += 1
            try:
                candidate_tree = sample_tree_from_pcfg(
                    grammar,
                    rng=rng,
                    prod_table=prod_table,
                    max_depth=max_depth,
                )
            except RuntimeError as exc:
                if "Maximum derivation depth" not in str(exc):
                    raise
                depth_limited_attempts += 1
                continue
            if generation_mode == "normal":
                tree = candidate_tree
                break
            if generation_mode == "gaussian":
                if accept_sentence_by_length(candidate_tree.leaves(), mean_len, stdev_len, rng=rng):
                    tree = candidate_tree
                    break
                continue
            raise ValueError(f"Unknown generation_mode: {generation_mode}")
        if tree is None:
            stopped_early = True
            break
        sentence = " ".join(tree.leaves())
        if unique_sentences and sentence in seen_sentences:
            continue
        seen_sentences.add(sentence)
        sampled_trees.append(tree)
        sampled_sentences.append(sentence)
        if progress_every_sentences > 0 and len(sampled_sentences) % progress_every_sentences == 0:
            log(
                f"Generated {len(sampled_sentences)}/{n_sentences} sentences "
                f"after {total_attempts} derivation attempts."
            )

    return {
        "sentences": sampled_sentences,
        "trees": sampled_trees,
        "total_attempts": total_attempts,
        "depth_limited_attempts": depth_limited_attempts,
        "stopped_early": stopped_early,
    }


def generate_sentences_from_grammar(
    grammar: PCFG,
    n_sentences: int,
    output_dir: Path,
    seed: int,
    max_depth: int,
    max_attempts_per_sentence: int,
    max_total_attempts: int,
    mean_len: float,
    stdev_len: float,
    progress_every_sentences: int,
    generation_mode: str = "gaussian",
    unique_sentences: bool = False,
) -> Dict[str, object]:
    log("Generating sentences from the final LL(k) grammar.")
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_data = sample_sentences_from_grammar(
        grammar=grammar,
        n_sentences=n_sentences,
        seed=seed,
        max_depth=max_depth,
        max_attempts_per_sentence=max_attempts_per_sentence,
        max_total_attempts=max_total_attempts,
        mean_len=mean_len,
        stdev_len=stdev_len,
        progress_every_sentences=progress_every_sentences,
        generation_mode=generation_mode,
        unique_sentences=unique_sentences,
    )
    sampled_sentences = sample_data["sentences"]
    sampled_trees = sample_data["trees"]
    total_attempts = sample_data["total_attempts"]
    stopped_early = sample_data["stopped_early"]
    if stopped_early:
        log("Generation stopped early before the requested number of sentences was produced.")

    data_path = output_dir / "data.txt"
    data_trees_path = output_dir / "data_trees.txt"
    metadata_path = output_dir / "generation_metadata.json"
    data_path.write_text("\n".join(sampled_sentences), encoding="utf-8")
    save_sentence_tree_pairs(sampled_sentences, sampled_trees, data_trees_path)
    metadata = build_generation_metadata(
        sampled_sentences,
        sampled_trees,
        requested_sentences=n_sentences,
        unique_sentences=unique_sentences,
        generation_mode=generation_mode,
        seed=seed,
        max_depth=max_depth,
        max_attempts_per_sentence=max_attempts_per_sentence,
        max_total_attempts=max_total_attempts,
        mean_len=mean_len,
        stdev_len=stdev_len,
        total_attempts=total_attempts,
        stopped_early=stopped_early,
        data_path=data_path,
        data_trees_path=data_trees_path,
    )
    write_json_file(metadata_path, metadata)
    log("Sentence generation finished.")
    log(f"Wrote generation metadata to {metadata_path}")
    return {
        "num_generated_sentences": len(sampled_sentences),
        "requested_generated_sentences": n_sentences,
        "unique_sentences": unique_sentences,
        "generation_total_attempts": total_attempts,
        "generation_stopped_early": stopped_early,
        "data_path": str(data_path),
        "data_trees_path": str(data_trees_path),
        "metadata_path": str(metadata_path),
        "metadata": metadata,
        "trees": sampled_trees,
    }



def tree_terminal_rules(tree: Tree) -> List[LexRule]:
    return [
        (str(subtree.label()), subtree[0])
        for subtree in tree.subtrees()
        if len(subtree) == 1 and isinstance(subtree[0], str)
    ]


def source_structural_rules(source_grammar: PCFG) -> Set[Rule]:
    return {
        (str(prod.lhs()), tuple(str(sym) for sym in prod.rhs()))
        for prod in source_grammar.productions()
        if any(isinstance(sym, Nonterminal) for sym in prod.rhs())
    }


def choose_disjoint_lexicon_from_source_grammar(
    source_grammar: PCFG,
    allowed_preterminals: Set[str],
) -> Dict[str, Dict[str, float]]:
    by_word: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

    for (lhs, word), prob in terminal_rules_from_grammar(source_grammar).items():
        if lhs in allowed_preterminals and prob > 0.0:
            by_word[word].append((lhs, float(prob)))

    assignment: Dict[str, Dict[str, float]] = defaultdict(dict)
    for word, entries in by_word.items():
        best_lhs, best_prob = max(entries, key=lambda item: (item[1], item[0]))
        assignment[best_lhs][word] = best_prob

    return dict(assignment)


def build_schema(structural_rules: Iterable[Rule], preterminals: Set[str]) -> Dict[str, List[Tuple[str, ...]]]:
    schema: Dict[str, List[Tuple[str, ...]]] = defaultdict(list)
    for lhs, rhs in structural_rules:
        schema[lhs].append(rhs)
    for preterminal in preterminals:
        schema[preterminal].append((f"__lex_{preterminal}",))
    return dict(schema)


def summarize_conflicts(conflicts: Dict[str, List[Dict[str, object]]]) -> Dict[str, object]:
    conflicting_rules: Set[Rule] = set()
    for lhs, entries in conflicts.items():
        for entry in entries:
            conflicting_rules.add((lhs, tuple(entry["alt1_rhs"])))
            conflicting_rules.add((lhs, tuple(entry["alt2_rhs"])))

    return {
        "num_conflicts": count_conflicts(conflicts),
        "num_conflicting_lhs": len(conflicts),
        "num_conflicting_rules": len(conflicting_rules),
        "conflicts_by_lhs": {
            lhs: len(entries)
            for lhs, entries in sorted(conflicts.items())
        },
    }


def validate_disjoint_lexicon(lexicon: Dict[str, Dict[str, float]]) -> None:
    owner_by_word: Dict[str, str] = {}
    for lhs, word_probs in lexicon.items():
        if not word_probs:
            raise RuntimeError(f"Lexical preterminal {lhs!r} has no assigned terminals.")
        for word, weight in word_probs.items():
            if weight <= 0.0:
                raise RuntimeError(
                    f"Lexical rule {lhs!r} -> {word!r} has non-positive weight {weight}."
                )
            previous_owner = owner_by_word.get(word)
            if previous_owner is not None and previous_owner != lhs:
                raise RuntimeError(
                    f"Terminal {word!r} is assigned to both {previous_owner!r} and {lhs!r}."
                )
            owner_by_word[word] = lhs


def build_llk_superset_metadata(
    final_schema: Dict[str, List[Tuple[str, ...]]],
    final_grammar: PCFG,
    final_rules: Set[Rule],
    final_lexicon: Dict[str, Dict[str, float]],
    *,
    root_symbol: str,
    lookahead_k: int,
    include_lexicalized_conflicts: bool = False,
) -> Dict[str, object]:
    final_grammar_dict = grammar_dict_from_pcfg(final_grammar) if include_lexicalized_conflicts else None
    structural_rules_by_lhs = Counter(lhs for lhs, _rhs in final_rules)
    lexical_rule_count = sum(len(words) for words in final_lexicon.values())
    skipped_lexicalized_conflicts = {
        "computed": False,
        "reason": (
            "Skipped by default because pairwise LL(k) conflict checks over lexicalized "
            "preterminal alternatives can be very expensive; the dummy schema is validated "
            "and the lexicon is checked to be disjoint."
        ),
    }

    rule_inventory = {
        "num_structural_rules": len(final_rules),
        "num_preterminals": len(final_lexicon),
        "num_dummy_schema_rules": len(final_rules) + len(final_lexicon),
        "num_lexical_rules": lexical_rule_count,
        "num_total_pcfg_productions": len(final_grammar.productions()),
        "structural_rules_by_lhs": {
            lhs: count
            for lhs, count in sorted(structural_rules_by_lhs.items())
        },
    }

    smaller_lookaheads: List[Dict[str, object]] = []
    for candidate_k in range(1, lookahead_k):
        dummy_conflicts = llk_conflict_map(final_schema, root_symbol, candidate_k)
        lexicalized_conflicts = (
            llk_conflict_map(final_grammar_dict, root_symbol, candidate_k)
            if final_grammar_dict is not None
            else None
        )
        smaller_lookaheads.append(
            {
                "lookahead_k": candidate_k,
                "is_llk_dummy_schema": not has_conflicts(dummy_conflicts),
                "dummy_schema_conflicts": summarize_conflicts(dummy_conflicts),
                "is_llk_lexicalized_grammar": (
                    not has_conflicts(lexicalized_conflicts)
                    if lexicalized_conflicts is not None
                    else None
                ),
                "lexicalized_grammar_conflicts": (
                    summarize_conflicts(lexicalized_conflicts)
                    if lexicalized_conflicts is not None
                    else skipped_lexicalized_conflicts
                ),
                "num_structural_rules": rule_inventory["num_structural_rules"],
                "num_dummy_schema_rules": rule_inventory["num_dummy_schema_rules"],
                "num_total_pcfg_productions": rule_inventory["num_total_pcfg_productions"],
            }
        )

    target_dummy_conflicts = llk_conflict_map(final_schema, root_symbol, lookahead_k)
    target_lexicalized_conflicts = (
        llk_conflict_map(final_grammar_dict, root_symbol, lookahead_k)
        if final_grammar_dict is not None
        else None
    )

    return {
        "target_lookahead_k": lookahead_k,
        "root_symbol": root_symbol,
        "rule_inventory": rule_inventory,
        "target_lookahead_validation": {
            "is_llk_dummy_schema": not has_conflicts(target_dummy_conflicts),
            "dummy_schema_conflicts": summarize_conflicts(target_dummy_conflicts),
            "is_llk_lexicalized_grammar": (
                not has_conflicts(target_lexicalized_conflicts)
                if target_lexicalized_conflicts is not None
                else None
            ),
            "lexicalized_grammar_conflicts": (
                summarize_conflicts(target_lexicalized_conflicts)
                if target_lexicalized_conflicts is not None
                else skipped_lexicalized_conflicts
            ),
        },
        "smaller_lookahead_diagnostics": smaller_lookaheads,
    }


def write_metadata_file(
    metadata_path: Path,
    metadata_update: Dict[str, object],
    *,
    remove_keys: Iterable[str] = (),
) -> None:
    metadata: Dict[str, object] = {}
    if metadata_path.exists():
        try:
            loaded = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                metadata.update(loaded)
        except json.JSONDecodeError:
            metadata["previous_metadata_parse_error"] = True

    for key in remove_keys:
        metadata.pop(key, None)
    metadata.update(metadata_update)
    write_json_file(metadata_path, metadata)


def exceeds_max_rules_per_lhs(
    accepted_rules: Set[Rule],
    new_rules: Set[Rule],
    max_rules_per_lhs: int,
) -> bool:
    counts = Counter(lhs for lhs, _rhs in accepted_rules)
    for lhs, _rhs in new_rules:
        counts[lhs] += 1
        if counts[lhs] > max_rules_per_lhs:
            return True
    return False


def accept_tree_prefix(
    accepted_rules: Set[Rule],
    tree_rules: List[Rule],
    known_preterminals: Set[str],
    tree_preterminals: Set[str],
    source_rules: Set[Rule],
    lookahead_k: int,
    max_rules_per_lhs: int,
    root_symbol: str,
) -> Tuple[Set[Rule], List[Rule]]:
    candidate_rules = set(accepted_rules)
    candidate_preterminals = set(known_preterminals) | set(tree_preterminals)
    local_added: List[Rule] = []

    for rule in tree_rules:
        if rule in candidate_rules:
            continue
        if rule not in source_rules:
            break
        if exceeds_max_rules_per_lhs(candidate_rules, {rule}, max_rules_per_lhs):
            break

        proposed_rules = candidate_rules | {rule}
        if not any(lhs == root_symbol for lhs, _rhs in proposed_rules):
            break
        schema = build_schema(proposed_rules, candidate_preterminals)
        conflicts = llk_conflict_map(schema, root_symbol, lookahead_k)
        if has_conflicts(conflicts):
            break

        candidate_rules = proposed_rules
        local_added.append(rule)

    return candidate_rules, local_added


def prune_accepted_state(
    source_grammar: PCFG,
    accepted_rules: Set[Rule],
    structural_counts: Counter,
    terminal_counts: Counter,
    root_symbol: str,
    protected_rules: Optional[Set[Rule]] = None,
    protected_lexicon: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[Set[Rule], Counter, Counter]:
    protected_rules = set(protected_rules or set())
    protected_lexicon = protected_lexicon or {}

    observed_preterminals = {lhs for lhs, _word in terminal_counts} | set(protected_lexicon)
    lexicon = merge_protected_lexicon(
        choose_disjoint_lexicon_from_source_grammar(source_grammar, observed_preterminals),
        protected_lexicon,
    )
    productive_preterminals = set(lexicon)

    reachable_rules, _reachable_nts, _reachable_pts = prune_unreachable_structural_backbone(
        accepted_rules,
        productive_preterminals,
        root_symbol,
    )
    productive_rules, _productive_symbols = prune_nonproductive_structural_backbone(
        reachable_rules,
        productive_preterminal_labels=productive_preterminals,
    )
    final_rules, _reachable_nts, reachable_preterminals = prune_unreachable_structural_backbone(
        productive_rules,
        productive_preterminals,
        root_symbol,
    )
    final_rules = set(final_rules)

    missing_protected_rules = protected_rules - final_rules
    if missing_protected_rules:
        preview = sorted(missing_protected_rules)[:5]
        raise RuntimeError(
            "Initial grammar backbone would be removed by pruning. "
            f"First missing protected structural rules: {preview}"
        )

    pruned_structural_counts = Counter(
        {
            rule: structural_counts[rule] if structural_counts[rule] > 0 else 1.0
            for rule in final_rules
            if structural_counts[rule] > 0 or rule in protected_rules
        }
    )
    pruned_terminal_counts = Counter(
        {
            (lhs, word): count
            for (lhs, word), count in terminal_counts.items()
            if lhs in reachable_preterminals and count > 0
        }
    )
    for lhs, word_probs in protected_lexicon.items():
        if lhs not in reachable_preterminals:
            continue
        for word, weight in word_probs.items():
            if pruned_terminal_counts[(lhs, word)] <= 0:
                pruned_terminal_counts[(lhs, word)] = float(weight)

    return final_rules, pruned_structural_counts, pruned_terminal_counts


def finalize_rules_and_lexicon(
    source_grammar: PCFG,
    accepted_rules: Set[Rule],
    structural_counts: Counter,
    terminal_counts: Counter,
    root_symbol: str,
    protected_rules: Optional[Set[Rule]] = None,
    protected_lexicon: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[Set[Rule], Counter, Dict[str, Dict[str, float]]]:
    protected_lexicon = protected_lexicon or {}
    final_rules, final_counts, final_terminal_counts = prune_accepted_state(
        source_grammar,
        accepted_rules,
        structural_counts,
        terminal_counts,
        root_symbol,
        protected_rules=protected_rules,
        protected_lexicon=protected_lexicon,
    )

    reachable_preterminals = {lhs for lhs, _word in final_terminal_counts} | set(protected_lexicon)
    final_lexicon = merge_protected_lexicon(
        choose_disjoint_lexicon_from_source_grammar(source_grammar, reachable_preterminals),
        protected_lexicon,
    )

    reachable_rhs_symbols = {
        symbol
        for _lhs, rhs in final_rules
        for symbol in rhs
    }
    unreachable_protected_preterminals = set(protected_lexicon) - reachable_rhs_symbols
    if unreachable_protected_preterminals:
        raise RuntimeError(
            "Initial grammar lexical preterminals are not reachable from the final structural backbone: "
            f"{sorted(unreachable_protected_preterminals)[:10]}"
        )

    if not final_rules:
        raise RuntimeError("No structural rules remain after final pruning.")
    if not final_lexicon:
        raise RuntimeError("No lexical rules remain after final pruning.")

    return set(final_rules), final_counts, final_lexicon


def build_pcfg_from_counts(
    structural_counts: Counter,
    lexical_assignment: Dict[str, Dict[str, float]],
    root_symbol: str,
) -> PCFG:
    productions: List[ProbabilisticProduction] = []

    grouped: Dict[str, List[Tuple[Tuple[str, ...], float]]] = defaultdict(list)
    for (lhs, rhs), count in structural_counts.items():
        if count > 0:
            grouped[lhs].append((rhs, float(count)))

    for lhs, rhs_counts in grouped.items():
        total = sum(count for _rhs, count in rhs_counts)
        for rhs, count in rhs_counts:
            productions.append(
                ProbabilisticProduction(
                    Nonterminal(lhs),
                    tuple(Nonterminal(sym) for sym in rhs),
                    prob=count / total,
                )
            )

    for lhs, word_probs in lexical_assignment.items():
        total = sum(word_probs.values())
        for word, weight in sorted(word_probs.items()):
            productions.append(
                ProbabilisticProduction(Nonterminal(lhs), (word,), prob=weight / total)
            )

    if not productions:
        raise RuntimeError("No productions available for the final PCFG.")
    return PCFG(Nonterminal(root_symbol), productions)


def materialize_current_llk_grammar(
    source_grammar: PCFG,
    accepted_rules: Set[Rule],
    structural_counts: Counter,
    terminal_counts: Counter,
    root_symbol: str,
    lookahead_k: int,
    *,
    protected_rules: Optional[Set[Rule]] = None,
    protected_lexicon: Optional[Dict[str, Dict[str, float]]] = None,
    stage_label: str = "Current",
) -> Tuple[Set[Rule], Counter, Dict[str, Dict[str, float]], Dict[str, List[Tuple[str, ...]]], PCFG]:
    current_rules, current_counts, current_lexicon = finalize_rules_and_lexicon(
        source_grammar=source_grammar,
        accepted_rules=accepted_rules,
        structural_counts=structural_counts,
        terminal_counts=terminal_counts,
        root_symbol=root_symbol,
        protected_rules=protected_rules,
        protected_lexicon=protected_lexicon,
    )

    current_schema = build_schema(current_rules, set(current_lexicon))
    current_conflicts = llk_conflict_map(current_schema, root_symbol, lookahead_k)
    if has_conflicts(current_conflicts):
        raise RuntimeError(
            f"{stage_label} grammar is not LL({lookahead_k}); "
            f"found {count_conflicts(current_conflicts)} conflicts."
        )

    current_grammar = prune_unreachable_and_renormalize(
        build_pcfg_from_counts(current_counts, current_lexicon, root_symbol)
    )
    validate_disjoint_lexicon(current_lexicon)
    return current_rules, current_counts, current_lexicon, current_schema, current_grammar


def grammar_inventory(
    grammar: PCFG,
    structural_rules: Set[Rule],
    lexicon: Dict[str, Dict[str, float]],
) -> Dict[str, object]:
    structural_rules_by_lhs = Counter(lhs for lhs, _rhs in structural_rules)
    return {
        "num_structural_rules": len(structural_rules),
        "num_preterminals": len(lexicon),
        "num_lexical_rules": sum(len(words) for words in lexicon.values()),
        "num_total_pcfg_productions": len(grammar.productions()),
        "structural_rules_by_lhs": {
            lhs: count
            for lhs, count in sorted(structural_rules_by_lhs.items())
        },
    }


def build_pruning_checkpoint(
    *,
    source_grammar: PCFG,
    accepted_rules: Set[Rule],
    structural_counts: Counter,
    terminal_counts: Counter,
    root_symbol: str,
    lookahead_k: int,
    protected_rules: Set[Rule],
    protected_lexicon: Dict[str, Dict[str, float]],
    output_dir: Path,
    grammar_path: Path,
    trees_path: Path,
    num_trees: Optional[int],
    tree_sample_seed: int,
    tree_sample_mode: str,
    max_rules_per_lhs: int,
    init_grammar_path: Optional[Path],
    pruning_round: int,
    processed_trees: int,
    accepted_trees: int,
    generation_seed: int,
    generation_max_depth: int,
    checkpoint_sample_sentences: int,
    checkpoint_sample_max_attempts_per_sentence: int,
    checkpoint_sample_max_total_attempts: Optional[int],
    mean_len: float,
    stdev_len: float,
    prune_every_accepted_trees: int,
    progress_log_every_trees: int,
    generation_progress_every_sentences: int,
    protected_count_floor: float,
    sample_history: List[Dict[str, object]],
) -> None:
    (
        checkpoint_rules,
        _checkpoint_counts,
        checkpoint_lexicon,
        _checkpoint_schema,
        checkpoint_grammar,
    ) = materialize_current_llk_grammar(
        source_grammar=source_grammar,
        accepted_rules=accepted_rules,
        structural_counts=structural_counts,
        terminal_counts=terminal_counts,
        root_symbol=root_symbol,
        lookahead_k=lookahead_k,
        protected_rules=protected_rules,
        protected_lexicon=protected_lexicon,
        stage_label=f"Pruning round {pruning_round}",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    latest_grammar_path = output_dir / "grammar.txt"
    checkpoint_dir = output_dir / "pruning_checkpoints" / f"round_{pruning_round:05d}"
    checkpoint_grammar_path = checkpoint_dir / "grammar.txt"
    checkpoint_metadata_path = checkpoint_dir / "metadata.json"
    save_llk_grammar(checkpoint_grammar, latest_grammar_path)
    save_llk_grammar(checkpoint_grammar, checkpoint_grammar_path)

    sample_summary = sample_sentence_length_summary(
        checkpoint_grammar,
        requested_samples=checkpoint_sample_sentences,
        seed=generation_seed + pruning_round,
        max_depth=generation_max_depth,
        max_attempts_per_sentence=checkpoint_sample_max_attempts_per_sentence,
        max_total_attempts=(
            checkpoint_sample_max_total_attempts
            if checkpoint_sample_max_total_attempts is not None
            else checkpoint_sample_sentences * checkpoint_sample_max_attempts_per_sentence
        ),
        generation_mode="gaussian",
        mean_len=mean_len,
        stdev_len=stdev_len,
        unique_sentences=True,
    )
    log_sentence_length_checkpoint(pruning_round, sample_summary)

    target_conflicts: Dict[str, List[Dict[str, object]]] = {}
    checkpoint_inventory = grammar_inventory(
        checkpoint_grammar,
        checkpoint_rules,
        checkpoint_lexicon,
    )
    checkpoint_summary = {
        "pruning_round": pruning_round,
        "processed_trees": processed_trees,
        "accepted_trees": accepted_trees,
        "latest_grammar_path": str(latest_grammar_path),
        "checkpoint_grammar_path": str(checkpoint_grammar_path),
        "checkpoint_metadata_path": str(checkpoint_metadata_path),
        "num_structural_rules": checkpoint_inventory["num_structural_rules"],
        "num_preterminals": checkpoint_inventory["num_preterminals"],
        "num_lexical_rules": checkpoint_inventory["num_lexical_rules"],
        "num_total_pcfg_productions": checkpoint_inventory["num_total_pcfg_productions"],
        "sentence_length_tokens": sample_summary["sentence_length_tokens"],
        "sampled_sentences": sample_summary["sampled_sentences"],
        "requested_samples": sample_summary["requested_samples"],
        "sample_stopped_early": sample_summary["stopped_early"],
    }
    sample_history.append(checkpoint_summary)

    checkpoint_metadata = {
        "checkpoint_type": "pruning_round",
        "checkpoint_summary": checkpoint_summary,
        "build_state": {
            "input_grammar": str(grammar_path),
            "input_trees": str(trees_path),
            "lookahead_k": lookahead_k,
            "num_trees_requested": num_trees,
            "tree_sample_seed": tree_sample_seed,
            "tree_sample_mode": tree_sample_mode,
            "tree_processing_order": tree_sample_mode,
            "max_rules_per_lhs": max_rules_per_lhs,
            "checkpoint_sample_sentences": checkpoint_sample_sentences,
            "checkpoint_sample_max_attempts_per_sentence": checkpoint_sample_max_attempts_per_sentence,
            "checkpoint_sample_max_total_attempts": checkpoint_sample_max_total_attempts,
            "checkpoint_sampling_mode": "gaussian",
            "checkpoint_unique_sentences": True,
            "prune_every_accepted_trees": prune_every_accepted_trees,
            "progress_log_every_trees": progress_log_every_trees,
            "generation_progress_every_sentences": generation_progress_every_sentences,
            "protected_count_floor": protected_count_floor,
            "init_grammar_path": str(init_grammar_path) if init_grammar_path is not None else None,
            "init_grammar_was_used": init_grammar_path is not None,
            "processed_trees": processed_trees,
            "accepted_trees": accepted_trees,
            "accepted_trees_definition": "Trees contributing at least one rule already present in the accepted grammar after prefix admission.",
            "output_grammar_path": str(latest_grammar_path),
        },
        "grammar_paths": {
            "latest_grammar_path": str(latest_grammar_path),
            "checkpoint_grammar_path": str(checkpoint_grammar_path),
        },
        "grammar_inventory": checkpoint_inventory,
        "target_lookahead_validation": {
            "lookahead_k": lookahead_k,
            "is_llk_dummy_schema": not has_conflicts(target_conflicts),
            "dummy_schema_conflicts": summarize_conflicts(target_conflicts),
            "lexicon_is_disjoint": True,
        },
        "sampling_diagnostic": sample_summary,
    }
    write_json_file(checkpoint_metadata_path, checkpoint_metadata)

    metadata_path = output_dir / "generation_metadata.json"
    write_metadata_file(
        metadata_path,
        {
            "build_summary": checkpoint_metadata["build_state"],
            "generation_was_run": False,
            "latest_pruning_checkpoint": checkpoint_summary,
            "num_pruning_checkpoints": len(sample_history),
            "pruning_checkpoints_dir": str(output_dir / "pruning_checkpoints"),
            "metadata_path": str(metadata_path),
        },
        remove_keys=("pruning_checkpoint_history",),
    )
    log(
        f"Checkpointed pruning round {pruning_round} grammar to {latest_grammar_path} "
        f"and {checkpoint_grammar_path}."
    )


def build_llk_atomic_tree_grammar(
    grammar_path: Path,
    trees_path: Path,
    output_dir: Path,
    lookahead_k: int,
    num_trees: Optional[int],
    tree_sample_seed: int,
    tree_sample_mode: str,
    max_rules_per_lhs: int,
    root_symbol: str,
    init_grammar_path: Optional[Path],
    generate_n: int,
    generation_seed: int,
    generation_max_depth: int,
    generation_max_attempts_per_sentence: int,
    generation_max_total_attempts: int,
    mean_len: float,
    stdev_len: float,
    checkpoint_sample_sentences: int,
    checkpoint_sample_max_attempts_per_sentence: int,
    checkpoint_sample_max_total_attempts: Optional[int],
    prune_every_accepted_trees: int,
    progress_log_every_trees: int,
    generation_progress_every_sentences: int,
    protected_count_floor: float,
) -> None:
    if lookahead_k < 1:
        raise ValueError("lookahead_k must be at least 1.")
    if max_rules_per_lhs < 1:
        raise ValueError("max_rules_per_lhs must be at least 1.")
    if tree_sample_mode not in {"random", "original"}:
        raise ValueError("tree_sample_mode must be either 'random' or 'original'.")
    if checkpoint_sample_sentences < 1:
        raise ValueError("checkpoint_sample_sentences must be at least 1.")
    if checkpoint_sample_max_attempts_per_sentence < 1:
        raise ValueError("checkpoint_sample_max_attempts_per_sentence must be at least 1.")
    if checkpoint_sample_max_total_attempts is not None and checkpoint_sample_max_total_attempts < 1:
        raise ValueError("checkpoint_sample_max_total_attempts must be at least 1 when set.")
    if prune_every_accepted_trees < 1:
        raise ValueError("prune_every_accepted_trees must be at least 1.")
    if progress_log_every_trees < 0:
        raise ValueError("progress_log_every_trees must be non-negative.")
    if generation_progress_every_sentences < 0:
        raise ValueError("generation_progress_every_sentences must be non-negative.")
    if protected_count_floor <= 0:
        raise ValueError("protected_count_floor must be positive.")

    log("Loading source grammar.")
    source_grammar = load_grammar(grammar_path)
    source_rules = source_structural_rules(source_grammar)

    protected_structural_weights: Dict[Rule, float] = {}
    protected_rules: Set[Rule] = set()
    protected_lexicon: Dict[str, Dict[str, float]] = {}
    protected_lexical_rules: Set[LexRule] = set()

    if init_grammar_path is not None:
        log(f"Loading initial grammar backbone from {init_grammar_path}.")
        init_grammar = load_grammar(init_grammar_path)
        protected_structural_weights = structural_rule_weights_from_grammar(init_grammar)
        protected_rules = set(protected_structural_weights)
        protected_lexicon = lexicon_from_grammar(init_grammar)
        protected_lexical_rules = {
            (lhs, word)
            for lhs, word_probs in protected_lexicon.items()
            for word in word_probs
        }

        if not protected_rules:
            raise RuntimeError(f"Initial grammar {init_grammar_path} contains no structural rules.")
        if not protected_lexicon:
            raise RuntimeError(f"Initial grammar {init_grammar_path} contains no lexical rules.")
        if not any(lhs == root_symbol for lhs, _rhs in protected_rules):
            raise RuntimeError(
                f"Initial grammar {init_grammar_path} does not contain root symbol {root_symbol!r}."
            )
        validate_disjoint_lexicon(protected_lexicon)

        init_schema = build_schema(protected_rules, set(protected_lexicon))
        init_conflicts = llk_conflict_map(init_schema, root_symbol, lookahead_k)
        if has_conflicts(init_conflicts):
            raise RuntimeError(
                f"Initial grammar backbone is not LL({lookahead_k}); "
                f"found {count_conflicts(init_conflicts)} conflicts."
            )

        source_rules |= protected_rules
        log(
            "Keeping initial grammar fixed: "
            f"{len(protected_rules)} structural rules and "
            f"{len(protected_lexical_rules)} lexical rules."
        )

    log(f"Sampling trees in {tree_sample_mode} mode.")
    sampled_lines = sample_tree_lines(
        trees_path,
        num_trees,
        tree_sample_seed,
        tree_sample_mode,
    )

    accepted_rules: Set[Rule] = set(protected_rules)
    structural_counts: Counter = Counter(
        {
            rule: max(weight, protected_count_floor)
            for rule, weight in protected_structural_weights.items()
        }
    )
    terminal_counts: Counter = Counter(
        {
            (lhs, word): max(weight, protected_count_floor)
            for lhs, word_probs in protected_lexicon.items()
            for word, weight in word_probs.items()
        }
    )
    processed_trees = 0
    accepted_trees = 0
    pruning_rounds = 0
    pruning_checkpoint_history: List[Dict[str, object]] = []

    log("Processing sampled trees with top-down prefix LL(k) admission.")
    for tree in iter_trees_from_lines(sampled_lines):
        processed_trees += 1
        tree_rules = tree_structural_rules_top_down(tree)
        tree_lexical_rules = tree_terminal_rules(tree)
        tree_preterminals = {lhs for lhs, _word in tree_lexical_rules}

        candidate_rules, local_added = accept_tree_prefix(
            accepted_rules=accepted_rules,
            tree_rules=tree_rules,
            known_preterminals={lhs for lhs, _word in terminal_counts},
            tree_preterminals=tree_preterminals,
            source_rules=source_rules,
            lookahead_k=lookahead_k,
            max_rules_per_lhs=max_rules_per_lhs,
            root_symbol=root_symbol,
        )

        if local_added:
            accepted_rules = candidate_rules

        accepted_tree_rules = [rule for rule in tree_rules if rule in accepted_rules]
        if accepted_tree_rules:
            structural_counts.update(accepted_tree_rules)
            accepted_preterminals = {
                symbol
                for _lhs, rhs in accepted_tree_rules
                for symbol in rhs
            }
            terminal_counts.update(
                (lhs, word)
                for lhs, word in tree_lexical_rules
                if lhs in accepted_preterminals
            )
            accepted_trees += 1

            if accepted_trees % prune_every_accepted_trees == 0:
                before_rules = len(accepted_rules)
                pruning_rounds += 1
                accepted_rules, structural_counts, terminal_counts = prune_accepted_state(
                    source_grammar,
                    accepted_rules,
                    structural_counts,
                    terminal_counts,
                    root_symbol,
                    protected_rules=protected_rules,
                    protected_lexicon=protected_lexicon,
                )
                log(
                    f"Pruning round {pruning_rounds} after {accepted_trees} accepted trees; "
                    f"structural rules {before_rules}->{len(accepted_rules)}."
                )
                build_pruning_checkpoint(
                    source_grammar=source_grammar,
                    accepted_rules=accepted_rules,
                    structural_counts=structural_counts,
                    terminal_counts=terminal_counts,
                    root_symbol=root_symbol,
                    lookahead_k=lookahead_k,
                    protected_rules=protected_rules,
                    protected_lexicon=protected_lexicon,
                    output_dir=output_dir,
                    grammar_path=grammar_path,
                    trees_path=trees_path,
                    num_trees=num_trees,
                    tree_sample_seed=tree_sample_seed,
                    tree_sample_mode=tree_sample_mode,
                    max_rules_per_lhs=max_rules_per_lhs,
                    init_grammar_path=init_grammar_path,
                    pruning_round=pruning_rounds,
                    processed_trees=processed_trees,
                    accepted_trees=accepted_trees,
                    generation_seed=generation_seed,
                    generation_max_depth=generation_max_depth,
                    checkpoint_sample_sentences=checkpoint_sample_sentences,
                    checkpoint_sample_max_attempts_per_sentence=checkpoint_sample_max_attempts_per_sentence,
                    checkpoint_sample_max_total_attempts=checkpoint_sample_max_total_attempts,
                    mean_len=mean_len,
                    stdev_len=stdev_len,
                    prune_every_accepted_trees=prune_every_accepted_trees,
                    progress_log_every_trees=progress_log_every_trees,
                    generation_progress_every_sentences=generation_progress_every_sentences,
                    protected_count_floor=protected_count_floor,
                    sample_history=pruning_checkpoint_history,
                )

        if progress_log_every_trees > 0 and processed_trees % progress_log_every_trees == 0:
            current_preterminals = {lhs for lhs, _word in terminal_counts}
            log(
                f"Processed {processed_trees} trees; accepted {accepted_trees}; "
                f"current LL(k) schema rules={len(accepted_rules) + len(current_preterminals)}; "
                f"structural rules={len(accepted_rules)}; "
                f"preterminals={len(current_preterminals)}."
            )

    if processed_trees == 0:
        raise RuntimeError("No valid trees were processed.")
    if not accepted_rules:
        raise RuntimeError("No tree could be accepted under the given constraints.")

    log("Running final reachability/productivity pruning.")
    final_rules, final_counts, final_lexicon = finalize_rules_and_lexicon(
        source_grammar=source_grammar,
        accepted_rules=accepted_rules,
        structural_counts=structural_counts,
        terminal_counts=terminal_counts,
        root_symbol=root_symbol,
        protected_rules=protected_rules,
        protected_lexicon=protected_lexicon,
    )

    log("Running final LL(k) validation.")
    final_schema = build_schema(final_rules, set(final_lexicon))
    final_conflicts = llk_conflict_map(final_schema, root_symbol, lookahead_k)
    if has_conflicts(final_conflicts):
        raise RuntimeError(
            f"Final grammar is not LL({lookahead_k}); found {count_conflicts(final_conflicts)} conflicts."
        )

    final_grammar = prune_unreachable_and_renormalize(
        build_pcfg_from_counts(final_counts, final_lexicon, root_symbol)
    )
    validate_disjoint_lexicon(final_lexicon)
    if protected_rules or protected_lexical_rules:
        final_structural_rules = set(structural_rule_weights_from_grammar(final_grammar))
        final_lexical_rules = set(terminal_rules_from_grammar(final_grammar))
        missing_structural_rules = protected_rules - final_structural_rules
        missing_lexical_rules = protected_lexical_rules - final_lexical_rules
        if missing_structural_rules or missing_lexical_rules:
            raise RuntimeError(
                "Final grammar does not preserve the complete initial grammar backbone. "
                f"Missing structural rules: {sorted(missing_structural_rules)[:5]}; "
                f"missing lexical rules: {sorted(missing_lexical_rules)[:5]}."
            )

    final_grammar_dict = grammar_dict_from_pcfg(final_grammar)
    final_lexicalized_conflicts = llk_conflict_map(
        final_grammar_dict,
        root_symbol,
        lookahead_k,
    )
    if has_conflicts(final_lexicalized_conflicts):
        raise RuntimeError(
            f"Final lexicalized grammar is not LL({lookahead_k}); "
            f"found {count_conflicts(final_lexicalized_conflicts)} conflicts."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "grammar.txt"
    save_llk_grammar(final_grammar, output_path)
    final_sample_summary = sample_sentence_length_summary(
        final_grammar,
        requested_samples=checkpoint_sample_sentences,
        seed=generation_seed + pruning_rounds + 1,
        max_depth=generation_max_depth,
        max_attempts_per_sentence=checkpoint_sample_max_attempts_per_sentence,
        max_total_attempts=(
            checkpoint_sample_max_total_attempts
            if checkpoint_sample_max_total_attempts is not None
            else checkpoint_sample_sentences * checkpoint_sample_max_attempts_per_sentence
        ),
        generation_mode="gaussian",
        mean_len=mean_len,
        stdev_len=stdev_len,
        unique_sentences=True,
    )
    log_sentence_length_summary("Final pre-generation grammar", final_sample_summary)

    metadata_path = output_dir / "generation_metadata.json"
    final_metadata_update = {
        "build_summary": {
            "input_grammar": str(grammar_path),
            "input_trees": str(trees_path),
            "lookahead_k": lookahead_k,
            "num_trees_requested": num_trees,
            "tree_sample_seed": tree_sample_seed,
            "tree_sample_mode": tree_sample_mode,
            "tree_processing_order": tree_sample_mode,
            "max_rules_per_lhs": max_rules_per_lhs,
            "checkpoint_sample_sentences": checkpoint_sample_sentences,
            "checkpoint_sample_max_attempts_per_sentence": checkpoint_sample_max_attempts_per_sentence,
            "checkpoint_sample_max_total_attempts": checkpoint_sample_max_total_attempts,
            "checkpoint_sampling_mode": "gaussian",
            "checkpoint_unique_sentences": True,
            "prune_every_accepted_trees": prune_every_accepted_trees,
            "progress_log_every_trees": progress_log_every_trees,
            "generation_progress_every_sentences": generation_progress_every_sentences,
            "protected_count_floor": protected_count_floor,
            "init_grammar_path": str(init_grammar_path) if init_grammar_path is not None else None,
            "init_grammar_was_used": init_grammar_path is not None,
            "protected_structural_rules": len(protected_rules),
            "protected_lexical_rules": len(protected_lexical_rules),
            "processed_trees": processed_trees,
            "accepted_trees": accepted_trees,
            "pruning_rounds": pruning_rounds,
            "accepted_trees_definition": "Trees contributing at least one rule already present in the accepted grammar after prefix admission.",
            "output_grammar_path": str(output_path),
        },
        "grammar_diagnostics": build_llk_superset_metadata(
            final_schema,
            final_grammar,
            final_rules,
            final_lexicon,
            root_symbol=root_symbol,
            lookahead_k=lookahead_k,
        ),
        "metadata_path": str(metadata_path),
        "generation_was_run": False,
        "latest_pruning_checkpoint": (
            pruning_checkpoint_history[-1]
            if pruning_checkpoint_history
            else None
        ),
        "num_pruning_checkpoints": len(pruning_checkpoint_history),
        "pruning_checkpoints_dir": str(output_dir / "pruning_checkpoints"),
        "final_pre_generation_checkpoint": {
            "grammar_path": str(output_path),
            "sampling_diagnostic": final_sample_summary,
        },
    }
    write_metadata_file(
        metadata_path,
        final_metadata_update,
        remove_keys=("pruning_checkpoint_history",),
    )
    log(f"Wrote pre-generation grammar diagnostics metadata to {metadata_path}")

    generated_data = None
    if generate_n > 0:
        log(f"Generating {generate_n} Gaussian-filtered sample sentences.")
        generated_data = generate_sentences_from_grammar(
            grammar=final_grammar,
            n_sentences=generate_n,
            output_dir=output_dir,
            seed=generation_seed,
            max_depth=generation_max_depth,
            max_attempts_per_sentence=generation_max_attempts_per_sentence,
            max_total_attempts=generation_max_total_attempts,
            generation_mode="gaussian",
            mean_len=mean_len,
            stdev_len=stdev_len,
            unique_sentences=True,
            progress_every_sentences=generation_progress_every_sentences,
        )

    final_metadata_update["generation_was_run"] = generated_data is not None
    write_metadata_file(
        metadata_path,
        final_metadata_update,
        remove_keys=("pruning_checkpoint_history",),
    )
    log(f"Wrote grammar diagnostics metadata to {metadata_path}")

    log(f"Done. Wrote {output_path}")
    log(f"Processed trees: {processed_trees}")
    log(f"Accepted trees: {accepted_trees}")
    log(f"Final structural rules: {len(final_rules)}")
    log(f"Final productions: {len(final_grammar.productions())}")


def run_final_sampling_only(
    *,
    output_dir: Path,
    lookahead_k: int,
    root_symbol: str,
    generate_n: int,
    generation_seed: int,
    generation_max_depth: int,
    generation_max_attempts_per_sentence: int,
    generation_max_total_attempts: int,
    mean_len: float,
    stdev_len: float,
    checkpoint_sample_sentences: int,
    checkpoint_sample_max_attempts_per_sentence: int,
    checkpoint_sample_max_total_attempts: Optional[int],
    generation_progress_every_sentences: int,
) -> None:
    grammar_path = output_dir / "grammar.txt"
    if not grammar_path.exists():
        raise FileNotFoundError(
            f"final_sampling_only expected an existing checkpoint grammar at {grammar_path}."
        )

    log(f"Loading existing checkpoint grammar from {grammar_path}.")
    final_grammar = load_grammar(grammar_path, start_symbol_name=root_symbol)
    log(f"Using start symbol {final_grammar.start()} for final sampling-only generation.")
    final_rules = set(structural_rule_weights_from_grammar(final_grammar))
    final_lexicon = lexicon_from_grammar(final_grammar)
    if not final_rules:
        raise RuntimeError(f"Checkpoint grammar {grammar_path} contains no structural rules.")
    if not final_lexicon:
        raise RuntimeError(f"Checkpoint grammar {grammar_path} contains no lexical rules.")

    validate_disjoint_lexicon(final_lexicon)
    final_schema = build_schema(final_rules, set(final_lexicon))
    log(f"Running dummy-schema LL({lookahead_k}) validation only.")
    final_conflicts = llk_conflict_map(final_schema, root_symbol, lookahead_k)
    if has_conflicts(final_conflicts):
        raise RuntimeError(
            f"Checkpoint dummy schema is not LL({lookahead_k}); "
            f"found {count_conflicts(final_conflicts)} conflicts."
        )

    final_sample_summary = sample_sentence_length_summary(
        final_grammar,
        requested_samples=checkpoint_sample_sentences,
        seed=generation_seed,
        max_depth=generation_max_depth,
        max_attempts_per_sentence=checkpoint_sample_max_attempts_per_sentence,
        max_total_attempts=(
            checkpoint_sample_max_total_attempts
            if checkpoint_sample_max_total_attempts is not None
            else checkpoint_sample_sentences * checkpoint_sample_max_attempts_per_sentence
        ),
        generation_mode="gaussian",
        mean_len=mean_len,
        stdev_len=stdev_len,
        unique_sentences=True,
    )
    log_sentence_length_summary("Final sampling-only grammar", final_sample_summary)

    metadata_path = output_dir / "generation_metadata.json"
    final_metadata_update = {
        "build_summary": {
            "final_sampling_only": True,
            "resumed_grammar_path": str(grammar_path),
            "lookahead_k": lookahead_k,
            "root_symbol": root_symbol,
            "checkpoint_sample_sentences": checkpoint_sample_sentences,
            "checkpoint_sample_max_attempts_per_sentence": checkpoint_sample_max_attempts_per_sentence,
            "checkpoint_sample_max_total_attempts": checkpoint_sample_max_total_attempts,
            "checkpoint_sampling_mode": "gaussian",
            "checkpoint_unique_sentences": True,
            "generation_progress_every_sentences": generation_progress_every_sentences,
            "lexicalized_llk_validation_skipped": True,
        },
        "grammar_diagnostics": build_llk_superset_metadata(
            final_schema,
            final_grammar,
            final_rules,
            final_lexicon,
            root_symbol=root_symbol,
            lookahead_k=lookahead_k,
            include_lexicalized_conflicts=False,
        ),
        "metadata_path": str(metadata_path),
        "generation_was_run": False,
        "final_pre_generation_checkpoint": {
            "grammar_path": str(grammar_path),
            "sampling_diagnostic": final_sample_summary,
            "final_sampling_only": True,
        },
    }
    write_metadata_file(
        metadata_path,
        final_metadata_update,
        remove_keys=("pruning_checkpoint_history",),
    )
    log(f"Wrote final sampling-only diagnostics metadata to {metadata_path}")

    generated_data = None
    if generate_n > 0:
        log(f"Generating {generate_n} Gaussian-filtered sample sentences from existing grammar.")
        generated_data = generate_sentences_from_grammar(
            grammar=final_grammar,
            n_sentences=generate_n,
            output_dir=output_dir,
            seed=generation_seed,
            max_depth=generation_max_depth,
            max_attempts_per_sentence=generation_max_attempts_per_sentence,
            max_total_attempts=generation_max_total_attempts,
            generation_mode="gaussian",
            mean_len=mean_len,
            stdev_len=stdev_len,
            unique_sentences=True,
            progress_every_sentences=generation_progress_every_sentences,
        )

    final_metadata_update["generation_was_run"] = generated_data is not None
    write_metadata_file(
        metadata_path,
        final_metadata_update,
        remove_keys=("pruning_checkpoint_history",),
    )
    log(f"Done. Final sampling-only mode used existing grammar {grammar_path}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an LL(k) source-subgrammar from trees using top-down prefix admission."
    )


    parser.add_argument("--input_grammar", type=Path, default=None, help="Path to the source grammar file used for lexical probabilities")
    parser.add_argument("--input_trees", type=Path, default=None, help="Path to data_trees.txt")
    parser.add_argument("--output_dir", type=Path, default=None, help="Directory for the generated grammar")
    parser.add_argument(
        "--final_sampling_only",
        "-final_sampling_only",
        action="store_true",
        help=(
            "Resume from output_dir/grammar.txt, skip tree admission and lexicalized LL(k) validation, "
            "validate only the dummy schema, and run final diagnostics/generation without writing pruning checkpoints."
        ),
    )

    parser.add_argument("--lookahead_k", type=int, default=1)
    parser.add_argument("--num_trees", type=int, default=10000)
    parser.add_argument("--tree_sample_seed", type=int, default=1)
    parser.add_argument(
        "--tree_sample_mode",
        choices=("random", "original"),
        default="random",
        help=(
            "Tree line selection mode: 'random' keeps reservoir sampling; "
            "'original' reads the first num_trees non-empty lines in file order."
        ),
    )
    parser.add_argument("--max_rules_per_lhs", type=int, default=10000000)
    parser.add_argument("--root_symbol", type=str, default=ROOT_SYMBOL)
    parser.add_argument(
        "--init_grammar_path",
        type=Path,
        default=None,
        help=(
            "Optional grammar.txt from a smaller-k run. Its structural and lexical rules "
            "are kept as a fixed backbone while building the larger-k grammar."
        ),
    )
    parser.add_argument("--generate_n", type=int, default=100000)
    parser.add_argument("--generation_seed", type=int, default=1)
    parser.add_argument("--generation_max_depth", type=int, default=200)
    parser.add_argument("--generation_max_attempts_per_sentence", type=int, default=2000)
    parser.add_argument("--generation_max_total_attempts", type=int, default=100000000)
    parser.add_argument("--mean_len", type=float, default=20.25)
    parser.add_argument("--stdev_len", type=float, default=13.11)
    parser.add_argument(
        "--checkpoint_sample_sentences",
        type=int,
        default=1000,
        help="Number of sentences sampled for pruning/final length diagnostics.",
    )
    parser.add_argument(
        "--checkpoint_sample_max_attempts_per_sentence",
        type=int,
        default=200,
        help="Per-sentence derivation attempt limit for checkpoint length diagnostics.",
    )
    parser.add_argument(
        "--checkpoint_sample_max_total_attempts",
        type=int,
        default=None,
        help=(
            "Total derivation attempt limit for checkpoint length diagnostics. "
            "Defaults to checkpoint_sample_sentences * checkpoint_sample_max_attempts_per_sentence."
        ),
    )
    parser.add_argument(
        "--prune_every_accepted_trees",
        type=int,
        default=1000,
        help="Run pruning and checkpointing after this many accepted trees.",
    )
    parser.add_argument(
        "--progress_log_every_trees",
        type=int,
        default=100,
        help="Log tree-processing progress every N processed trees; use 0 to disable.",
    )
    parser.add_argument(
        "--generation_progress_every_sentences",
        type=int,
        default=10000,
        help="Log generation progress every N generated sentences; use 0 to disable.",
    )
    parser.add_argument(
        "--protected_count_floor",
        type=float,
        default=1e-12,
        help="Minimum initial count assigned to protected backbone rules.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.final_sampling_only:
        run_final_sampling_only(
            output_dir=args.output_dir,
            lookahead_k=args.lookahead_k,
            root_symbol=args.root_symbol,
            generate_n=args.generate_n,
            generation_seed=args.generation_seed,
            generation_max_depth=args.generation_max_depth,
            generation_max_attempts_per_sentence=args.generation_max_attempts_per_sentence,
            generation_max_total_attempts=args.generation_max_total_attempts,
            mean_len=args.mean_len,
            stdev_len=args.stdev_len,
            checkpoint_sample_sentences=args.checkpoint_sample_sentences,
            checkpoint_sample_max_attempts_per_sentence=args.checkpoint_sample_max_attempts_per_sentence,
            checkpoint_sample_max_total_attempts=args.checkpoint_sample_max_total_attempts,
            generation_progress_every_sentences=args.generation_progress_every_sentences,
        )
        return

    build_llk_atomic_tree_grammar(
        grammar_path=args.input_grammar,
        trees_path=args.input_trees,
        output_dir=args.output_dir,
        lookahead_k=args.lookahead_k,
        num_trees=args.num_trees,
        tree_sample_seed=args.tree_sample_seed,
        tree_sample_mode=args.tree_sample_mode,
        max_rules_per_lhs=args.max_rules_per_lhs,
        root_symbol=args.root_symbol,
        init_grammar_path=args.init_grammar_path,
        generate_n=args.generate_n,
        generation_seed=args.generation_seed,
        generation_max_depth=args.generation_max_depth,
        generation_max_attempts_per_sentence=args.generation_max_attempts_per_sentence,
        generation_max_total_attempts=args.generation_max_total_attempts,
        mean_len=args.mean_len,
        stdev_len=args.stdev_len,
        checkpoint_sample_sentences=args.checkpoint_sample_sentences,
        checkpoint_sample_max_attempts_per_sentence=args.checkpoint_sample_max_attempts_per_sentence,
        checkpoint_sample_max_total_attempts=args.checkpoint_sample_max_total_attempts,
        prune_every_accepted_trees=args.prune_every_accepted_trees,
        progress_log_every_trees=args.progress_log_every_trees,
        generation_progress_every_sentences=args.generation_progress_every_sentences,
        protected_count_floor=args.protected_count_floor,
    )


if __name__ == "__main__":
    main()
