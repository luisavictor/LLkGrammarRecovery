"""LL(k) grammar utilities."""

from collections import Counter, defaultdict
from typing import Optional
from nltk import Nonterminal, PCFG, ProbabilisticProduction, Tree
import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple


EPS = "<eps>"
END = "$"
Rule = Tuple[str, Tuple[str, ...]]
EPSILON_LOOKAHEAD: Tuple[int, ...] = ()
Lookahead = Tuple[int, ...]


class LookaheadCodec:
    def __init__(self, grammar: Dict[str, List[Tuple[str, ...]]]) -> None:
        self.token_to_id: Dict[str, int] = {}
        self.id_to_token: Dict[int, str] = {}

        terminals = sorted(
            {
                sym
                for rhss in grammar.values()
                for rhs in rhss
                for sym in rhs
                if sym not in grammar
            }
            | {END}
        )

        for token in terminals:
            self.encode(token)

    def encode(self, token: str) -> int:
        token_id = self.token_to_id.get(token)
        if token_id is None:
            token_id = len(self.token_to_id)
            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token
        return token_id

    def decode_lookahead(self, lookahead: Lookahead) -> List[str]:
        return [self.id_to_token[token_id] for token_id in lookahead]



def is_nonterminal_symbol(sym: str, grammar: Dict[str, List[Tuple[str, ...]]]) -> bool:
    return sym in grammar


def trunc_k(seq: Lookahead, k: int) -> Lookahead:
    return seq[:k]


def index_productions(grammar: PCFG) -> Dict[Nonterminal, List[Tuple[Tuple[object, ...], float]]]:
    table: Dict[Nonterminal, List[Tuple[Tuple[object, ...], float]]] = defaultdict(list)
    for prod in grammar.productions():
        table[prod.lhs()].append((prod.rhs(), prod.prob()))
    return table


def terminal_rules_from_grammar(grammar: PCFG) -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    for prod in grammar.productions():
        if len(prod.rhs()) == 1 and isinstance(prod.rhs()[0], str):
            out[(str(prod.lhs()), prod.rhs()[0])] = float(prod.prob())
    return out

def concat_k(
    left_set: Set[Lookahead],
    right_set: Set[Lookahead],
    k: int,
) -> Set[Lookahead]:
    result: Set[Lookahead] = set()

    for left in left_set:
        if len(left) >= k:
            result.add(left[:k])
            continue

        for right in right_set:
            result.add(trunc_k(left + right, k))

    return result




def prune_nonproductive_structural_backbone(
    structural_rules: Set[Tuple[str, Tuple[str, ...]]],
    productive_preterminal_labels: Set[str],
) -> Tuple[Set[Tuple[str, Tuple[str, ...]]], Set[str]]:
    productive_symbols: Set[str] = set(productive_preterminal_labels)
    by_lhs: Dict[str, List[Tuple[str, ...]]] = defaultdict(list)
    for lhs, rhs in structural_rules:
        by_lhs[lhs].append(rhs)

    changed = True
    while changed:
        changed = False
        for lhs, rhss in by_lhs.items():
            if lhs in productive_symbols:
                continue
            for rhs in rhss:
                if all(sym in productive_symbols for sym in rhs):
                    productive_symbols.add(lhs)
                    changed = True
                    break

    pruned_rules = {
        (lhs, rhs)
        for lhs, rhs in structural_rules
        if lhs in productive_symbols and all(sym in productive_symbols for sym in rhs)
    }
    return pruned_rules, productive_symbols


def prune_unreachable_and_renormalize(grammar: PCFG) -> PCFG:
    reachable: Set[Nonterminal] = {grammar.start()}
    queue: List[Nonterminal] = [grammar.start()]
    by_lhs: Dict[Nonterminal, List[ProbabilisticProduction]] = defaultdict(list)
    for prod in grammar.productions():
        by_lhs[prod.lhs()].append(prod)

    while queue:
        lhs = queue.pop()
        for prod in by_lhs.get(lhs, []):
            for sym in prod.rhs():
                if isinstance(sym, Nonterminal) and sym not in reachable:
                    reachable.add(sym)
                    queue.append(sym)

    kept = [prod for prod in grammar.productions() if prod.lhs() in reachable]
    grouped: Dict[Nonterminal, List[ProbabilisticProduction]] = defaultdict(list)
    for prod in kept:
        grouped[prod.lhs()].append(prod)

    renormalized: List[ProbabilisticProduction] = []
    for lhs, prods in grouped.items():
        total_prob = sum(prod.prob() for prod in prods)
        if total_prob <= 0:
            continue
        for prod in prods:
            renormalized.append(
                ProbabilisticProduction(lhs, prod.rhs(), prob=prod.prob() / total_prob)
            )

    return PCFG(grammar.start(), renormalized)



def prune_unreachable_structural_backbone(
    structural_rules: Set[Tuple[str, Tuple[str, ...]]],
    preterminal_labels: Set[str],
    start_symbol: str,
) -> Tuple[Set[Tuple[str, Tuple[str, ...]]], Set[str], Set[str]]:
    reachable_nonterminals: Set[str] = {start_symbol}
    reachable_preterminals: Set[str] = set()
    queue: List[str] = [start_symbol]
    by_lhs: Dict[str, List[Tuple[str, ...]]] = defaultdict(list)

    for lhs, rhs in structural_rules:
        by_lhs[lhs].append(rhs)

    while queue:
        lhs = queue.pop()
        for rhs in by_lhs.get(lhs, []):
            for sym in rhs:
                if sym in preterminal_labels:
                    reachable_preterminals.add(sym)
                elif sym in by_lhs and sym not in reachable_nonterminals:
                    reachable_nonterminals.add(sym)
                    queue.append(sym)

    pruned_rules = {rule for rule in structural_rules if rule[0] in reachable_nonterminals}
    return pruned_rules, reachable_nonterminals, reachable_preterminals



def compute_nullable(grammar: Dict[str, List[Tuple[str, ...]]]) -> Set[str]:
    nullable: Set[str] = set()

    changed = True
    while changed:
        changed = False

        for lhs, rhss in grammar.items():
            if lhs in nullable:
                continue

            for rhs in rhss:
                if not rhs:
                    nullable.add(lhs)
                    changed = True
                    break

                if all(
                    is_nonterminal_symbol(sym, grammar) and sym in nullable
                    for sym in rhs
                ):
                    nullable.add(lhs)
                    changed = True
                    break

    return nullable


def first_seq(
    seq: Sequence[str],
    first_k: Dict[str, Set[Lookahead]],
    nullable: Set[str],
    grammar: Dict[str, List[Tuple[str, ...]]],
    codec: LookaheadCodec,
    k: int,
    cache: Optional[Dict[Tuple[str, ...], Set[Lookahead]]] = None,
) -> Set[Lookahead]:
    key = tuple(seq)

    if cache is not None and key in cache:
        return cache[key]

    result: Set[Lookahead] = {EPSILON_LOOKAHEAD}

    for symbol in seq:
        symbol_first = (
            first_k[symbol]
            if is_nonterminal_symbol(symbol, grammar)
            else {(codec.encode(symbol),)}
        )

        new_result: Set[Lookahead] = set()

        for prefix in result:
            if len(prefix) >= k:
                new_result.add(prefix[:k])
                continue

            for suffix in symbol_first:
                new_result.add(trunc_k(prefix + suffix, k))

        result = new_result

        # Important for real LL(k):
        # do not stop after the first non-nullable symbol;
        # stop only once every prefix already has length k.
        if all(len(prefix) >= k for prefix in result):
            break

    if cache is not None:
        cache[key] = result

    return result


def compute_first_k_sets(
    grammar: Dict[str, List[Tuple[str, ...]]],
    nullable: Set[str],
    codec: LookaheadCodec,
    k: int,
) -> Dict[str, Set[Lookahead]]:
    first_k: Dict[str, Set[Lookahead]] = {nt: set() for nt in grammar}

    changed = True
    while changed:
        changed = False
        seq_cache: Dict[Tuple[str, ...], Set[Lookahead]] = {}

        for lhs, rhss in grammar.items():
            before = set(first_k[lhs])

            for rhs in rhss:
                first_k[lhs].update(
                    first_seq(
                        rhs,
                        first_k,
                        nullable,
                        grammar,
                        codec,
                        k,
                        cache=seq_cache,
                    )
                )

            if first_k[lhs] != before:
                changed = True

    return first_k


def compute_follow_k_sets(
    grammar: Dict[str, List[Tuple[str, ...]]],
    start_symbol: str,
    first_k: Dict[str, Set[Lookahead]],
    nullable: Set[str],
    codec: LookaheadCodec,
    k: int,
) -> Dict[str, Set[Lookahead]]:
    follow_k: Dict[str, Set[Lookahead]] = {nt: set() for nt in grammar}
    follow_k[start_symbol].add((codec.encode(END),))

    changed = True
    while changed:
        changed = False
        seq_cache: Dict[Tuple[str, ...], Set[Lookahead]] = {}

        for lhs, rhss in grammar.items():
            for rhs in rhss:
                for i, sym in enumerate(rhs):
                    if not is_nonterminal_symbol(sym, grammar):
                        continue

                    beta_first = first_seq(
                        rhs[i + 1 :],
                        first_k,
                        nullable,
                        grammar,
                        codec,
                        k,
                        cache=seq_cache,
                    )

                    new_strings = concat_k(beta_first, follow_k[lhs], k)

                    before = len(follow_k[sym])
                    follow_k[sym].update(new_strings)

                    if len(follow_k[sym]) != before:
                        changed = True

    return follow_k


def compute_rule_lookaheads_k(
    grammar: Dict[str, List[Tuple[str, ...]]],
    start_symbol: str,
    k: int,
) -> Tuple[
    Set[str],
    Dict[str, Set[Lookahead]],
    Dict[str, Set[Lookahead]],
    Dict[Rule, Set[Lookahead]],
    LookaheadCodec,
]:
    if k < 1:
        raise ValueError("k must be >= 1")
    if start_symbol not in grammar:
        raise ValueError(f"Start symbol {start_symbol!r} is not present in the grammar.")
    if not grammar[start_symbol]:
        raise ValueError(f"Start symbol {start_symbol!r} has no productions.")
    for lhs, rhss in grammar.items():
        if not lhs:
            raise ValueError("Grammar contains an empty nonterminal name.")
        for rhs in rhss:
            for symbol in rhs:
                if not symbol:
                    raise ValueError(f"Grammar rule {lhs!r} -> {rhs!r} contains an empty symbol.")

    codec = LookaheadCodec(grammar)
    nullable = compute_nullable(grammar)
    first_k = compute_first_k_sets(grammar, nullable, codec, k)
    follow_k = compute_follow_k_sets(
        grammar,
        start_symbol,
        first_k,
        nullable,
        codec,
        k,
    )

    rule_lookaheads: Dict[Rule, Set[Lookahead]] = {}
    seq_cache: Dict[Tuple[str, ...], Set[Lookahead]] = {}

    for lhs, rhss in grammar.items():
        for rhs in rhss:
            first_alpha = first_seq(
                rhs,
                first_k,
                nullable,
                grammar,
                codec,
                k,
                cache=seq_cache,
            )

            rule_lookaheads[(lhs, rhs)] = concat_k(
                first_alpha,
                follow_k[lhs],
                k,
            )

    return nullable, first_k, follow_k, rule_lookaheads, codec


def compute_internal_conflict_map(
    grammar: Dict[str, List[Tuple[str, ...]]],
    rule_lookaheads: Dict[Rule, Set[Lookahead]],
) -> Dict[str, List[Dict[str, object]]]:
    conflicts: Dict[str, List[Dict[str, object]]] = {}

    for lhs, rhss in grammar.items():
        entries: List[Dict[str, object]] = []

        for i in range(len(rhss)):
            rhs_i = rhss[i]
            lookahead_i = rule_lookaheads[(lhs, rhs_i)]

            for j in range(i + 1, len(rhss)):
                rhs_j = rhss[j]
                overlap = lookahead_i & rule_lookaheads[(lhs, rhs_j)]

                if overlap:
                    entries.append(
                        {
                            "alt1_index": i,
                            "alt2_index": j,
                            "witnesses": overlap,
                        }
                    )

        if entries:
            conflicts[lhs] = entries

    return conflicts



def serialize_conflict_map(
    grammar: Dict[str, List[Tuple[str, ...]]],
    conflict_map: Dict[str, List[Dict[str, object]]],
    codec: LookaheadCodec,
) -> Dict[str, List[Dict[str, object]]]:
    serialized: Dict[str, List[Dict[str, object]]] = {}

    for lhs, entries in conflict_map.items():
        rhss = grammar.get(lhs, [])
        serialized_entries: List[Dict[str, object]] = []

        for entry in entries:
            rhs1 = rhss[entry["alt1_index"]]
            rhs2 = rhss[entry["alt2_index"]]

            serialized_entries.append(
                {
                    "alt1_index": entry["alt1_index"],
                    "alt2_index": entry["alt2_index"],
                    "alt1_rhs": list(rhs1),
                    "alt2_rhs": list(rhs2),
                    "witnesses": [
                        codec.decode_lookahead(witness)
                        for witness in sorted(entry["witnesses"])
                    ],
                }
            )

        serialized[lhs] = serialized_entries

    return serialized


def llk_conflict_map(
    grammar: Dict[str, List[Tuple[str, ...]]],
    start_symbol: str,
    k: int,
) -> Dict[str, List[Dict[str, object]]]:
    if k < 1:
        raise ValueError("k must be >= 1")

    _nullable, _first_k, _follow_k, rule_lookaheads, codec = (
        compute_rule_lookaheads_k(grammar, start_symbol, k)
    )

    internal_conflicts = compute_internal_conflict_map(
        grammar,
        rule_lookaheads,
    )

    return serialize_conflict_map(grammar, internal_conflicts, codec)


def has_conflicts(conflict_map: Dict[str, List[Dict[str, object]]]) -> bool:
    return any(conflict_map.values())


def count_conflicts(conflict_map: Dict[str, List[Dict[str, object]]]) -> int:
    return sum(len(entries) for entries in conflict_map.values())


def grammar_dict_from_pcfg(grammar: PCFG) -> Dict[str, List[Tuple[str, ...]]]:
    out: Dict[str, List[Tuple[str, ...]]] = defaultdict(list)
    for prod in grammar.productions():
        lhs = str(prod.lhs())
        rhs = tuple(str(sym) for sym in prod.rhs())
        out[lhs].append(rhs)
    return dict(out)


def tree_structural_productions(tree: Tree) -> List[Tuple[str, Tuple[str, ...]]]:
    productions: List[Tuple[str, Tuple[str, ...]]] = []
    for prod in tree.productions():
        if any(isinstance(rhs, Nonterminal) for rhs in prod.rhs()):
            lhs = str(prod.lhs())
            rhs = tuple(str(sym) for sym in prod.rhs())
            productions.append((lhs, rhs))
    return productions


def tree_terminal_productions(tree: Tree) -> List[Tuple[str, str]]:
    productions: List[Tuple[str, str]] = []
    for prod in tree.productions():
        if len(prod.rhs()) == 1 and isinstance(prod.rhs()[0], str):
            productions.append((str(prod.lhs()), prod.rhs()[0]))
    return productions


def structural_rule_yield_lengths(tree: Tree) -> List[Tuple[Tuple[str, Tuple[str, ...]], int]]:
    out: List[Tuple[Tuple[str, Tuple[str, ...]], int]] = []
    for subtree in tree.subtrees():
        if not isinstance(subtree, Tree):
            continue
        if len(subtree) == 0:
            continue
        if any(isinstance(child, Tree) for child in subtree):
            lhs = str(subtree.label())
            rhs = tuple(str(child.label()) if isinstance(child, Tree) else child for child in subtree)
            out.append(((lhs, rhs), len(subtree.leaves())))
    return out


def parse_earleyx_rhs(raw_rhs: str) -> Tuple[object, ...]:
    items = [tok for tok in raw_rhs.strip().split() if tok]
    rhs: List[object] = []
    for tok in items:
        if tok.startswith("_"):
            rhs.append(tok[1:])
        else:
            rhs.append(Nonterminal(tok))
    return tuple(rhs)


def save_grammar_earleyx(grammar: PCFG, path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for prod in grammar.productions():
            rhs_items: List[str] = []
            for sym in prod.rhs():
                rhs_items.append(f"_{sym}" if isinstance(sym, str) else str(sym))
            f.write(f"{prod.lhs()}->[{ ' '.join(rhs_items) }] : {prod.prob():.8f}\n")


def save_sentence_tree_pairs(sentences: Sequence[str], trees: Sequence[Tree], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for sentence, tree in zip(sentences, trees):
            row = {
                "sentence": sentence,
                "tree": tree.pformat(margin=10**9),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def count_productions_from_trees(trees: Sequence[Tree]) -> Counter:
    counts: Counter = Counter()
    for tree in trees:
        for prod in tree.productions():
            counts[prod] += 1
    return counts


def mle_estimate_pcfg(start_symbol: Nonterminal, trees: Sequence[Tree]) -> PCFG:
    counts = count_productions_from_trees(trees)
    lhs_groups: Dict[Nonterminal, List[Tuple[Tuple[object, ...], int]]] = defaultdict(list)
    for prod, count in counts.items():
        lhs_groups[prod.lhs()].append((prod.rhs(), count))

    productions: List[ProbabilisticProduction] = []
    for lhs, rules in lhs_groups.items():
        total = sum(count for _rhs, count in rules)
        for rhs, count in rules:
            productions.append(ProbabilisticProduction(lhs, rhs, prob=count / total))

    return PCFG(start_symbol, productions)


def structural_rules_from_grammar(grammar: PCFG) -> Dict[Tuple[str, Tuple[str, ...]], float]:
    out: Dict[Tuple[str, Tuple[str, ...]], float] = {}
    for prod in grammar.productions():
        if any(isinstance(sym, Nonterminal) for sym in prod.rhs()):
            out[(str(prod.lhs()), tuple(str(sym) for sym in prod.rhs()))] = float(prod.prob())
    return out



def terminal_vocabulary_from_grammar(grammar: PCFG) -> Set[str]:
    return {word for _lhs, word in terminal_rules_from_grammar(grammar)}


def preterminal_labels_from_grammar(grammar: PCFG) -> Set[str]:
    labels: Set[str] = set()
    for prod in grammar.productions():
        if len(prod.rhs()) == 1 and isinstance(prod.rhs()[0], str):
            labels.add(str(prod.lhs()))
    return labels


def jensen_shannon_divergence(p: Dict[object, float], q: Dict[object, float]) -> float:
    universe = set(p) | set(q)
    if not universe:
        return 0.0
    m = {key: 0.5 * (p.get(key, 0.0) + q.get(key, 0.0)) for key in universe}

    def kl(a: Dict[object, float], b: Dict[object, float]) -> float:
        total = 0.0
        for key in universe:
            av = a.get(key, 0.0)
            if av <= 0.0:
                continue
            bv = b.get(key, 0.0)
            total += av * math.log(av / bv, 2)
        return total

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def global_rule_distribution(
    rule_support: Counter,
) -> Dict[Tuple[str, Tuple[str, ...]], float]:
    total = float(sum(rule_support.values()))
    if total <= 0.0:
        return {}
    return {rule: count / total for rule, count in rule_support.items()}


def lexical_preterminal_shares(
    grammar: PCFG,
    terminal_support: Counter,
    allowed_preterminals: Set[str],
) -> Dict[str, float]:
    terminal_probs = terminal_rules_from_grammar(grammar)
    surviving_preterminals = sorted(allowed_preterminals)
    if not surviving_preterminals:
        return {}

    weighted_mass: Dict[str, float] = {}
    for lhs in surviving_preterminals:
        support_mass = float(sum(terminal_support[(lhs, word)] for (pt, word) in terminal_probs if pt == lhs))
        prob_mass = float(sum(prob for (pt, _word), prob in terminal_probs.items() if pt == lhs))
        weighted_mass[lhs] = support_mass + prob_mass

    total_mass = sum(weighted_mass.values())
    if total_mass <= 0.0:
        uniform_share = 1.0 / len(surviving_preterminals)
        return {lhs: uniform_share for lhs in surviving_preterminals}
    return {lhs: weighted_mass[lhs] / total_mass for lhs in surviving_preterminals}


def extend_lexicon_to_cover_terminal_vocabulary(
    grammar: PCFG,
    lexical_assignment: Dict[str, Dict[str, float]],
    terminal_support: Counter,
    allowed_preterminals: Set[str],
) -> Tuple[Dict[str, Dict[str, float]], List[str]]:
    all_words = terminal_vocabulary_from_grammar(grammar)
    assigned_words = {word for word_probs in lexical_assignment.values() for word in word_probs}
    missing_words = sorted(all_words - assigned_words)
    if not missing_words:
        return lexical_assignment, []

    surviving_preterminals = sorted(allowed_preterminals)
    if not surviving_preterminals:
        return lexical_assignment, missing_words

    extended: Dict[str, Dict[str, float]] = {
        lhs: dict(word_probs) for lhs, word_probs in lexical_assignment.items()
    }
    for lhs in surviving_preterminals:
        extended.setdefault(lhs, {})

    current_counts = {lhs: len(extended[lhs]) for lhs in surviving_preterminals}
    base_shares = lexical_preterminal_shares(grammar, terminal_support, set(surviving_preterminals))
    final_total = len(all_words)
    target_counts = {
        lhs: base_shares[lhs] * final_total for lhs in surviving_preterminals
    }

    positive_probs = [
        prob
        for word_probs in extended.values()
        for prob in word_probs.values()
        if prob > 0.0
    ]
    global_fallback_prob = min(positive_probs) * 0.5 if positive_probs else 1e-12

    for word in missing_words:
        target_lhs = max(
            surviving_preterminals,
            key=lambda lhs: (target_counts[lhs] - current_counts[lhs], -current_counts[lhs], lhs),
        )
        existing_probs = [prob for prob in extended[target_lhs].values() if prob > 0.0]
        fallback_prob = min(existing_probs) * 0.5 if existing_probs else global_fallback_prob
        extended[target_lhs][word] = fallback_prob
        current_counts[target_lhs] += 1

    extended_words = {word for word_probs in extended.values() for word in word_probs}
    still_missing_words = sorted(all_words - extended_words)
    if still_missing_words:
        raise RuntimeError(
            "Failed to preserve the full source terminal vocabulary after lexical reassignment."
        )

    return extended, missing_words


def build_final_pcfg(
    source_grammar: PCFG,
    selected_structural_rules: Set[Tuple[str, Tuple[str, ...]]],
    lexical_assignment: Dict[str, Dict[str, float]],
    start_symbol: str,
) -> PCFG:
    grammar_structural_probs = structural_rules_from_grammar(source_grammar)

    grouped_structural: Dict[str, List[Tuple[Tuple[str, ...], float]]] = defaultdict(list)
    for lhs, rhs in selected_structural_rules:
        grouped_structural[lhs].append((rhs, grammar_structural_probs[(lhs, rhs)]))

    productions: List[ProbabilisticProduction] = []
    for lhs, rhs_probs in grouped_structural.items():
        total = sum(prob for _, prob in rhs_probs)
        if total <= 0:
            continue
        for rhs, prob in rhs_probs:
            productions.append(
                ProbabilisticProduction(
                    Nonterminal(lhs),
                    tuple(Nonterminal(sym) for sym in rhs),
                    prob=prob / total,
                )
            )

    for lhs, word_probs in lexical_assignment.items():
        total = sum(word_probs.values())
        if total <= 0:
            continue
        for word, prob in sorted(word_probs.items()):
            productions.append(
                ProbabilisticProduction(Nonterminal(lhs), (word,), prob=prob / total)
            )

    return PCFG(Nonterminal(start_symbol), productions)
