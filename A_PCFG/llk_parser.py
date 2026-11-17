"""LL(k)-aware grammar loading, diagnostics, and predictive parsing."""

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from nltk import Nonterminal, PCFG, ProbabilisticProduction

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from A_PCFG.LLk_core import (
    END,
    Lookahead,
    Rule,
    compute_internal_conflict_map,
    compute_rule_lookaheads_k,
    grammar_dict_from_pcfg,
)
ROOT_SYMBOL = "ROOT"
GRAMMAR_LINE_RE = re.compile(r"^\s*(?P<lhs>\S+)\s*->\s*\[(?P<rhs>.*?)\]\s*:\s*(?P<prob>[0-9eE+\-.]+)\s*$")



@dataclass
class ParseNode:
    symbol: str
    is_terminal: bool
    children: Optional[List["ParseNode"]] = None
    matched: bool = False

    def render(self) -> str:
        if self.is_terminal:
            return self.symbol if self.matched else f"?{self.symbol}"
        if self.children is None:
            return f"({self.symbol} ?)"
        if not self.children:
            return f"({self.symbol})"
        return f"({self.symbol} {' '.join(child.render() for child in self.children)})"


@dataclass
class ParserState:
    stack_symbol: str
    node: Optional[ParseNode]



def parse_earleyx_rhs(raw_rhs: str) -> Tuple[object, ...]:
    items = [tok for tok in raw_rhs.strip().split() if tok]
    rhs: List[object] = []
    for tok in items:
        if tok.startswith("_"):
            terminal = tok[1:]
            if terminal == "":
                raise ValueError("Grammar contains an empty lexical terminal.")
            rhs.append(terminal)
        else:
            rhs.append(Nonterminal(tok))
    return tuple(rhs)


def load_grammar(path: Path, start_symbol_name: Optional[str] = None) -> PCFG:
    productions: List[ProbabilisticProduction] = []
    seen_lhs: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            match = GRAMMAR_LINE_RE.match(line)
            if not match:
                raise ValueError(f"Cannot parse grammar line {line_no}: {line}")
            lhs_text = match.group("lhs")
            rhs = parse_earleyx_rhs(match.group("rhs"))
            prob = float(match.group("prob"))
            lhs = Nonterminal(lhs_text)
            productions.append(ProbabilisticProduction(lhs, rhs, prob=prob))
            seen_lhs.append(lhs_text)
    if not productions:
        raise ValueError(f"No productions loaded from {path}.")
    start_symbol = Nonterminal(start_symbol_name if start_symbol_name is not None else seen_lhs[0])
    if start_symbol_name is not None and start_symbol_name not in seen_lhs:
        raise ValueError(f"Start symbol {start_symbol_name!r} was not found as an LHS in {path}.")
    return PCFG(start_symbol, productions)



def is_nonterminal(symbol: str, grammar: Dict[str, List[Tuple[str, ...]]]) -> bool:
    return symbol in grammar


def lookahead_from_input(tokens: Sequence[str], position: int, k: int, codec) -> Lookahead:
    visible = list(tokens[position : position + k])
    while len(visible) < k:
        visible.append(END)
    return tuple(codec.encode(token) for token in visible[:k])

def decode_lookahead(lookahead: Lookahead, codec) -> List[str]:
    return codec.decode_lookahead(lookahead)


def lookahead_compatible(witness: Lookahead, observed: Lookahead, codec) -> bool:
    witness_tokens = decode_lookahead(witness, codec)
    observed_tokens = decode_lookahead(observed, codec)
    for expected, actual in zip(witness_tokens, observed_tokens):
        if expected == END:
            return True
        if expected != actual:
            return False
    return len(witness_tokens) <= len(observed_tokens)


def format_stack(states: Sequence[ParserState]) -> List[str]:
    return [state.stack_symbol for state in reversed(states)]


def serialize_rule_label(lhs: str, rhs: Sequence[str]) -> str:
    return f"{lhs}->[{ ' '.join(rhs) }]"


def extract_parser_targets(
    parse_result: Dict[str, object],
    *,
    empty_stack_label: str = "<EMPTY_STACK>",
    no_rule_label: str = "<NO_RULE>",
) -> List[Dict[str, object]]:
    token_states = list(parse_result.get("token_states", []))
    action_trace = list(parse_result.get("action_trace", []))
    expand_by_position: Dict[int, List[Dict[str, object]]] = {}
    for step in action_trace:
        if step.get("action") != "expand":
            continue
        position = int(step["position"])
        expand_by_position.setdefault(position, []).append(step)

    targets: List[Dict[str, object]] = []
    for token_state in token_states:
        token_index = int(token_state["token_index"])
        stack_after_match = list(token_state.get("stack_top_to_bottom_after_match", []))
        next_expands = expand_by_position.get(token_index + 1, [])
        next_rule = no_rule_label
        if next_expands:
            next_expand = next_expands[0]
            next_rule = serialize_rule_label(
                str(next_expand["expanded_nonterminal"]),
                [str(sym) for sym in next_expand.get("selected_rhs", [])],
            )
        targets.append(
            {
                "token_index": token_index,
                "token": str(token_state["token"]),
                "stack_depth": int(token_state["stack_depth_after_match"]),
                "top_of_stack": stack_after_match[0] if stack_after_match else empty_stack_label,
                "selected_rule": next_rule,
            }
        )
    return targets


def build_rejection_result(
    grammar_path: Path,
    start_symbol: str,
    lookahead_k: int,
    tokens: Sequence[str],
    max_stack_depth: int,
    applied_rules: List[Dict[str, object]],
    action_trace: List[Dict[str, object]],
    token_states: List[Dict[str, object]],
    root: ParseNode,
    failure_state: Dict[str, object],
) -> Dict[str, object]:
    return {
        "grammar_path": str(grammar_path),
        "start_symbol": start_symbol,
        "lookahead_k": lookahead_k,
        "input_tokens": list(tokens),
        "accepted": False,
        "max_stack_depth": max_stack_depth,
        "applied_rules": applied_rules,
        "action_trace": action_trace + [failure_state],
        "token_states": token_states,
        "final_partial_parse": root.render(),
        "rejection": failure_state,
    }


def build_predictive_table(
    grammar: Dict[str, List[Tuple[str, ...]]],
    start_symbol: str,
    k: int,
):
    nullable, first_k, follow_k, rule_lookaheads, codec = compute_rule_lookaheads_k(grammar, start_symbol, k)
    conflicts = compute_internal_conflict_map(grammar, rule_lookaheads)
    if conflicts:
        rendered_conflicts = {}
        for lhs, entries in conflicts.items():
            rendered_conflicts[lhs] = [
                {
                    "alt1_rhs": list(grammar[lhs][entry["alt1_index"]]),
                    "alt2_rhs": list(grammar[lhs][entry["alt2_index"]]),
                    "witnesses": [decode_lookahead(witness, codec) for witness in sorted(entry["witnesses"])],
                }
                for entry in entries
            ]
        raise ValueError(
            "Grammar is not LL(k); predictive parsing would be ambiguous.\n"
            + json.dumps(rendered_conflicts, indent=2)
        )

    table: Dict[str, Dict[Lookahead, Tuple[str, ...]]] = {}
    for lhs, rhss in grammar.items():
        lhs_table: Dict[Lookahead, Tuple[str, ...]] = {}
        for rhs in rhss:
            for witness in rule_lookaheads[(lhs, rhs)]:
                lhs_table[witness] = rhs
        table[lhs] = lhs_table

    return table, nullable, first_k, follow_k, rule_lookaheads, codec


def select_rhs_for_lookahead(
    nonterminal: str,
    grammar: Dict[str, List[Tuple[str, ...]]],
    rule_lookaheads: Dict[Rule, set],
    observed_lookahead: Lookahead,
    codec,
) -> Tuple[Optional[Tuple[str, ...]], List[Dict[str, object]]]:
    candidates: List[Tuple[Tuple[str, ...], int, List[List[str]]]] = []
    for rhs in grammar.get(nonterminal, []):
        compatible_witnesses = [
            decode_lookahead(witness, codec)
            for witness in sorted(rule_lookaheads[(nonterminal, rhs)])
            if lookahead_compatible(witness, observed_lookahead, codec)
        ]
        if compatible_witnesses:
            longest_prefix = max(
                sum(1 for token in witness if token != END)
                for witness in compatible_witnesses
            )
            candidates.append((rhs, longest_prefix, compatible_witnesses))

    if not candidates:
        return None, []

    best_prefix = max(prefix_len for _rhs, prefix_len, _witnesses in candidates)
    best = [
        {
            "rhs": list(rhs),
            "compatible_witnesses": compatible_witnesses,
            "matched_prefix_len": prefix_len,
        }
        for rhs, prefix_len, compatible_witnesses in candidates
        if prefix_len == best_prefix
    ]
    if len(best) != 1:
        return None, best
    return tuple(best[0]["rhs"]), best


def parse_with_state_trace(
    grammar_path: Path,
    sentence: Sequence[str],
    lookahead_k: int,
    start_symbol_override: Optional[str] = None,
    include_analysis: bool = False,
    include_rule_lookaheads: bool = False,
) -> Dict[str, object]:
    pcfg = load_grammar(grammar_path)
    grammar = grammar_dict_from_pcfg(pcfg)
    start_symbol = start_symbol_override or (ROOT_SYMBOL if ROOT_SYMBOL in grammar else str(pcfg.start()))
    if start_symbol not in grammar:
        raise ValueError(
            f"Requested start symbol '{start_symbol}' is not present in grammar. "
            f"Available nonterminals include: {', '.join(sorted(grammar)[:25])}"
        )
    table, nullable, first_k, follow_k, rule_lookaheads, codec = build_predictive_table(
        grammar,
        start_symbol,
        lookahead_k,
    )

    root = ParseNode(start_symbol, is_terminal=False)
    stack: List[ParserState] = [
        ParserState(END, None),
        ParserState(start_symbol, root),
    ]
    tokens = list(sentence)
    position = 0
    step_index = 0
    max_stack_depth = len(stack)
    action_trace: List[Dict[str, object]] = []
    token_states: List[Dict[str, object]] = []
    applied_rules: List[Dict[str, object]] = []

    def snapshot(action: str, details: Dict[str, object]) -> Dict[str, object]:
        nonlocal step_index, max_stack_depth
        max_stack_depth = max(max_stack_depth, len(stack))
        current_lookahead = lookahead_from_input(tokens, position, lookahead_k, codec)
        state = {
            "step": step_index,
            "action": action,
            "position": position,
            "consumed_tokens": tokens[:position],
            "remaining_tokens": tokens[position:],
            "lookahead": decode_lookahead(current_lookahead, codec),
            "stack_top_to_bottom": format_stack(stack),
            "stack_depth": len(stack),
            "partial_parse": root.render(),
        }
        state.update(details)
        step_index += 1
        return state

    while stack:
        top = stack.pop()
        symbol = top.stack_symbol

        if symbol == END:
            if position != len(tokens):
                failure_state = snapshot(
                    "error",
                    {
                        "error": "Reached end-of-stack before consuming all input.",
                        "unexpected_token": tokens[position],
                    },
                )
                return build_rejection_result(
                    grammar_path,
                    start_symbol,
                    lookahead_k,
                    tokens,
                    max_stack_depth,
                    applied_rules,
                    action_trace,
                    token_states,
                    root,
                    failure_state,
                )
            action_trace.append(snapshot("accept", {"matched_end_marker": True}))
            break

        if not is_nonterminal(symbol, grammar):
            if position >= len(tokens):
                failure_state = snapshot(
                    "error",
                    {
                        "error": f"Expected terminal '{symbol}' but input ended.",
                    },
                )
                return build_rejection_result(
                    grammar_path,
                    start_symbol,
                    lookahead_k,
                    tokens,
                    max_stack_depth,
                    applied_rules,
                    action_trace,
                    token_states,
                    root,
                    failure_state,
                )
            observed = tokens[position]
            if observed != symbol:
                failure_state = snapshot(
                    "error",
                    {
                        "error": f"Terminal mismatch: expected '{symbol}', observed '{observed}'.",
                        "expected_terminal": symbol,
                        "observed_terminal": observed,
                    },
                )
                return build_rejection_result(
                    grammar_path,
                    start_symbol,
                    lookahead_k,
                    tokens,
                    max_stack_depth,
                    applied_rules,
                    action_trace,
                    token_states,
                    root,
                    failure_state,
                )
            if top.node is not None:
                top.node.matched = True
            action_trace.append(
                snapshot(
                    "match",
                    {
                        "matched_token": observed,
                        "matched_symbol": symbol,
                    },
                )
            )
            token_states.append(
                {
                    "token_index": position,
                    "token": observed,
                    "stack_depth_after_match": len(stack),
                    "stack_top_to_bottom_after_match": format_stack(stack),
                    "next_lookahead_after_match": decode_lookahead(
                        lookahead_from_input(tokens, position + 1, lookahead_k, codec),
                        codec,
                    ),
                    "partial_parse_after_match": root.render(),
                }
            )
            position += 1
            continue

        current_lookahead = lookahead_from_input(tokens, position, lookahead_k, codec)
        rhs = table.get(symbol, {}).get(current_lookahead)
        compatible_rules: List[Dict[str, object]] = []
        if rhs is None:
            rhs, compatible_rules = select_rhs_for_lookahead(
                symbol,
                grammar,
                rule_lookaheads,
                current_lookahead,
                codec,
            )
        if rhs is None:
            available = [
                decode_lookahead(lookahead, codec)
                for lookahead in sorted(table.get(symbol, {}))
            ]
            failure_state = snapshot(
                "error",
                {
                    "error": f"No predictive rule for nonterminal '{symbol}'.",
                    "nonterminal": symbol,
                    "available_lookaheads": available,
                    "compatible_rules": compatible_rules,
                },
            )
            return build_rejection_result(
                grammar_path,
                start_symbol,
                lookahead_k,
                tokens,
                max_stack_depth,
                applied_rules,
                action_trace,
                token_states,
                root,
                failure_state,
            )

        children = [
            ParseNode(child_symbol, is_terminal=not is_nonterminal(child_symbol, grammar))
            for child_symbol in rhs
        ]
        if top.node is not None:
            top.node.children = children
        for child in reversed(children):
            stack.append(ParserState(child.symbol, child))

        applied_rules.append(
            {
                "lhs": symbol,
                "rhs": list(rhs),
                "position": position,
                "lookahead": decode_lookahead(current_lookahead, codec),
            }
        )
        expand_details = {
            "expanded_nonterminal": symbol,
            "selected_rhs": list(rhs),
        }
        if include_rule_lookaheads:
            expand_details["rule_lookaheads"] = [
                decode_lookahead(lookahead, codec)
                for lookahead in sorted(rule_lookaheads[(symbol, rhs)])
            ]
        action_trace.append(snapshot("expand", expand_details))

    if position != len(tokens):
        failure_state = snapshot(
            "error",
            {
                "error": "Parsing stopped before all input was consumed.",
            },
        )
        return build_rejection_result(
            grammar_path,
            start_symbol,
            lookahead_k,
            tokens,
            max_stack_depth,
            applied_rules,
            action_trace,
            token_states,
            root,
            failure_state,
        )

    result = {
        "grammar_path": str(grammar_path),
        "start_symbol": start_symbol,
        "lookahead_k": lookahead_k,
        "input_tokens": tokens,
        "accepted": True,
        "max_stack_depth": max_stack_depth,
        "applied_rules": applied_rules,
        "action_trace": action_trace,
        "token_states": token_states,
        "parser_targets": extract_parser_targets(
            {
                "token_states": token_states,
                "action_trace": action_trace,
            }
        ),
        "final_partial_parse": root.render(),
    }
    if include_analysis:
        result["analysis"] = {
            "nullable": sorted(nullable),
            "first_k": {
                symbol: [decode_lookahead(lookahead, codec) for lookahead in sorted(values)]
                for symbol, values in sorted(first_k.items())
            },
            "follow_k": {
                symbol: [decode_lookahead(lookahead, codec) for lookahead in sorted(values)]
                for symbol, values in sorted(follow_k.items())
            },
        }
    return result
