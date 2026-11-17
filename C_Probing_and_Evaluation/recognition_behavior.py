"""Recognition-behavior evaluation for induced grammars.

For each k in {1, 2, 3}, this script compares exactly two grammars:

* the final-layer probe-induced grammar G_ip(k)
* the corresponding gold evaluation grammar G_LL(k)^test

on exactly two sentence sources:

* S_M_LL(k): samples drawn internally from model M_k using the provided seed(s)
* S_GS: samples drawn internally from the broader source grammar using the same
  provided seed(s)

It reports mean and standard deviation across the provided seeds for acceptance
rates and for the Jaccard similarity between the two accepted sentence sets for
each source.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Sequence, Set, Tuple

try:
    from nltk import Nonterminal, PCFG, ProbabilisticProduction
    from nltk.parse import EarleyChartParser
except Exception as exc:  # pragma: no cover - dependency failure path
    raise SystemExit(
        "This script requires nltk. Install/use the environment that contains nltk "
        f"before running it. Import error: {exc}"
    )


GRAMMAR_LINE_RE = re.compile(
    r"^\s*(?P<lhs>\S+)\s*->\s*\[(?P<rhs>.*?)\]\s*:\s*"
    r"(?P<prob>[0-9eE+\-.]+)\s*$"
)
ROOT_SYMBOL = "ROOT"
K_VALUES = (1, 2, 3)
SOURCE_ORDER = ("S_M_LL(k)", "S_GS")
GRAMMAR_ORDER = ("G_ip(k)", "G_LL(k)^test")
PARSER_CHOICES = ("earley", "dummy-earley")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


@dataclass(frozen=True)
class SentenceRecord:
    text: str
    tokens: Tuple[str, ...]
    line_number: int


@dataclass(frozen=True)
class ExperimentSetting:
    k: int
    induced_grammars: Tuple[Path, ...]
    gold_grammars: Tuple[Path, ...]
    lm_models: Optional[Tuple[Path, ...]]
    lm_tokenizers: Optional[Tuple[Path, ...]]


@dataclass(frozen=True)
class DummyEarleyState:
    lhs: str
    rhs: Tuple[str, ...]
    dot: int
    origin: int


@dataclass
class DummyEarleyRuntime:
    grammar: Dict[str, List[Tuple[str, ...]]]
    dummy_lexicon: Dict[str, Set[str]]
    start_symbol: str


@dataclass
class GrammarRuntime:
    name: str
    path: Path
    grammar: PCFG
    parser_mode: str
    start_symbol: str
    terminal_symbols: Set[str]
    earley_parser: Optional[EarleyChartParser] = None
    dummy_earley: Optional[DummyEarleyRuntime] = None


@dataclass
class LmRuntime:
    model_path: Path
    tokenizer_path: Path
    model: Any
    tokenizer: Any
    device: Any
    torch: Any


def parse_earleyx_rhs(raw_rhs: str) -> Tuple[object, ...]:
    rhs: List[object] = []
    for token in raw_rhs.strip().split():
        if token.startswith("_"):
            terminal = token[1:]
            if not terminal:
                raise ValueError("Grammar contains an empty lexical terminal.")
            rhs.append(terminal)
        else:
            rhs.append(Nonterminal(token))
    return tuple(rhs)


def load_grammar_earleyx(path: Path) -> PCFG:
    productions: List[ProbabilisticProduction] = []
    seen_lhs: List[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            match = GRAMMAR_LINE_RE.match(stripped)
            if match is None:
                raise ValueError(f"Cannot parse grammar line {line_no} in {path}: {stripped}")
            lhs_text = match.group("lhs")
            productions.append(
                ProbabilisticProduction(
                    Nonterminal(lhs_text),
                    parse_earleyx_rhs(match.group("rhs")),
                    prob=float(match.group("prob")),
                )
            )
            seen_lhs.append(lhs_text)
    if not productions:
        raise ValueError(f"No productions loaded from {path}.")
    return PCFG(Nonterminal(seen_lhs[0]), productions)


def grammar_lhs_symbols(grammar: PCFG) -> Set[str]:
    return {str(prod.lhs()) for prod in grammar.productions()}


def grammar_terminal_symbols(grammar: PCFG) -> Set[str]:
    return {
        str(symbol)
        for prod in grammar.productions()
        for symbol in prod.rhs()
        if isinstance(symbol, str)
    }


def resolve_start_symbol(grammar: PCFG, requested: Optional[str]) -> str:
    available = grammar_lhs_symbols(grammar)
    if requested is not None:
        if requested not in available:
            preview = ", ".join(sorted(available)[:25])
            raise ValueError(
                f"Requested start symbol {requested!r} is absent. "
                f"Available symbols include: {preview}"
            )
        return requested
    if ROOT_SYMBOL in available:
        return ROOT_SYMBOL
    return str(grammar.start())


def with_start_symbol(grammar: PCFG, start_symbol: str) -> PCFG:
    start = Nonterminal(start_symbol)
    return grammar if grammar.start() == start else PCFG(start, list(grammar.productions()))


def dedupe_rhss(rhss: Iterable[Tuple[str, ...]]) -> List[Tuple[str, ...]]:
    seen: Set[Tuple[str, ...]] = set()
    out: List[Tuple[str, ...]] = []
    for rhs in rhss:
        if rhs not in seen:
            seen.add(rhs)
            out.append(rhs)
    return out


def build_dummy_earley_runtime(grammar: PCFG, start_symbol: str) -> DummyEarleyRuntime:
    schema: Dict[str, List[Tuple[str, ...]]] = defaultdict(list)
    dummy_lexicon: Dict[str, Set[str]] = defaultdict(set)
    nonterminals = grammar_lhs_symbols(grammar)

    for prod in grammar.productions():
        lhs = str(prod.lhs())
        rhs = prod.rhs()
        if len(rhs) == 1 and isinstance(rhs[0], str):
            dummy_terminal = f"__lex_{lhs}"
            dummy_lexicon[dummy_terminal].add(str(rhs[0]))
            schema[lhs].append((dummy_terminal,))
        else:
            schema[lhs].append(tuple(str(symbol) for symbol in rhs))

    for lhs in nonterminals:
        schema.setdefault(lhs, [])

    compact_schema = {
        lhs: dedupe_rhss(rhss)
        for lhs, rhss in schema.items()
        if rhss
    }
    if start_symbol not in compact_schema:
        raise ValueError(f"Start symbol {start_symbol!r} has no productions.")
    return DummyEarleyRuntime(
        grammar=compact_schema,
        dummy_lexicon={key: set(value) for key, value in dummy_lexicon.items()},
        start_symbol=start_symbol,
    )


def prepare_runtime(
    name: str,
    path: Path,
    *,
    parser_mode: str,
    start_symbol: Optional[str],
) -> GrammarRuntime:
    grammar = load_grammar_earleyx(path)
    resolved_start = resolve_start_symbol(grammar, start_symbol)
    grammar = with_start_symbol(grammar, resolved_start)
    runtime = GrammarRuntime(
        name=name,
        path=path,
        grammar=grammar,
        parser_mode=parser_mode,
        start_symbol=resolved_start,
        terminal_symbols=grammar_terminal_symbols(grammar),
    )
    if parser_mode == "earley":
        runtime.earley_parser = EarleyChartParser(grammar)
    elif parser_mode == "dummy-earley":
        runtime.dummy_earley = build_dummy_earley_runtime(grammar, resolved_start)
    else:  # pragma: no cover - argparse prevents this
        raise ValueError(f"Unknown parser mode: {parser_mode}")
    return runtime


def sentence_text_from_line(line: str, path: Path, line_number: int) -> str:
    stripped = line.strip()
    if not stripped:
        return ""
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return stripped
    if isinstance(payload, dict):
        if "sentence" in payload:
            return str(payload["sentence"]).strip()
        if "text" in payload:
            return str(payload["text"]).strip()
        raise ValueError(
            f"JSONL row {line_number} in {path} has neither 'sentence' nor 'text'."
        )
    if isinstance(payload, str):
        return payload.strip()
    return stripped


def load_sentences(path: Path, max_sentences: Optional[int]) -> List[SentenceRecord]:
    records: List[SentenceRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = sentence_text_from_line(line, path, line_number)
            tokens = tuple(token for token in text.split() if token)
            if not tokens:
                continue
            records.append(SentenceRecord(text=text, tokens=tokens, line_number=line_number))
            if max_sentences is not None and len(records) >= max_sentences:
                break
    return records


def weighted_productions(
    grammar: PCFG,
) -> Dict[Nonterminal, Tuple[List[ProbabilisticProduction], List[float]]]:
    grouped: Dict[Nonterminal, List[ProbabilisticProduction]] = defaultdict(list)
    for production in grammar.productions():
        grouped[production.lhs()].append(production)
    return {
        lhs: (productions, [production.prob() for production in productions])
        for lhs, productions in grouped.items()
    }


def sample_tokens_from_grammar(
    productions_by_lhs: Dict[Nonterminal, Tuple[List[ProbabilisticProduction], List[float]]],
    *,
    start_symbol: str,
    rng: random.Random,
    max_depth: int,
) -> Tuple[str, ...]:
    def expand(symbol: object, depth: int) -> List[str]:
        if depth > max_depth:
            raise RecursionError(f"Exceeded source-sampling max depth {max_depth}.")
        if isinstance(symbol, str):
            return [symbol]
        assert isinstance(symbol, Nonterminal)
        productions, weights = productions_by_lhs[symbol]
        production = rng.choices(productions, weights=weights, k=1)[0]
        tokens: List[str] = []
        for rhs_symbol in production.rhs():
            tokens.extend(expand(rhs_symbol, depth + 1))
        return tokens

    return tuple(expand(Nonterminal(start_symbol), 0))


def sample_source_records(
    grammar_path: Path,
    *,
    seed: int,
    n_sentences: int,
    start_symbol: Optional[str],
    max_depth: int,
    max_attempts_per_sentence: int,
    log_prefix: Optional[str] = None,
) -> List[SentenceRecord]:
    started_at = time.monotonic()
    if log_prefix is not None:
        log(f"{log_prefix}: loading source grammar {grammar_path}")
    grammar = load_grammar_earleyx(grammar_path)
    resolved_start = resolve_start_symbol(grammar, start_symbol)
    grammar = with_start_symbol(grammar, resolved_start)
    productions_by_lhs = weighted_productions(grammar)
    rng = random.Random(seed)
    records: List[SentenceRecord] = []
    next_progress = 10

    while len(records) < n_sentences:
        for _ in range(max_attempts_per_sentence):
            try:
                tokens = sample_tokens_from_grammar(
                    productions_by_lhs,
                    start_symbol=resolved_start,
                    rng=rng,
                    max_depth=max_depth,
                )
            except RecursionError:
                continue
            if tokens:
                records.append(
                    SentenceRecord(
                        text=" ".join(tokens),
                        tokens=tokens,
                        line_number=len(records) + 1,
                    )
                )
                break
        else:
            raise RuntimeError(
                "Could not sample a non-empty S_GS sentence after "
                f"{max_attempts_per_sentence} attempts for seed {seed}."
            )
        if log_prefix is not None and (
            len(records) >= next_progress or len(records) == n_sentences
        ):
            log(f"{log_prefix}: sampled {len(records)}/{n_sentences} S_GS sentence(s)")
            while next_progress <= len(records):
                next_progress += 10
    if log_prefix is not None:
        log(f"{log_prefix}: finished S_GS sampling in {time.monotonic() - started_at:.1f}s")
    return records


def prepare_lm_runtime(
    model_path: Path,
    tokenizer_path: Optional[Path],
    *,
    log_prefix: Optional[str] = None,
) -> LmRuntime:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - dependency failure path
        raise SystemExit(
            "LM sampling requires torch and transformers in the active environment. "
            f"Import error: {exc}"
        )

    tokenizer_source = tokenizer_path if tokenizer_path is not None else model_path
    started_at = time.monotonic()
    if log_prefix is not None:
        log(f"{log_prefix}: loading tokenizer from {tokenizer_source}")
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_source))
    if log_prefix is not None:
        log(f"{log_prefix}: loading model from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(str(model_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if log_prefix is not None:
        log(f"{log_prefix}: moving model to {device}")
    model.to(device)
    model.eval()

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    start_id = tokenizer.bos_token_id
    if start_id is None and tokenizer.eos_token_id is None:
        raise ValueError(
            f"Tokenizer at {tokenizer_source} has neither BOS nor EOS token for generation."
        )
    if log_prefix is not None:
        log(f"{log_prefix}: model ready in {time.monotonic() - started_at:.1f}s")
    return LmRuntime(
        model_path=model_path,
        tokenizer_path=tokenizer_source,
        model=model,
        tokenizer=tokenizer,
        device=device,
        torch=torch,
    )


def sample_lm_records(
    runtime: LmRuntime,
    *,
    seed: int,
    n_sentences: int,
    max_new_tokens: int,
    batch_size: int,
    log_prefix: Optional[str] = None,
) -> List[SentenceRecord]:
    started_at = time.monotonic()
    tokenizer = runtime.tokenizer
    model = runtime.model
    device = runtime.device
    torch = runtime.torch
    start_id = tokenizer.bos_token_id
    if start_id is None:
        start_id = tokenizer.eos_token_id

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if log_prefix is not None:
        log(
            f"{log_prefix}: generating {n_sentences} S_M_LL(k) sentence(s) "
            f"with batch_size={batch_size}, max_new_tokens={max_new_tokens}"
        )
    records: List[SentenceRecord] = []
    with torch.no_grad():
        while len(records) < n_sentences:
            before_batch = len(records)
            current_batch_size = min(batch_size, n_sentences - len(records))
            input_ids = torch.full(
                (current_batch_size, 1),
                fill_value=start_id,
                dtype=torch.long,
                device=device,
            )
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for text in texts:
                stripped = text.strip()
                tokens = tuple(token for token in stripped.split() if token)
                if not tokens:
                    continue
                records.append(
                    SentenceRecord(
                        text=stripped,
                        tokens=tokens,
                        line_number=len(records) + 1,
                    )
                )
                if len(records) >= n_sentences:
                    break
            if log_prefix is not None:
                log(
                    f"{log_prefix}: generated {len(records)}/{n_sentences} "
                    f"S_M_LL(k) sentence(s) (+{len(records) - before_batch})"
                )
    if log_prefix is not None:
        log(f"{log_prefix}: finished LM generation in {time.monotonic() - started_at:.1f}s")
    return records


def has_oov_tokens(runtime: GrammarRuntime, tokens: Sequence[str]) -> bool:
    return any(token not in runtime.terminal_symbols for token in tokens)


def accepts_earley(runtime: GrammarRuntime, tokens: Sequence[str]) -> bool:
    if has_oov_tokens(runtime, tokens):
        return False
    assert runtime.earley_parser is not None
    return next(runtime.earley_parser.parse(tokens), None) is not None


def next_symbol(state: DummyEarleyState) -> Optional[str]:
    return None if state.dot >= len(state.rhs) else state.rhs[state.dot]


def dummy_terminal_matches(runtime: DummyEarleyRuntime, expected: str, observed: str) -> bool:
    lexical_tokens = runtime.dummy_lexicon.get(expected)
    return observed in lexical_tokens if lexical_tokens is not None else observed == expected


def accepts_dummy_earley(
    runtime: GrammarRuntime,
    tokens: Sequence[str],
    *,
    max_parser_steps: int,
    max_parser_seconds: Optional[float],
) -> bool:
    if has_oov_tokens(runtime, tokens):
        return False
    assert runtime.dummy_earley is not None
    dummy = runtime.dummy_earley
    start_lhs = "__RECOGNITION_START__"
    while start_lhs in dummy.grammar:
        start_lhs = f"_{start_lhs}"

    started_at = time.monotonic()
    n_tokens = len(tokens)
    charts: List[Set[DummyEarleyState]] = [set() for _ in range(n_tokens + 1)]
    waiting: List[Dict[str, Set[DummyEarleyState]]] = [
        defaultdict(set) for _ in range(n_tokens + 1)
    ]
    completed: List[Dict[str, Set[Tuple[int, DummyEarleyState]]]] = [
        defaultdict(set) for _ in range(n_tokens + 1)
    ]
    agenda: Deque[Tuple[int, DummyEarleyState]] = deque()
    accept_state = DummyEarleyState(start_lhs, (dummy.start_symbol,), 1, 0)
    accepted = False

    def add_state(position: int, state: DummyEarleyState) -> None:
        nonlocal accepted
        if state in charts[position]:
            return
        charts[position].add(state)
        if position == n_tokens and state == accept_state:
            accepted = True
        symbol = next_symbol(state)
        if symbol is not None:
            waiting[position][symbol].add(state)
            if symbol in dummy.grammar:
                for end_position, _ in list(completed[position].get(symbol, ())):
                    add_state(
                        end_position,
                        DummyEarleyState(state.lhs, state.rhs, state.dot + 1, state.origin),
                    )
        else:
            completed[state.origin][state.lhs].add((position, state))
        agenda.append((position, state))

    add_state(0, DummyEarleyState(start_lhs, (dummy.start_symbol,), 0, 0))
    steps = 0
    while agenda and not accepted:
        position, state = agenda.popleft()
        steps += 1
        if steps > max_parser_steps:
            raise RuntimeError(f"Exceeded --max-parser-steps={max_parser_steps}.")
        if max_parser_seconds is not None and time.monotonic() - started_at >= max_parser_seconds:
            raise RuntimeError(f"Exceeded --max-parser-seconds={max_parser_seconds:g}.")

        symbol = next_symbol(state)
        if symbol is None:
            for previous in list(waiting[state.origin].get(state.lhs, ())):
                add_state(
                    position,
                    DummyEarleyState(previous.lhs, previous.rhs, previous.dot + 1, previous.origin),
                )
            continue
        if symbol in dummy.grammar:
            for rhs in dummy.grammar[symbol]:
                add_state(position, DummyEarleyState(symbol, rhs, 0, position))
            continue
        if position < n_tokens and dummy_terminal_matches(dummy, symbol, tokens[position]):
            add_state(
                position + 1,
                DummyEarleyState(state.lhs, state.rhs, state.dot + 1, state.origin),
            )
    return accepted or accept_state in charts[n_tokens]


def accepts(
    runtime: GrammarRuntime,
    tokens: Sequence[str],
    *,
    max_parser_steps: int,
    max_parser_seconds: Optional[float],
) -> bool:
    if runtime.parser_mode == "earley":
        return accepts_earley(runtime, tokens)
    return accepts_dummy_earley(
        runtime,
        tokens,
        max_parser_steps=max_parser_steps,
        max_parser_seconds=max_parser_seconds,
    )


def evaluate_source(
    source_name: str,
    records: Sequence[SentenceRecord],
    induced: GrammarRuntime,
    gold: GrammarRuntime,
    *,
    max_parser_steps: int,
    max_parser_seconds: Optional[float],
    log_prefix: Optional[str] = None,
) -> Tuple[Dict[str, object], Dict[str, object]]:
    started_at = time.monotonic()
    if log_prefix is not None:
        log(f"{log_prefix}: evaluating {len(records)} {source_name} sentence(s)")
    induced_accepts: Set[int] = set()
    gold_accepts: Set[int] = set()
    progress_every = max(1, min(25, len(records) // 4 if records else 1))
    for index, record in enumerate(records):
        if accepts(
            induced,
            record.tokens,
            max_parser_steps=max_parser_steps,
            max_parser_seconds=max_parser_seconds,
        ):
            induced_accepts.add(index)
        if accepts(
            gold,
            record.tokens,
            max_parser_steps=max_parser_steps,
            max_parser_seconds=max_parser_seconds,
        ):
            gold_accepts.add(index)
        completed = index + 1
        if log_prefix is not None and (
            completed == len(records) or completed % progress_every == 0
        ):
            log(
                f"{log_prefix}: evaluated {completed}/{len(records)} {source_name} "
                f"sentence(s)"
            )

    total = len(records)
    acceptance_row = {
        "source": source_name,
        "total_sentences": total,
        "induced_accepted": len(induced_accepts),
        "induced_acceptance_rate": len(induced_accepts) / total if total else None,
        "gold_accepted": len(gold_accepts),
        "gold_acceptance_rate": len(gold_accepts) / total if total else None,
    }
    union = induced_accepts | gold_accepts
    jaccard_row = {
        "source": source_name,
        "intersection": len(induced_accepts & gold_accepts),
        "union": len(union),
        "jaccard_similarity": len(induced_accepts & gold_accepts) / len(union) if union else 1.0,
    }
    if log_prefix is not None:
        log(
            f"{log_prefix}: done {source_name} in {time.monotonic() - started_at:.1f}s "
            f"(induced={len(induced_accepts)}/{total}, gold={len(gold_accepts)}/{total})"
        )
    return acceptance_row, jaccard_row


def mean_and_stdev(values: Sequence[float]) -> Tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def optional_mean_and_stdev(
    values: Sequence[Optional[float]],
) -> Tuple[Optional[float], Optional[float]]:
    if any(value is None for value in values):
        return None, None
    concrete = [float(value) for value in values if value is not None]
    return mean_and_stdev(concrete)


def aggregate_acceptance_rows(
    source_name: str,
    k: int,
    rows: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    total_mean, total_stdev = mean_and_stdev([float(row["total_sentences"]) for row in rows])
    induced_count_mean, induced_count_stdev = mean_and_stdev(
        [float(row["induced_accepted"]) for row in rows]
    )
    induced_rate_mean, induced_rate_stdev = optional_mean_and_stdev(
        [
            None if row["induced_acceptance_rate"] is None else float(row["induced_acceptance_rate"])
            for row in rows
        ]
    )
    gold_count_mean, gold_count_stdev = mean_and_stdev(
        [float(row["gold_accepted"]) for row in rows]
    )
    gold_rate_mean, gold_rate_stdev = optional_mean_and_stdev(
        [
            None if row["gold_acceptance_rate"] is None else float(row["gold_acceptance_rate"])
            for row in rows
        ]
    )
    return {
        "source": source_name,
        "k": k,
        "n_seeds": len(rows),
        "total_sentences_mean": total_mean,
        "total_sentences_stdev": total_stdev,
        "induced_accepted_mean": induced_count_mean,
        "induced_accepted_stdev": induced_count_stdev,
        "induced_acceptance_rate_mean": induced_rate_mean,
        "induced_acceptance_rate_stdev": induced_rate_stdev,
        "gold_accepted_mean": gold_count_mean,
        "gold_accepted_stdev": gold_count_stdev,
        "gold_acceptance_rate_mean": gold_rate_mean,
        "gold_acceptance_rate_stdev": gold_rate_stdev,
    }


def aggregate_jaccard_rows(
    source_name: str,
    k: int,
    rows: Sequence[Dict[str, object]],
) -> Dict[str, object]:
    intersection_mean, intersection_stdev = mean_and_stdev(
        [float(row["intersection"]) for row in rows]
    )
    union_mean, union_stdev = mean_and_stdev([float(row["union"]) for row in rows])
    jaccard_mean, jaccard_stdev = mean_and_stdev(
        [float(row["jaccard_similarity"]) for row in rows]
    )
    return {
        "source": source_name,
        "k": k,
        "n_seeds": len(rows),
        "intersection_mean": intersection_mean,
        "intersection_stdev": intersection_stdev,
        "union_mean": union_mean,
        "union_stdev": union_stdev,
        "jaccard_similarity_mean": jaccard_mean,
        "jaccard_similarity_stdev": jaccard_stdev,
    }


def build_settings(args: argparse.Namespace) -> List[ExperimentSetting]:
    k_values = tuple(getattr(args, "k_values", K_VALUES))
    return [
        ExperimentSetting(
            k=k,
            induced_grammars=tuple(getattr(args, f"k{k}_induced_grammars")),
            gold_grammars=tuple(getattr(args, f"k{k}_gold_grammars")),
            lm_models=(
                None
                if getattr(args, f"k{k}_lm_models") is None
                else tuple(getattr(args, f"k{k}_lm_models"))
            ),
            lm_tokenizers=(
                None
                if getattr(args, f"k{k}_lm_tokenizers") is None
                else tuple(getattr(args, f"k{k}_lm_tokenizers"))
            ),
        )
        for k in k_values
    ]


def ensure_existing(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        joined = "\n  - ".join(missing)
        raise SystemExit(f"Required file(s) do not exist:\n  - {joined}")


def format_rate(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def format_mean_stdev(mean: Optional[float], stdev: Optional[float]) -> str:
    if mean is None or stdev is None:
        return "n/a"
    return f"{mean:.3f}±{stdev:.3f}"


def print_acceptance_table(
    rows: Sequence[Dict[str, object]],
    k_values: Sequence[int] = K_VALUES,
) -> None:
    by_source_grammar: Dict[
        Tuple[str, str], Dict[int, Tuple[Optional[float], Optional[float]]]
    ] = defaultdict(dict)
    for row in rows:
        source = str(row["source"])
        k = int(row["k"])
        induced_rate = row["induced_acceptance_rate_mean"]
        induced_stdev = row["induced_acceptance_rate_stdev"]
        gold_rate = row["gold_acceptance_rate_mean"]
        gold_stdev = row["gold_acceptance_rate_stdev"]
        by_source_grammar[(source, "G_ip(k)")][k] = (
            None if induced_rate is None else float(induced_rate),
            None if induced_stdev is None else float(induced_stdev),
        )
        by_source_grammar[(source, "G_LL(k)^test")][k] = (
            None if gold_rate is None else float(gold_rate),
            None if gold_stdev is None else float(gold_stdev),
        )

    print("\nAcceptance rates (mean±stdev across provided seeds)")
    k_header = " ".join(f"{f'k={k}':>15}" for k in k_values)
    print(f"{'Source':<12} {'Grammar':<16} {k_header}")
    print("-" * (30 + 16 * len(k_values)))
    for source in SOURCE_ORDER:
        for grammar in GRAMMAR_ORDER:
            values = by_source_grammar[(source, grammar)]
            rendered = " ".join(
                f"{format_mean_stdev(*values.get(k, (None, None))):>15}"
                for k in k_values
            )
            print(f"{source:<12} {grammar:<16} {rendered}")


def print_jaccard_table(
    rows: Sequence[Dict[str, object]],
    k_values: Sequence[int] = K_VALUES,
) -> None:
    by_source: Dict[str, Dict[int, Tuple[float, float]]] = defaultdict(dict)
    for row in rows:
        by_source[str(row["source"])][int(row["k"])] = (
            float(row["jaccard_similarity_mean"]),
            float(row["jaccard_similarity_stdev"]),
        )

    print("\nJaccard similarity (mean±stdev across provided seeds)")
    k_header = " ".join(f"{f'k={k}':>15}" for k in k_values)
    print(f"{'Source':<12} {k_header}")
    print("-" * (14 + 16 * len(k_values)))
    for source in SOURCE_ORDER:
        values = by_source[source]
        rendered = " ".join(
            f"{format_mean_stdev(*values.get(k, (None, None))):>15}"
            for k in k_values
        )
        print(f"{source:<12} {rendered}")


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare final-layer probe-induced grammars against the corresponding "
            "gold evaluation grammars on S_M_LL(k) and S_GS."
        )
    )
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    for k in K_VALUES:
        parser.add_argument(f"--k{k}-induced-grammars", type=Path, nargs="+", required=True)
        parser.add_argument(f"--k{k}-gold-grammars", type=Path, nargs="+", required=True)
        parser.add_argument(f"--k{k}-lm-models", type=Path, nargs="+", default=None)
        parser.add_argument(f"--k{k}-lm-tokenizers", type=Path, nargs="+", default=None)
    parser.add_argument("--lm-sample-size", type=int, default=None)
    parser.add_argument("--lm-max-new-tokens", type=int, default=None)
    parser.add_argument("--lm-batch-size", type=int, default=32)
    parser.add_argument("--source-grammars", type=Path, nargs="+", required=True)
    parser.add_argument("--source-sample-size", type=int, required=True)
    parser.add_argument("--source-max-depth", type=int, default=200)
    parser.add_argument("--source-max-attempts-per-sentence", type=int, default=2_000)
    parser.add_argument("--parser", choices=PARSER_CHOICES, default="dummy-earley")
    parser.add_argument("--start-symbol", default=None)
    parser.add_argument("--max-sentences", type=int, default=None)
    parser.add_argument("--max-parser-steps", type=int, default=100_000)
    parser.add_argument("--max-parser-seconds", type=float, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--acceptance-csv", type=Path, default=None)
    parser.add_argument("--jaccard-csv", type=Path, default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_sentences is not None and args.max_sentences < 0:
        raise SystemExit("--max-sentences must be non-negative.")
    if args.max_parser_steps <= 0:
        raise SystemExit("--max-parser-steps must be positive.")
    if args.max_parser_seconds is not None and args.max_parser_seconds <= 0:
        raise SystemExit("--max-parser-seconds must be positive.")
    settings = build_settings(args)
    n_seeds = len(args.seeds)
    if len(set(args.seeds)) != n_seeds:
        raise SystemExit("--seeds must not contain duplicates.")
    if len(args.source_grammars) != n_seeds:
        raise SystemExit("--source-grammars must contain exactly one path per seed.")
    for setting in settings:
        if len(setting.induced_grammars) != n_seeds:
            raise SystemExit(
                f"--k{setting.k}-induced-grammars must contain exactly one path per seed."
            )
        if len(setting.gold_grammars) != n_seeds:
            raise SystemExit(
                f"--k{setting.k}-gold-grammars must contain exactly one path per seed."
            )
        if setting.lm_models is not None and len(setting.lm_models) != n_seeds:
            raise SystemExit(
                f"--k{setting.k}-lm-models must contain exactly one path per seed."
            )
        if setting.lm_tokenizers is not None and len(setting.lm_tokenizers) != n_seeds:
            raise SystemExit(
                f"--k{setting.k}-lm-tokenizers must contain exactly one path per seed."
            )
        if setting.lm_models is None and setting.lm_tokenizers is not None:
            raise SystemExit(
                f"--k{setting.k}-lm-tokenizers requires --k{setting.k}-lm-models."
            )
    any_lm_model = any(setting.lm_models is not None for setting in settings)
    if any_lm_model and args.lm_sample_size is None:
        raise SystemExit("--lm-sample-size is required when any --k*-lm-model is provided.")
    if any_lm_model and args.lm_max_new_tokens is None:
        raise SystemExit("--lm-max-new-tokens is required when any --k*-lm-model is provided.")
    if args.lm_sample_size is not None and args.lm_sample_size < 0:
        raise SystemExit("--lm-sample-size must be non-negative.")
    if args.lm_max_new_tokens is not None and args.lm_max_new_tokens <= 0:
        raise SystemExit("--lm-max-new-tokens must be positive.")
    if args.lm_batch_size <= 0:
        raise SystemExit("--lm-batch-size must be positive.")
    if args.source_sample_size < 0:
        raise SystemExit("--source-sample-size must be non-negative.")
    if args.source_max_depth <= 0:
        raise SystemExit("--source-max-depth must be positive.")
    if args.source_max_attempts_per_sentence <= 0:
        raise SystemExit("--source-max-attempts-per-sentence must be positive.")
    ensure_existing(
        [
            *args.source_grammars,
            *[
                path
                for setting in settings
                for path in (
                    *setting.induced_grammars,
                    *setting.gold_grammars,
                    *([] if setting.lm_models is None else setting.lm_models),
                    *([] if setting.lm_tokenizers is None else setting.lm_tokenizers),
                )
            ],
        ]
    )


def run_with_args(args: argparse.Namespace) -> Dict[str, object]:
    run_started_at = time.monotonic()
    validate_args(args)
    settings = build_settings(args)
    k_values = tuple(setting.k for setting in settings)
    source_sample_size = (
        min(args.source_sample_size, args.max_sentences)
        if args.max_sentences is not None
        else args.source_sample_size
    )
    lm_sample_size = (
        None
        if args.lm_sample_size is None
        else (
            min(args.lm_sample_size, args.max_sentences)
            if args.max_sentences is not None
            else args.lm_sample_size
        )
    )
    log(
        "Starting acceptance-matrix run "
        f"for seeds={list(args.seeds)}, source_sample_size={source_sample_size}, "
        f"lm_sample_size={lm_sample_size}, parser={args.parser}"
    )
    source_records_by_seed: Dict[int, List[SentenceRecord]] = {}
    for seed, source_grammar in zip(args.seeds, args.source_grammars):
        source_records_by_seed[seed] = sample_source_records(
            source_grammar,
            seed=seed,
            n_sentences=source_sample_size,
            start_symbol=args.start_symbol,
            max_depth=args.source_max_depth,
            max_attempts_per_sentence=args.source_max_attempts_per_sentence,
            log_prefix=f"seed={seed} S_GS",
        )

    acceptance_rows: List[Dict[str, object]] = []
    jaccard_rows: List[Dict[str, object]] = []
    grammar_rows: List[Dict[str, object]] = []

    for setting in settings:
        log(f"k={setting.k}: starting")
        per_source_acceptance_rows: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        per_source_jaccard_rows: Dict[str, List[Dict[str, object]]] = defaultdict(list)
        for index, seed in enumerate(args.seeds):
            seed_prefix = f"k={setting.k} seed={seed}"
            log(f"{seed_prefix}: preparing grammars")
            induced = prepare_runtime(
                "G_ip(k)",
                setting.induced_grammars[index],
                parser_mode=args.parser,
                start_symbol=args.start_symbol,
            )
            gold = prepare_runtime(
                "G_LL(k)^test",
                setting.gold_grammars[index],
                parser_mode=args.parser,
                start_symbol=args.start_symbol,
            )
            log(
                f"{seed_prefix}: grammars ready "
                f"(induced_start={induced.start_symbol}, gold_start={gold.start_symbol})"
            )
            grammar_rows.extend(
                [
                    {
                        "seed": seed,
                        "k": setting.k,
                        "grammar": induced.name,
                        "path": str(induced.path),
                        "start_symbol": induced.start_symbol,
                        "parser": induced.parser_mode,
                    },
                    {
                        "seed": seed,
                        "k": setting.k,
                        "grammar": gold.name,
                        "path": str(gold.path),
                        "start_symbol": gold.start_symbol,
                        "parser": gold.parser_mode,
                    },
                ]
            )

            source_to_records = {"S_GS": source_records_by_seed[seed]}
            if setting.lm_models is not None:
                assert lm_sample_size is not None
                assert args.lm_max_new_tokens is not None
                tokenizer_path = (
                    None if setting.lm_tokenizers is None else setting.lm_tokenizers[index]
                )
                lm_runtime = prepare_lm_runtime(
                    setting.lm_models[index],
                    tokenizer_path,
                    log_prefix=f"{seed_prefix} S_M_LL(k)",
                )
                source_to_records["S_M_LL(k)"] = sample_lm_records(
                    lm_runtime,
                    seed=seed,
                    n_sentences=lm_sample_size,
                    max_new_tokens=args.lm_max_new_tokens,
                    batch_size=args.lm_batch_size,
                    log_prefix=f"{seed_prefix} S_M_LL(k)",
                )
            for source_name, records in source_to_records.items():
                acceptance_row, jaccard_row = evaluate_source(
                    source_name,
                    records,
                    induced,
                    gold,
                    max_parser_steps=args.max_parser_steps,
                    max_parser_seconds=args.max_parser_seconds,
                    log_prefix=f"{seed_prefix} {source_name}",
                )
                per_source_acceptance_rows[source_name].append(acceptance_row)
                per_source_jaccard_rows[source_name].append(jaccard_row)
        for source_name in SOURCE_ORDER:
            if source_name not in per_source_acceptance_rows:
                continue
            acceptance_rows.append(
                aggregate_acceptance_rows(
                    source_name,
                    setting.k,
                    per_source_acceptance_rows[source_name],
                )
            )
            jaccard_rows.append(
                aggregate_jaccard_rows(
                    source_name,
                    setting.k,
                    per_source_jaccard_rows[source_name],
                )
            )
        log(f"k={setting.k}: finished")

    print_acceptance_table(acceptance_rows, k_values)
    print_jaccard_table(jaccard_rows, k_values)

    payload = {
        "metadata": {
            "experiment": "recognition_behavior_of_induced_grammars",
            "k_values": list(k_values),
            "sources": list(SOURCE_ORDER),
            "grammars": list(GRAMMAR_ORDER),
            "seeds": list(args.seeds),
            "parser": args.parser,
            "start_symbol": args.start_symbol,
            "max_sentences": args.max_sentences,
            "lm_sample_size": lm_sample_size,
            "lm_max_new_tokens": args.lm_max_new_tokens,
            "lm_batch_size": args.lm_batch_size,
            "lm_sampling": (
                {
                    "do_sample": True,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "top_k": 0,
                }
                if any(setting.lm_models is not None for setting in settings)
                else None
            ),
            "source_grammars": [str(path) for path in args.source_grammars],
            "source_sample_size": source_sample_size,
            "source_max_depth": args.source_max_depth,
            "source_max_attempts_per_sentence": args.source_max_attempts_per_sentence,
        },
        "grammars": grammar_rows,
        "acceptance_rates": acceptance_rows,
        "jaccard_similarity": jaccard_rows,
    }
    if args.output_json is not None:
        write_json(args.output_json, payload)
        print(f"\nWrote JSON results to {args.output_json}")
    if args.acceptance_csv is not None:
        write_csv(args.acceptance_csv, acceptance_rows)
        print(f"Wrote acceptance-rate CSV to {args.acceptance_csv}")
    if args.jaccard_csv is not None:
        write_csv(args.jaccard_csv, jaccard_rows)
        print(f"Wrote Jaccard CSV to {args.jaccard_csv}")
    log(f"Finished acceptance-matrix run in {time.monotonic() - run_started_at:.1f}s")
    return payload


def main() -> None:
    run_with_args(parse_args())


if __name__ == "__main__":
    main()
