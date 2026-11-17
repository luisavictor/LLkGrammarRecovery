"""Construct the broad anonymized source PCFG and sample treebanks.

This module implements the Brown/Stanza-based source grammar construction used
before LL(k) extraction: read Brown sentences, parse them into constituency
trees, estimate structural PCFG rules, replace lexical items with anonymous
fixed-length tokens, assign Zipfian lexical probabilities, and sample
Gaussian-filtered sentence/tree pairs.
"""

import stanza
from collections import Counter, defaultdict
from nltk import Tree
import math
from pathlib import Path
import random
import nltk
from nltk.corpus import brown
import os
import json
from nltk import PCFG, Nonterminal, ProbabilisticProduction
import numpy as np
import torch

from A_PCFG.LLk_core import (
    index_productions,
    save_sentence_tree_pairs,
    mle_estimate_pcfg,
)


def set_global_seed(seed: int, deterministic_torch: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic_torch:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass





def read_brown_sentences(categories=None, limit=None):
    """
    Read sentences from the Brown corpus.
    categories could allow us to pick text of different kind, e.g., news or fiction
    Returns a list of sentences as strings
    """
    sentences = brown.sents(categories=categories)
    sentence_texts = [" ".join(sent) for sent in sentences]
    if limit:
        sentence_texts = sentence_texts[:limit]
    return sentence_texts

def build_stanza_pipeline():
    # Set up a Stanza pipeline for English
    stanza.download("en", processors="tokenize,pos,constituency", verbose=False)
    return stanza.Pipeline("en", processors="tokenize,pos,constituency", verbose=False)

def parse_sentences(sentences, nlp, limit=None):
    """
    Parse a list of sentences into NLTK Tree objects using Stanza.
    Returns a list of NLTK Tree objects
    """
    trees = []
    for sent in sentences[:limit] if limit else sentences:
        doc = nlp(sent)
        for s in doc.sentences:
            trees.append(Tree.fromstring(str(s.constituency)))
    return trees



# 2. Collect Non- and Preterminals

def collect_nonterminals(trees):
    # Collect all productions where the right-hand side has at least one nonterminal
    structural_counts = Counter()  # how often each structural production occurs
    lhs_counts = Counter()  # how often each left-hand side occurs

    # Walk through every production in every tree
    for tree in trees:
        for prod in tree.productions():
            # Only keep rules that expand into nonterminals (not preterminals/terminals)
            if any(isinstance(rhs, Nonterminal) for rhs in prod.rhs()):
                structural_counts[prod] += 1
                lhs_counts[prod.lhs()] += 1

    nonterm_prods = []
    # Compute probabilities for each structural production
    for prod, count in structural_counts.items():
        prob = count / lhs_counts[prod.lhs()]  # relative frequency
        nonterm_prods.append(
            ProbabilisticProduction(prod.lhs(), prod.rhs(), prob=prob)
        )
    return nonterm_prods

def collect_preterminal_vocab(trees):
    # Collect statistics for preterminals
    preterm_to_words = defaultdict(set)
    word_to_preterms = defaultdict(set)

    # collect word-preterminal mappings
    for tree in trees:
        for prod in tree.productions():
            # Preterminal rules look like: NN -> 'dog'
            if all(isinstance(rhs, str) for rhs in prod.rhs()):
                lhs = prod.lhs()
                for word in prod.rhs():
                    preterm_to_words[lhs].add(word)
                    word_to_preterms[word].add(lhs)

    # compute how much vocab each preterminal covers
    preterm_counts = {nt: len(words) for nt, words in preterm_to_words.items()}
    total_vocab = sum(preterm_counts.values())
    shares = {nt: count / total_vocab for nt, count in preterm_counts.items()}
    print("preterm counts", preterm_counts)
    print("total vocab", total_vocab)

    # find overlaps (words that belong to multiple preterminals, might be the case for natural language/we want to cover this case)
    combo_counter = Counter()
    for word, groups in word_to_preterms.items():
        if len(groups) > 1:  # word appears under more than one tag
            combo_counter[tuple(sorted(groups))] += 1

    overlap_rules = []
    for combo, count in combo_counter.items():
        proportions = {}
        for nt in combo:
            proportions[nt] = count / len(preterm_to_words[nt])  # proportion relative to this preterminal
        overlap_rules.append((list(combo), proportions))

    # sanity checks
    for lhs, words in preterm_to_words.items():
        assert isinstance(lhs, Nonterminal), f"LHS {lhs} is not Nonterminal!"
        for w in words:
            assert isinstance(w, str), f"Word {w} is not str!"

    for w, nts in word_to_preterms.items():
        assert isinstance(w, str), f"Word {w} is not str!"
        for nt in nts:
            assert isinstance(nt, Nonterminal), f"Preterminal {nt} is not Nonterminal!"

    return shares, overlap_rules



# 3. Assign synthetic vocab to preterminals
def generate_vocab(n, length, vocab_file, rng: random.Random):
    # Generate a synthetic vocabulary of n unique random strings.
    vocab, seen = [], set()
    os.makedirs(os.path.dirname(vocab_file), exist_ok=True)
    while len(vocab) < n:
        w = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=length)) # for experiments, we set length = 5
        if w not in seen:
            seen.add(w)
            vocab.append(w)
    with open(vocab_file, "w") as f:
        for word in vocab:
            f.write(word + "\n")
    print("len of generated vocab", len(vocab))
    return vocab

def assign_words_to_groups(num_words, shares, overlap_rules, seed, word_length, vocab_file):
    # Assign words to preterminals with shares and overlaps,
    rng = random.Random(seed)
    preterms = list(shares.keys())

    # generate synthetic vocabulary
    vocab = generate_vocab(num_words, word_length, vocab_file, rng=rng)

    # compute adjusted total count to account for overlaps
    overlap_fraction_total = 0
    for groups, prop_dict in overlap_rules:
        ref = groups[0]  # first preterminal as reference
        overlap_fraction_total += shares[ref] * prop_dict[ref]

    total_adjusted = int(round(num_words / (1 - overlap_fraction_total)))

    # compute target counts per preterminal
    target_counts = {nt: int(round(total_adjusted * shares[nt])) for nt in preterms}

    # assign words
    assignment = defaultdict(list)
    used_words = set()

    # assign overlapping words
    for groups, prop_dict in overlap_rules:
        ref = groups[0]
        overlap_count = int(round(target_counts[ref] * prop_dict[ref]))
        words_for_overlap = [w for w in vocab if w not in used_words][:overlap_count]
        for nt in groups:
            assignment[nt].extend(words_for_overlap)
        used_words.update(words_for_overlap)

    # assign remaining words to reach target count per preterminal
    for nt, count in target_counts.items():
        remaining_needed = count - len(assignment[nt])
        if remaining_needed <= 0:
            continue
        words_to_add = [w for w in vocab if w not in used_words][:remaining_needed]
        assignment[nt].extend(words_to_add)
        used_words.update(words_to_add)

    # assign any leftover words randomly (seeded)
    remaining_words = [w for w in vocab if w not in used_words]
    for w in remaining_words:
        chosen_nt = rng.choice(preterms)
        assignment[chosen_nt].append(w)
        used_words.add(w)

    # Print summary
    print("Final assignment sizes (including overlaps):")
    for nt in preterms:
        print(f"{nt}: {len(assignment[nt])} words")
    for nt, words in assignment.items():
        assert isinstance(nt, Nonterminal), f"{nt} is not a Nonterminal!"
        for w in words:
            assert isinstance(w, str), f"{w} is not a string!"
    distinct_words = set()
    for words in assignment.values():
        distinct_words.update(words)
    print("Total distinct words in assignment:", len(distinct_words))

    return assignment, vocab


# 4. Filter and normalize grammar
def filter_and_build_pcfg(nonterm_prods, assignment, seed, start_symbol):
    """
    Build a PCFG using a simple backward reachability:
      - Keep only nonterminals that can eventually reach terminals (since our terminal set is different from the original corpus)
      - Remove nonproductive rules  (rules that can never procue a terminal)
      - Renormalize rule probabilities
    """
    rng = random.Random(seed)

    # identify initially productive symbols (preterminals with words)
    productive = {nt for nt, words in assignment.items() if words}
    print("len(assignments)", len(assignment))
    print(f"Initial productive preterminals: {productive}")

    # build mapping LHS -> RHS lists
    lhs_to_rhs = defaultdict(list)
    for prod in nonterm_prods:
        lhs_to_rhs[prod.lhs()].append(list(prod.rhs()))

    # iteratively mark productive nonterminals
    changed = True
    while changed:
        changed = False
        for lhs, rhss in lhs_to_rhs.items():
            if lhs in productive:
                continue
            for rhs in rhss:
                if all((not isinstance(sym, Nonterminal)) or (sym in productive) for sym in rhs):
                    productive.add(lhs)
                    changed = True
                    break
    print(f"Final productive nonterminals: {productive}")

    # filter nonterminal productions
    filtered_prods = []
    for prod in nonterm_prods:
        lhs = prod.lhs()
        rhs = prod.rhs()
        if lhs not in productive:
            continue
        if all((not isinstance(sym, Nonterminal)) or (sym in productive) for sym in rhs):
            filtered_prods.append(prod)

    # renormalize probabilities per LHS
    grouped = defaultdict(list)
    for prod in filtered_prods:
        grouped[prod.lhs()].append(prod)

    normalized_prods = []
    for lhs, prods in grouped.items():
        total_prob = sum(p.prob() for p in prods)
        for p in prods:
            normalized_prods.append(
                ProbabilisticProduction(lhs, p.rhs(), prob=p.prob() / total_prob)
            )
    print(f"Filtered nonterminal productions: {len(normalized_prods)}")

    # add terminal productions using Zipf's law
    distinct_terminals = set()
    terminal_prods = []
    alpha, beta = 1.0, 2.7

    for nt, words in assignment.items():
        if not words or nt not in productive:
            print("error for", nt, words)
            continue

        shuffled_words = words[:]  # copy
        rng.shuffle(shuffled_words)  # seeded shuffle

        ranks = np.arange(1, len(shuffled_words) + 1)
        zipf_probs = 1 / ((ranks + beta) ** alpha)
        zipf_probs /= zipf_probs.sum()

        for w, p in zip(shuffled_words, zipf_probs):
            terminal_prods.append(ProbabilisticProduction(nt, (w,), prob=float(p)))
            distinct_terminals.add(w)

        print(f"Added {len(words)} terminal productions for {nt}")


    print("Total distinct terminals added:", len(distinct_terminals))
    print("Preterminals with assigned words:", len(assignment))
    print("Productive nonterminals after filtering:", len(productive))
    # build PCFG from nonterminal and terminal rules
    grammar = PCFG(Nonterminal(start_symbol), normalized_prods + terminal_prods)
    print(f"Total productions in final grammar: {len(grammar.productions())}")
    return grammar


# 5. Save grammar in EarleyX-compatible format
def save_grammar(grammar, path):
    with open(path, "w", encoding="utf-8") as f:
        for prod in grammar.productions():
            lhs = prod.lhs()
            rhs_items = []
            for s in prod.rhs():
                s_str = s
                rhs_items.append(f"_{s_str}" if isinstance(s, str) else str(s_str))
            rhs = " ".join(rhs_items)
            prob = prod.prob() if hasattr(prod, "prob") else 1.0
            f.write(f"{lhs}->[{rhs}] : {prob:.8f}\n")


# 6. Generate dataset
def sample_tree_from_pcfg(grammar, start_symbol, rng, max_depth=200, prod_table=None):
    # Recursively sample a parse tree from a PCFG.
    def expand(sym, depth=0):
        if depth >= max_depth:
            return Tree(str(sym), [])   # avoid infinite expansions, stop early

        if isinstance(sym, str):
            return sym   # terminal reached

        rules = prod_table[sym]
        rhs, _ = rng.choices(rules, weights=[p for _, p in rules], k=1)[0]  # choose next rule
        children = [expand(s, depth + 1) for s in rhs] #apply next rule recursively
        return Tree(str(sym), children)

    return expand(start_symbol)

def sample_tree_and_sentence(grammar, rng, prod_table):
    tree = sample_tree_from_pcfg(grammar, grammar.start(), rng=rng, prod_table=prod_table)
    sentence = " ".join(tree.leaves())
    return tree, sentence

def accept_sentence(sentence, mu, sigma, rng):
    # Accept a sentence based on a Gaussian-shaped acceptance probability over sentence length
    length = len(sentence)
    prob = math.exp(-0.5 * ((length - mu) / sigma) ** 2)
    prob /= (sigma * math.sqrt(2 * math.pi))

    max_pdf = 1 / (sigma * math.sqrt(2 * math.pi))
    acceptance = prob / max_pdf  # accept a sentence probabilistically
    return rng.random() < acceptance

def write_dataset_checkpoint(sentences, trees, output_dir):
    print(f"Writing dataset checkpoint with {len(sentences)} sentences to {output_dir}", flush=True)
    Path(output_dir, "data.txt").write_text("\n".join(sentences), encoding="utf-8")
    save_sentence_tree_pairs(
        sentences=sentences,
        trees=trees,
        path=Path(output_dir, "data_trees.txt"),
    )


def generate_datasets_with_trees(grammar, train_sizes, test_size,
                                 mean_len, stdev_len, base_seed, output_dir,
                                 checkpoint_interval=100_000,
                                 progress_interval=1_000,
                                 max_sampling_attempts=None):
    # Generate a dataset from the grammar by sampling parse trees, store sentences and parse trees for reestimating the final oracle grammar
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    rng = random.Random(base_seed)
    total_required = max(train_sizes.values()) + int(max(train_sizes.values()) * 0.2) + test_size # sample a set of sentences that sufficiently large for the experiments
    prod_table = index_productions(grammar)
    print(
        f"Generating dataset: target={total_required}, ",
        flush=True,
    )

    trees, sentences = [], []
    attempts = 0
    while len(sentences) < total_required:
        attempts += 1
        if max_sampling_attempts is not None and attempts > max_sampling_attempts:
            raise RuntimeError(
                f"Stopped after {max_sampling_attempts} sampling attempts with "
                f"{len(sentences)}/{total_required} accepted sentences."
            )
        tree, sentence = sample_tree_and_sentence(grammar, rng=rng, prod_table=prod_table)
        if accept_sentence(tree.leaves(), mean_len, stdev_len, rng=rng):
            trees.append(tree)
            sentences.append(sentence)
            if checkpoint_interval and len(sentences) % checkpoint_interval == 0:
                write_dataset_checkpoint(sentences, trees, output_dir)

    combined = list(zip(sentences, trees))
    rng.shuffle(combined)
    sentences, trees = zip(*combined)
    write_dataset_checkpoint(sentences, trees, output_dir)
    return list(trees)


def load_sentence_tree_index(path):
    index = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            index[row["sentence"]].append(row["tree"])
    return dict(index)


# 7. Reestimate grammar via MLE

# allows to load grammar stored in earleyx format
def load_grammar_earleyx(path):
    productions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            import re
            lhs_rhs, prob_str = re.split(r'(?<=\])\s*:', line, maxsplit=1)
            prob = float(prob_str)

            lhs_text, rhs_text = lhs_rhs.split("->")
            lhs = Nonterminal(lhs_text.strip())

            rhs_text = rhs_text.strip()[1:-1].strip()
            rhs_symbols = []
            if rhs_text:
                for sym in rhs_text.split():
                    if sym.startswith("_"):
                        rhs_symbols.append(sym[1:])
                    else:
                        rhs_symbols.append(Nonterminal(sym))

            prod = ProbabilisticProduction(lhs, rhs_symbols, prob=prob)
            productions.append(prod)

    start_symbol = productions[0].lhs()
    grammar = PCFG(start_symbol, productions)
    print("Number of distinct terminals in grammar:", len(set(grammar._lexical_index.keys())))
    return grammar


# Main
def run_all(path_grammar, BASE_SEED, VOCAB_SIZE, WORD_LENGTH,
            OUTPUT_DIR, TRAIN_SIZES, TEST_SIZE, MEAN_LEN, STDEV_LEN,
            VOCAB_FILE, limit):
    """
    Steps that are done here:
    1) Parse real sentences into trees
    2) Collect nonterminal rules + preterminal vocab stats
    3) Assign synthetic vocab
    4) Filter/normalize grammar  (grammar G that we will NOT use as oracle)
    5) Sample datasets with trees from G
    6) MLE reestimate grammar and save (grammar G' that we will use as oracle)
    """
    set_global_seed(BASE_SEED)

    nltk.download("brown")
    nltk.download("punkt")

    sentences = read_brown_sentences()
    nlp = build_stanza_pipeline()
    trees = parse_sentences(sentences, nlp, limit=limit)

    nonterm_prods = collect_nonterminals(trees)
    shares, overlap_rules = collect_preterminal_vocab(trees)

    assignment, vocab = assign_words_to_groups(
        num_words=VOCAB_SIZE,
        shares=shares,
        overlap_rules=overlap_rules,
        seed=BASE_SEED,
        word_length=WORD_LENGTH,
        vocab_file=VOCAB_FILE
    )

    start_symbol = trees[0].label()
    grammar = filter_and_build_pcfg(nonterm_prods, assignment, BASE_SEED, start_symbol) # generates a grammar that we will use for sampling sentences and trees, not the final oracle grammar but typically very close to it
    save_grammar(grammar, path_grammar)
    grammar = load_grammar_earleyx(path_grammar) # used this during debugging, not really necessary
    all_trees = generate_datasets_with_trees(
        grammar,
        TRAIN_SIZES,
        TEST_SIZE,
        MEAN_LEN,
        STDEV_LEN,
        BASE_SEED,
        OUTPUT_DIR
    )
    grammar_mle = mle_estimate_pcfg(grammar.start(), all_trees) # this is the final grammar we use as oracle
    print("Number of distinct terminals in grammar_mle:", len(set(grammar_mle._lexical_index.keys())))
    save_grammar(grammar_mle, Path(OUTPUT_DIR) / "grammar_mle.txt")
