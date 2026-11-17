"""Demonstration entry point for running the grammar-generation.

The values in this file are intentionally small/debug-oriented examples of how
to call the implementation. For full-scale arguments, use the reported
experimental settings or the argparse defaults in the underlying long modules.
"""

import B_LLMTraining.globals as G
from pathlib import Path
from A_PCFG.pcfg import run_all
from A_PCFG.llk_grammar_from_ambiguous_trees import build_llk_atomic_tree_grammar

if __name__ == '__main__':

    output_dir = G.REPO_ROOT / Path(G.OUTPUT_DIR)

    # limit should be set to None for a full run which uses all Brown Corpus sentences, here set to 10 for test runs
    run_all(
        output_dir / "grammar.txt",
        G.BASE_SEED,
        G.VOCAB_SIZE,
        G.WORD_LENGTH,
        output_dir,
        G.TRAIN_SIZES,
        G.TEST_SIZE,
        G.MEAN_LEN,
        G.STDEV_LEN,
        output_dir / "vocab.txt",
        limit = 10,
    )

    # 2. Build LL(k) grammar from pcfg.py outputs

    build_llk_atomic_tree_grammar(
        grammar_path=output_dir / "grammar_mle.txt",
        trees_path=output_dir / "data_trees.txt",
        output_dir=output_dir / "ll1",
        lookahead_k=1,
        num_trees=1000,
        tree_sample_seed=G.BASE_SEED,
        tree_sample_mode="random",
        max_rules_per_lhs=10000, # if set to a large number, effectively no limit
        root_symbol="ROOT",
        init_grammar_path=None,
        generate_n=5000,
        generation_seed=G.BASE_SEED,
        generation_max_depth=200,
        generation_max_attempts_per_sentence=20000,
        generation_max_total_attempts=100000000,
        mean_len=G.MEAN_LEN,
        stdev_len=G.STDEV_LEN,
        checkpoint_sample_sentences=100,
        checkpoint_sample_max_attempts_per_sentence=2000,
        checkpoint_sample_max_total_attempts=None,
        prune_every_accepted_trees=1000,
        progress_log_every_trees=100,
        generation_progress_every_sentences=10000,
        protected_count_floor=1e-12,
    )
