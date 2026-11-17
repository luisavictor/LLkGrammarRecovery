"""Demonstration entry point for running probing/evaluation locally.

The values in this file are intentionally small/debug-oriented examples of how
to call the grammar-comparison, plotting, and recognition-behavior code. For
full-scale arguments, use the reported experimental settings or the argparse
defaults in the underlying long modules.
"""

from argparse import Namespace
from pathlib import Path
from typing import Optional, Tuple
from C_Probing_and_Evaluation.grammar_induction_and_comparison import print_summary, run_one, run_size_sweep
from C_Probing_and_Evaluation.plot_grammar_comparison_results import (
    collect_records,
    infer_x_axis,
    make_plots,
    output_dir_with_mode_suffix,
    parse_thresholds,
    validate_data_seed_consistency,
    write_aggregate_csv,
    write_best_metrics_csv,
    write_paper_results_csv,
    x_value_for_record,
)
from C_Probing_and_Evaluation.recognition_behavior import run_with_args as run_acceptance_matrix_with_args


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "C_Probing_and_Evaluation" / "results_grammar_comparison"
ACCEPTANCE_RESULTS_DIR = REPO_ROOT / "C_Probing_and_Evaluation" / "results_acceptance_matrix"


def one_matching_file(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one file matching {pattern!r} in {directory}, found {len(matches)}."
        )
    return matches[0]


def build_args(*, run_baseline: bool) -> Namespace:
    model_path = REPO_ROOT / "B_LLMTraining" / "runs" / "seed_1" / "gpt2_debug_run_data"
    split_dir = REPO_ROOT / "B_LLMTraining" / "splits" / "seed_1"

    return Namespace(
        run_mode="individual",
        layer=12,
        train_trees=100,
        dev_trees=50,
        fixed_test_size=500,
        train_tree_sizes="100,500",
        dev_tree_sizes=None,
        eval_tree_sizes=None,
        batch_size=16,
        extract_batch_size=8,
        epochs=20,
        patience=3,
        lr=5e-4,
        weight_decay=1e-4,
        rule_support_thresholds="1,2,3,5,10,20,50",
        seed=1,
        device="auto",
        local_files_only=True,
        model_path=str(model_path),
        train_trees_path=str(one_matching_file(split_dir, "train_*_trees.txt")),
        dev_trees_path=str(one_matching_file(split_dir, "valid_*_trees.txt")),
        test_trees_path=str(split_dir / "test_trees.txt"),
        supergrammar_path=str(REPO_ROOT / "A_PCFG" / "data" / "grammar_mle.txt"),
        output_dir=str(RESULTS_DIR),
        model_init_seed=1,
        run_baseline=run_baseline,
    )


def run_plots() -> None:
    results_root = RESULTS_DIR
    thresholds = parse_thresholds("1, 5,10")
    records = collect_records([(results_root, None)], default_train_size=None, final_step=None)
    if not records:
        raise RuntimeError(f"No supergrammar_check.json files found under {results_root}.")

    x_axis = infer_x_axis(records, "auto")
    for record in records:
        record["x_axis"] = x_axis
        value = x_value_for_record(record, x_axis)
        record["x_value"] = "" if value is None else value

    validate_data_seed_consistency(records, x_axis, allow_mixed_data_seeds=False)

    output_dir = output_dir_with_mode_suffix(results_root / "plots", records, x_axis)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_paper_results_csv(records, thresholds, output_dir / "paper_results.csv")
    write_aggregate_csv(records, thresholds, output_dir / "supergrammar_plot_data.csv")
    write_best_metrics_csv(records, thresholds, output_dir / "supergrammar_compact_metrics.csv")
    make_plots(records, output_dir, thresholds, x_axis)
    print(f"plot_output_dir={output_dir}")


def train_size_key(path: Path) -> Tuple[int, float]:
    train_size = -1
    for part in path.parts:
        if part.startswith("train_"):
            try:
                train_size = max(train_size, int(part.split("_", 2)[1]))
            except (IndexError, ValueError):
                pass
    return train_size, path.stat().st_mtime


def trained_model_grammar_pair() -> Optional[Tuple[Path, Path]]:
    candidates = sorted((RESULTS_DIR / "trained_model").glob("**/final_layer_12/test_model.grammar.txt"))
    if not candidates:
        candidates = sorted(
            (RESULTS_DIR / "trained_model").glob("**/final_layer_12/test_model_channel.grammar.txt")
        )
    if not candidates:
        return None

    induced = max(candidates, key=train_size_key)
    gold = induced.with_name("test_gold_original.grammar.txt")
    if not gold.exists():
        raise RuntimeError(f"Missing gold acceptance grammar next to {induced}: {gold}")
    return induced, gold


def run_acceptance_matrix() -> None:
    grammar_pair = trained_model_grammar_pair()
    if grammar_pair is None:
        print(
            "acceptance_matrix_skipped="
            f"no trained-model grammar found under {RESULTS_DIR / 'trained_model'}"
        )
        return

    model_path = REPO_ROOT / "B_LLMTraining" / "runs" / "seed_1" / "gpt2_debug_run_data"
    source_grammar = REPO_ROOT / "A_PCFG" / "data" / "grammar_mle.txt"
    induced, gold = grammar_pair
    ACCEPTANCE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    run_acceptance_matrix_with_args(
        Namespace(
            k_values=(1,),
            seeds=[1],
            k1_induced_grammars=[induced],
            k1_gold_grammars=[gold],
            k1_lm_models=[model_path],
            k1_lm_tokenizers=[model_path],
            lm_sample_size=100,
            lm_max_new_tokens=64,
            lm_batch_size=16,
            source_grammars=[source_grammar],
            source_sample_size=100,
            source_max_depth=200,
            source_max_attempts_per_sentence=2_000,
            parser="dummy-earley",
            start_symbol=None,
            max_sentences=None,
            max_parser_steps=100_000,
            max_parser_seconds=None,
            output_json=ACCEPTANCE_RESULTS_DIR / "acceptance_matrix.json",
            acceptance_csv=ACCEPTANCE_RESULTS_DIR / "acceptance_rates.csv",
            jaccard_csv=ACCEPTANCE_RESULTS_DIR / "jaccard_similarity.csv",
        )
    )


def main() -> None:
    for run_baseline in (False, True):
        args = build_args(run_baseline=run_baseline)
        if args.train_tree_sizes or args.dev_tree_sizes or args.eval_tree_sizes:
            summary = run_size_sweep(args)
        else:
            summary = run_one(args)
        print_summary(summary)
    run_plots()
    run_acceptance_matrix()


if __name__ == "__main__":
    main()
