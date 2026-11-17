"""Demonstration entry point for running the language-model training code locally.

The values exposed here are convenience/debug defaults showing how to call the
training implementation. For full-scale arguments, use the reported
experimental settings or the argparse defaults in this file's CLI.
"""

import argparse
import random
from pathlib import Path
import torch
import json
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
try:
    from B_LLMTraining import globals as G
    from B_LLMTraining.metrics import evaluate_llm, log_metrics_to_csv
    from B_LLMTraining.training import barrier, is_rank0, train
except ModuleNotFoundError:
    import globals as G
    from metrics import evaluate_llm, log_metrics_to_csv
    from training import barrier, is_rank0, train


def build_split(examples, seed: int, train_size: int, val_ratio: float, unique_train_sentences: bool = False):
    rng = random.Random(seed)
    shuffled = examples[:]
    rng.shuffle(shuffled)

    val_size = max(1, int(train_size * val_ratio))

    if unique_train_sentences:
        sentence_to_example = {}
        ordered_unique_examples = []
        for example in shuffled:
            sentence = example["sentence"]
            if sentence in sentence_to_example:
                continue
            sentence_to_example[sentence] = example
            ordered_unique_examples.append(example)

        required_examples = G.TEST_SIZE + train_size + val_size
        if len(ordered_unique_examples) < required_examples:
            raise ValueError(
                "Not enough unique examples for the requested split: "
                f"need {required_examples}, found {len(ordered_unique_examples)}. "
                "Lower --split_size, --val_ratio, or G.TEST_SIZE."
            )

        test_examples = ordered_unique_examples[: G.TEST_SIZE]
        remaining_unique = ordered_unique_examples[G.TEST_SIZE :]

        train_examples = remaining_unique[:train_size]
        val_examples = remaining_unique[train_size : train_size + val_size]
    else:
        required_examples = G.TEST_SIZE + train_size + val_size
        if len(shuffled) < required_examples:
            raise ValueError(
                "Not enough examples for the requested split: "
                f"need {required_examples}, found {len(shuffled)}. "
                "Lower --split_size, --val_ratio, or G.TEST_SIZE."
            )

        test_examples = shuffled[: G.TEST_SIZE]
        remaining = shuffled[G.TEST_SIZE :]
        rng_local = random.Random(seed)
        remaining_pool = remaining[:]
        rng_local.shuffle(remaining_pool)
        subset = remaining_pool[: train_size + val_size]
        train_examples = subset[:train_size]
        val_examples = subset[train_size : train_size + val_size]

    train_dataset = Dataset.from_dict({"text": [example["sentence"] for example in train_examples]})
    val_dataset = Dataset.from_dict({"text": [example["sentence"] for example in val_examples]})
    return train_dataset, val_dataset, test_examples, train_examples, val_examples


def repo_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(G.REPO_ROOT))
    except ValueError:
        return str(path)


def repo_path(path_value: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return G.REPO_ROOT / path


def model_run_label(model_name: str) -> str:
    model_path = repo_path(model_name)
    if model_path.exists():
        return model_path.name
    return model_name.replace("/", "_").replace("\\", "_")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=str(G.DEFAULT_MODEL_NAME))
    parser.add_argument("--output_dir", type=str, default=str(G.TRAINING_OUTPUT_DIR))
    parser.add_argument("--data_trees_file", type=str, default=str(G.DATA_TREES_FILE))
    parser.add_argument("--seed_output_dir", type=str, default=str(G.SPLIT_OUTPUT_DIR))

    parser.add_argument("--data_seed", type=int, default=1)
    parser.add_argument("--model_seed", type=int, default=1)

    parser.add_argument("--split_name", type=str, default="debug_run_data")
    parser.add_argument("--split_size", type=int, default=1000)  # just for debug
    parser.add_argument("--val_ratio", type=float, default=0.01)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--vocab_size", type=int, default=70000)
    parser.add_argument("--tokenizer_type", type=str, default="wordlevel", choices=["bpe", "wordlevel"])
    parser.add_argument("--unique_train_sentences", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--eval_after_train", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable_wandb", action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args(argv)
    model_seed = args.data_seed if args.model_seed is None else args.model_seed

    data_path = repo_path(args.data_trees_file)
    examples = []
    for line in data_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        examples.append({"sentence": row["sentence"], "tree": row["tree"]})

    train_dataset, val_dataset, test_examples, train_examples, val_examples = build_split(
        examples=examples,
        seed=args.data_seed,
        train_size=args.split_size,
        val_ratio=args.val_ratio,
        unique_train_sentences=args.unique_train_sentences,
    )

    seed_output_root = repo_path(args.seed_output_dir)
    seed_dir = seed_output_root / f"seed_{args.data_seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    Path(seed_dir, "test.txt").write_text("\n".join(example["sentence"] for example in test_examples), encoding="utf-8")
    Path(seed_dir, "test_trees.txt").write_text("\n".join(example["tree"] for example in test_examples), encoding="utf-8")
    Path(seed_dir, f"train_{args.split_name}.txt").write_text("\n".join(train_dataset["text"]), encoding="utf-8")
    Path(seed_dir, f"train_{args.split_name}_trees.txt").write_text(
        "\n".join(example["tree"] for example in train_examples),
        encoding="utf-8",
    )
    Path(seed_dir, f"valid_{args.split_name}.txt").write_text("\n".join(val_dataset["text"]), encoding="utf-8")
    Path(seed_dir, f"valid_{args.split_name}_trees.txt").write_text(
        "\n".join(example["tree"] for example in val_examples),
        encoding="utf-8",
    )

    model_name = str(repo_path(args.model_name)) if repo_path(args.model_name).exists() else args.model_name
    run_dir = repo_path(args.output_dir) / f"seed_{model_seed}" / f"{model_run_label(args.model_name)}_{args.split_name}"

    if is_rank0():
        print("data_trees_file:", repo_relative(data_path))
        print("data_seed:", args.data_seed)
        print("model_seed:", model_seed)
        print("len of train_dataset:", len(train_dataset))
        print("len of val_dataset:", len(val_dataset))
        print("seed_output_dir:", repo_relative(seed_dir))
        print("output_dir:", repo_relative(run_dir))

    train(
        model_name=model_name,
        output_dir=str(run_dir),
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        seed=model_seed,
        data_seed=args.data_seed,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        vocab_size=args.vocab_size,
        tokenizer_type=args.tokenizer_type,
        block_size=args.block_size,
        enable_wandb=args.enable_wandb,
    )

    barrier()

    if args.eval_after_train and is_rank0():
        model = AutoModelForCausalLM.from_pretrained(
            str(run_dir),
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype="auto",
            use_safetensors=True,
            local_files_only=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(str(run_dir), local_files_only=True)

        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        test_file = Path(seed_dir, "test.txt")
        lines = test_file.read_text(encoding="utf-8").splitlines()
        dataset = Dataset.from_list([{"text": line.strip()} for line in lines if line.strip()])

        metrics = evaluate_llm(dataset, tokenizer, model, device, batch_size=128, max_length=1024)
        print(metrics)
        log_metrics_to_csv(metrics, "perplexity.csv", run_name=str(run_dir))

    barrier()


if __name__ == "__main__":
    main()
