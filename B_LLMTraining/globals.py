"""Shared paths and lightweight defaults for local training runs.

These values are intentionally conservative/debug-oriented so the repository can
be exercised on small generated datasets. They are not the full-scale experiment
sizes: the paper trains language models on 10M sampled sentences and evaluates
on held-out sets of 10K sentences. For reproduction runs, override these values through
the training CLI or explicit config.
"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

A_PCFG_OUTPUT_DIR = Path("A_PCFG") / "data"
DEFAULT_LL_DIR = A_PCFG_OUTPUT_DIR / "ll1"

OUTPUT_DIR = A_PCFG_OUTPUT_DIR
SENTENCES = DEFAULT_LL_DIR / "data.txt"
DATA_TREES_FILE = DEFAULT_LL_DIR / "data_trees.txt"
GRAMMAR_FILE = A_PCFG_OUTPUT_DIR / "grammar.txt"
VOCAB_FILE = A_PCFG_OUTPUT_DIR / "vocab.txt"

DEFAULT_MODEL_NAME = Path("configs") / "gpt2"
TRAINING_OUTPUT_DIR = Path("B_LLMTraining") / "runs"
SPLIT_OUTPUT_DIR = Path("B_LLMTraining") / "splits"

VOCAB_SIZE = 65536
WORD_LENGTH = 5

TEST_SIZE = 100  # Debug/default split size, not the paper held-out size (10K).

# Debug/demo training sizes. Full-scale training uses 10M train sentences.
TRAIN_SIZES = {
    "100": 100,
    "1K": 1000,
}

# used for Gaussian sampling/rejection criterion
MEAN_LEN = 20.25
STDEV_LEN = 13.11

EOS_TOKEN = "[EOS]"
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"

MAX_LENGTH = 1024

BASE_SEED = 1
