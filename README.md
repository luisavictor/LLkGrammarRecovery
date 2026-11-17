# Beyond Tree Accuracy: Grammar-Level Evaluation of Syntactic Representations in LLMs

This repository contains the code for **Beyond Tree Accuracy: Grammar-Level Evaluation of Syntactic Representations in LLMs**.

## Overview

The project evaluates whether decoder-only language models trained on controlled grammar-generated languages learn syntactic regularities that can be recovered as grammars, not only as individual decoded parse trees.

The pipeline has three stages:

**A: Grammar construction**

**B: Language-model training**

**C: Probing and grammar-level evaluation**

The small `main_*` files are demonstration entry points showing how to call the code locally. They use small/debug-oriented settings. For full-scale runs, use the paper settings and pass them to the configurable scripts described below. 
The pipeline should be run in order (first A, then B, then C).

The `resources/grammars/` directory provides the original LL(k) grammars and the broader source grammar used for the experiments in the paper.


## Repository Structure

| Path | Purpose |
| --- | --- |
| `A_PCFG/` | Source PCFG construction, lexical anonymization, treebank sampling, LL(k) subgrammar extraction, and LL(k) parsing utilities. |
| `B_LLMTraining/` | GPT-2-style decoder-only language-model training from scratch on generated grammar data. |
| `C_Probing_and_Evaluation/` | Atomic constituency probing, tree decoding, grammar induction, grammar comparison, plotting, and recognition-behavior evaluation. |
| `resources/grammars/` | Grammar files used in the experiments: `grammar_super.txt` for `G_S`, and `grammar_ll1.txt`, `grammar_ll2.txt`, `grammar_ll3.txt` for `G_1`, `G_2`, `G_3`. |
| `configs/gpt2/` | GPT-2-small-style architecture configuration used for training from scratch. |

## Installation

Recommended Python version:

```text
Python 3.10 or 3.11
```

Install dependencies with:

```bash
pip install -r requirements.txt
```

To reconstruct the source grammar from Brown Corpus parses, also download the NLTK Brown corpus:

```python
import nltk
nltk.download("brown")
```

Stanza English constituency resources are downloaded by the grammar-construction code when the Stanza pipeline is built.
## Workflow

### A. Grammar Construction

Block A constructs the source PCFG, anonymizes lexical items, samples sentence/tree pairs, and extracts LL(k) subgrammars used for training and evaluation.

Demonstration call:

```bash
python -m A_PCFG.main_grammar_generation
```

Configurable entry point:

```bash
python -m A_PCFG.llk_grammar_from_ambiguous_trees --help
```

For full-scale runs, choose the paper values for `k` (k=1,2,3), sample sizes (10M training sentences, 10K test sentences), seeds, and the LL(3) left-hand-side rule cap (cap = 25).

Released grammar resources:

```text
resources/grammars/grammar_super.txt
resources/grammars/grammar_ll1.txt
resources/grammars/grammar_ll2.txt
resources/grammars/grammar_ll3.txt
```

### B. Language-Model Training

Block B trains a decoder-only Transformer language model from scratch on sentences sampled from an LL(k) grammar. The training code uses a word-level tokenizer so grammar terminals remain atomic model tokens.

Demonstration call:

```bash
python -m B_LLMTraining.main_training
```

For full-scale runs, pass the paper train size (10M), test size (10K), random seeds and output paths explicitly.

### C. Probing and Evaluation

Block C trains token-level linear probes on model hidden states, decodes predicted constituency trees, induces grammars from the decoded trees, and compares those grammars to gold evaluation grammars and the broader source grammar.

Demonstration call:

```bash
python -m C_Probing_and_Evaluation.main_probing_evaluation
```

Configurable entry points:

```bash
python -m C_Probing_and_Evaluation.grammar_induction_and_comparison --help
python -m C_Probing_and_Evaluation.plot_grammar_comparison_results --help
python -m C_Probing_and_Evaluation.recognition_behavior --help
python -m C_Probing_and_Evaluation.vilares_constituency_probe --help
```

For full-scale runs, set the number of probe training trees to 1K to 50K and test trees to 10K.
## Outputs

Typical generated outputs are:

| Stage | Example output |
| --- | --- |
| A | `A_PCFG/data/grammar_mle.txt`, `A_PCFG/data/data_trees.txt`, `A_PCFG/data/ll1/grammar.txt`, `A_PCFG/data/ll1/data_trees.txt` |
| B | `B_LLMTraining/runs/seed_*/...`, `B_LLMTraining/splits/seed_*/...` |
| C grammar comparison | `C_Probing_and_Evaluation/results_grammar_comparison/` |
| C recognition behavior | `C_Probing_and_Evaluation/results_acceptance_matrix/` |
