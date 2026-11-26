# Beyond Tree Accuracy: Grammar-Level Evaluation of Syntactic Representations in LLMs

Code and grammar resources for the paper **Beyond Tree Accuracy: Grammar-Level Evaluation of Syntactic Representations in LLMs**, accepted to the Main Conference of EACL 2027.
## Overview

This repository provides the implementation of our framework for evaluating syntactic information in language-model representations at the grammar level.

We train decoder-only Transformers on controlled, grammar-generated languages and use structural probes to decode constituency trees from their hidden representations. By inducing grammars from these trees, we investigate whether the recovered structures capture systematic syntactic regularities beyond individual parse-tree accuracy.


The experimental pipeline consists of three stages:

**A. Grammar Construction:** Construct a source PCFG and extract deterministic LL(k) subgrammars.

**B. Language-Model Training:** Train decoder-only Transformers from scratch on grammar-generated sentences.

**C. Probing and Grammar-Level Evaluation:** Decode constituency trees, induce grammars, and evaluate grammar recovery, source-grammar compatibility, and recognition behavior.

The repository includes demonstration entry points (main_*) with reduced settings for local experimentation. These are not intended to reproduce the full-scale experiments directly. To reproduce the paper's experiments, use the configurable scripts with the corresponding hyperparameters and dataset sizes.

The resources/grammars/ directory contains the source grammar and LL(1), LL(2), and LL(3) grammars used in the paper.

## Repository Structure

| Path | Purpose |
| --- | --- |
| `A_PCFG/` | Source PCFG construction, lexical anonymization, treebank sampling, LL(k) subgrammar extraction, and LL(k) parsing utilities. |
| `B_LLMTraining/` | GPT-2-style decoder-only language-model training from scratch on generated grammar data. |
| `C_Probing_and_Evaluation/` | Atomic constituency probing, tree decoding, grammar induction, grammar comparison, plotting, and recognition-behavior evaluation. |
| `resources/grammars/` | Grammar files used in the experiments: `grammar_super.txt` for `G_S`, and `grammar_ll1.txt`, `grammar_ll2.txt`, `grammar_ll3.txt` for `G_1`, `G_2`, `G_3`. |
| `configs/gpt2/` | GPT-2-small-style architecture configuration used for training from scratch. |

## Installation

Recommended Python version: **3.10 or 3.11**.

Install dependencies:

```bash
pip install -r requirements.txt
```

To reconstruct the source grammar, download the NLTK Brown corpus:

```bash
python -c "import nltk; nltk.download('brown')"
```

To reconstruct the source grammar from Brown Corpus parses, also download the NLTK Brown corpus:

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





## Citation

If you use this code or the released grammar resources in your research, please cite our paper:

```bibtex
@inproceedings{victor2027beyond,
  title     = {Beyond Tree Accuracy: Grammar-Level Evaluation of Syntactic Representations in LLMs},
  author    = {Victor, Luisa and Frey, Markus and Horv{\'a}th, Tam{\'a}s},
  booktitle = {Proceedings of the 2027 Conference of the European Chapter of the Association for Computational Linguistics},
  year      = {2027}
}
```
The official ACL Anthology citation will be added once the proceedings are published.
