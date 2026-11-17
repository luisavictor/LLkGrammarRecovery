"""Decoder-only language-model training implementation.

This module contains the reusable training code for block B. It trains a
GPT-2-style causal language model from scratch with a tokenizer learned from the
training split, packs text into fixed-length LM blocks, configures Hugging Face
`Trainer`, and writes checkpoints for later probing.
"""

import os
import math
import inspect
from pathlib import Path
from typing import List
import torch
import torch.distributed as dist
try:
    import wandb
except ModuleNotFoundError:
    wandb = None
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE, WordLevel
from tokenizers.normalizers import NFC
from tokenizers.pre_tokenizers import ByteLevel, Whitespace
from tokenizers.trainers import BpeTrainer, WordLevelTrainer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    PreTrainedTokenizerFast,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

try:
    from B_LLMTraining.globals import EOS_TOKEN, MAX_LENGTH, PAD_TOKEN, UNK_TOKEN
except ModuleNotFoundError:
    from globals import EOS_TOKEN, MAX_LENGTH, PAD_TOKEN, UNK_TOKEN


def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_initialized() else 0


def is_rank0() -> bool:
    return get_rank() == 0


def env_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def barrier() -> None:
    if is_dist_initialized():
        dist.barrier()


class SaveAtStepsCallback(TrainerCallback):
    def __init__(self, save_steps: List[int]):
        self.save_steps = set(int(step) for step in save_steps if int(step) > 0)

    def on_step_end(self, args, state, control, **kwargs):
        if int(state.global_step) in self.save_steps:
            control.should_save = True
        return control


def infer_layer_class(model_name: str) -> str:
    model_lower = model_name.lower()
    if "llama" in model_lower or "teuken" in model_lower:
        return "LlamaDecoderLayer"
    if "gpt2" in model_lower:
        return "GPT2Block"
    if "mistral" in model_lower:
        return "MistralDecoderLayer"
    if "qwen" in model_lower:
        return "Qwen3Block"
    return "TransformerLayer"


def fsdp_save(trainer: Trainer, tokenizer: PreTrainedTokenizerFast, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    model = trainer.model
    rank = get_rank()
    is_fsdp = isinstance(model, FSDP)

    if is_fsdp:
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            full_state_dict = model.state_dict()
        if rank == 0:
            model.module.save_pretrained(output_dir, state_dict=full_state_dict, safe_serialization=True)
    else:
        if rank == 0:
            model.save_pretrained(output_dir, safe_serialization=True)

    if rank == 0:
        tokenizer.save_pretrained(output_dir)

    barrier()


def save_initial_checkpoint(model: AutoModelForCausalLM, tokenizer: PreTrainedTokenizerFast, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)


def train_tokenizer_from_train_data(
    train_dataset,
    tokenizer_file: Path,
    vocab_size: int = 70000,
    tokenizer_type: str = "wordlevel",
) -> None:
    tokenizer_file.parent.mkdir(parents=True, exist_ok=True)

    def batch_iterator(batch_size: int = 10000):
        n = len(train_dataset)
        for i in range(0, n, batch_size):
            batch = train_dataset[i : i + batch_size]
            yield batch["text"]

    tokenizer_type = str(tokenizer_type).lower()
    if tokenizer_type == "wordlevel":
        tok = Tokenizer(WordLevel(unk_token=UNK_TOKEN))
        tok.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=[PAD_TOKEN, UNK_TOKEN, EOS_TOKEN],
            min_frequency=1,
            vocab_size=vocab_size,
        )
    elif tokenizer_type == "bpe":
        tok = Tokenizer(BPE(unk_token=UNK_TOKEN))
        tok.normalizer = NFC()
        tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tok.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(
            special_tokens=[PAD_TOKEN, UNK_TOKEN, EOS_TOKEN],
            min_frequency=2,
            vocab_size=vocab_size,
            initial_alphabet=ByteLevel.alphabet(),
        )
    else:
        raise ValueError(f"Unsupported tokenizer_type: {tokenizer_type}")

    tok.train_from_iterator(batch_iterator(), trainer=trainer, length=len(train_dataset))
    tok.save(str(tokenizer_file))


def pack_dataset(tokenized_ds, tokenizer: PreTrainedTokenizerFast, block_size: int, drop_remainder: bool):
    eos = tokenizer.eos_token_id
    pad = tokenizer.pad_token_id

    def append_eos(ex):
        ids = ex["input_ids"]
        if len(ids) == 0 or ids[-1] != eos:
            ids = ids + [eos]
        ex["input_ids"] = ids
        return ex

    tokenized_ds = tokenized_ds.map(
        append_eos,
        desc="Append EOS",
        load_from_cache_file=False,
        keep_in_memory=True,
    )

    def group_texts(examples):
        concatenated = []
        for ids in examples["input_ids"]:
            concatenated.extend(ids)

        total_length = len(concatenated)
        if total_length == 0:
            return {"input_ids": [], "attention_mask": [], "labels": []}

        if drop_remainder:
            total_length = (total_length // block_size) * block_size
            if total_length == 0:
                return {"input_ids": [], "attention_mask": [], "labels": []}
            concatenated = concatenated[:total_length]
        else:
            rem = total_length % block_size
            if rem != 0:
                concatenated = concatenated + [pad] * (block_size - rem)

        chunks = [concatenated[i : i + block_size] for i in range(0, len(concatenated), block_size)]
        attention_mask = [[1 if tok != pad else 0 for tok in chunk] for chunk in chunks]
        labels = [chunk.copy() for chunk in chunks]

        for lab, am in zip(labels, attention_mask):
            for j, m in enumerate(am):
                if m == 0:
                    lab[j] = -100

        return {"input_ids": chunks, "attention_mask": attention_mask, "labels": labels}

    keep_cols = ["input_ids"]
    remove_cols = [c for c in tokenized_ds.column_names if c not in keep_cols]

    return tokenized_ds.map(
        group_texts,
        batched=True,
        remove_columns=remove_cols,
        desc=f"Pack to {block_size} (drop={drop_remainder})",
        load_from_cache_file=False,
        keep_in_memory=True,
    )


def build_save_schedule(total_steps: int, target_save_events: int, schedule: str, schedule_power: float) -> List[int]:
    total_steps = max(1, int(total_steps))
    target_save_events = max(1, int(target_save_events))

    if schedule == "uniform":
        save_steps = max(1, total_steps // target_save_events)
        return list(range(save_steps, total_steps + 1, save_steps))

    if schedule != "exp_early":
        raise ValueError(f"Unsupported save schedule: {schedule}")

    save_points = set()
    power = max(1.0, float(schedule_power))
    for idx in range(1, target_save_events + 1):
        frac = idx / target_save_events
        step = max(1, int(round(total_steps * (frac ** power))))
        save_points.add(min(total_steps, step))
    save_points.add(total_steps)
    return sorted(save_points)


def train(
    model_name,
    output_dir,
    train_dataset,
    val_dataset,
    layer_class=None,
    seed=1,
    data_seed=1,
    local_rank=0,
    num_train_epochs=5,
    learning_rate=5e-4,
    warmup_ratio=0.05,
    per_device_bs=8,   #8
    grad_accum=8,     #8
    weight_decay=0.1,
    vocab_size=70000,
    tokenizer_type="wordlevel",
    block_size=None,
    use_fsdp=None,
    checkpoint_schedule="uniform",
    checkpoint_schedule_power=2.0,
    target_save_events=50,
    enable_wandb=False,
):
    set_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    local_rank = int(os.environ.get("LOCAL_RANK", local_rank))
    has_cuda = torch.cuda.is_available()
    if has_cuda:
        torch.cuda.set_device(local_rank)

    output_path = Path(output_dir)
    tokenizer_path = output_path / "tokenizer.json"

    if is_rank0():
        print(f"Training tokenizer: type={tokenizer_type}, vocab_size={vocab_size}")
        train_tokenizer_from_train_data(
            train_dataset,
            tokenizer_path,
            vocab_size=vocab_size,
            tokenizer_type=tokenizer_type,
        )
    barrier()

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_path),
        unk_token=UNK_TOKEN,
        pad_token=PAD_TOKEN,
        eos_token=EOS_TOKEN,
    )

    config = AutoConfig.from_pretrained(model_name)
    config.vocab_size = tokenizer.vocab_size
    config.pad_token_id = tokenizer.pad_token_id
    config.eos_token_id = tokenizer.eos_token_id
    if hasattr(config, "bos_token_id"):
        config.bos_token_id = tokenizer.eos_token_id

    if block_size is None:
        max_positions = getattr(config, "max_position_embeddings", None)
        n_positions = getattr(config, "n_positions", None)
        model_ctx = max_positions or n_positions or MAX_LENGTH
        block_size = min(MAX_LENGTH, int(model_ctx))
    print(f"Using block_size: {block_size}")
    model = AutoModelForCausalLM.from_config(config)
    model.config.use_cache = False

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=block_size, add_special_tokens=False)

    tokenized_train = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized_val = val_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    packed_train = pack_dataset(tokenized_train, tokenizer, block_size=block_size, drop_remainder=True)
    packed_val = pack_dataset(tokenized_val, tokenizer, block_size=block_size, drop_remainder=False)

    if is_rank0():
        print(f"Packed train blocks: {len(packed_train)}")
        print(f"Packed val blocks: {len(packed_val)}")

    world_size = env_world_size()
    steps_per_epoch = math.ceil(len(packed_train) / (per_device_bs * world_size * grad_accum))
    total_steps = max(1, steps_per_epoch * int(num_train_epochs))

    target_eval_events = 500
    eval_steps = max(1, total_steps // target_eval_events)
    save_schedule = build_save_schedule(
        total_steps=total_steps,
        target_save_events=target_save_events,
        schedule=checkpoint_schedule,
        schedule_power=checkpoint_schedule_power,
    )
    use_custom_save_callback = checkpoint_schedule != "uniform"
    save_steps = max(1, total_steps + 1) if use_custom_save_callback else max(1, total_steps // target_save_events)

    if is_rank0():
        print(f"Steps/epoch: {steps_per_epoch}")
        print(f"Total steps: {total_steps}")
        print(f"LR peak: {learning_rate:.2e}")
        print(
            f"Checkpoint schedule: {checkpoint_schedule}"
            f" | requested={target_save_events}"
            f" | actual={len(save_schedule)}"
        )
        print(f"Checkpoint steps: {save_schedule}")

    if use_fsdp is None:
        use_fsdp = has_cuda and world_size > 1

    if layer_class is None:
        layer_class = infer_layer_class(model_name)

    report_to = ["wandb"] if enable_wandb and is_rank0() and wandb is not None else ["none"]

    training_kwargs = dict(
        output_dir=output_dir,
        report_to=report_to,
        run_name=str(output_dir),
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_bs,
        per_device_eval_batch_size=per_device_bs,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        adam_beta1=0.9,
        adam_beta2=0.95,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        gradient_checkpointing=has_cuda,
        max_grad_norm=1.0,
        logging_strategy="steps",
        logging_steps=1,
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=None,
        seed=seed,
        data_seed=data_seed,
        prediction_loss_only=False,
        dataloader_pin_memory=has_cuda,
        bf16=has_cuda and torch.cuda.is_bf16_supported(),
        fp16=has_cuda and not torch.cuda.is_bf16_supported(),
    )
    training_args_params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" not in training_args_params and "evaluation_strategy" in training_args_params:
        training_kwargs["evaluation_strategy"] = training_kwargs.pop("eval_strategy")

    if use_fsdp:
        training_kwargs["fsdp"] = "full_shard auto_wrap"
        training_kwargs["fsdp_config"] = {
            "transformer_layer_cls_to_wrap": layer_class,
            "sync_module_states": True,
            "use_orig_params": True,
            "backward_prefetch": "BACKWARD_PRE",
            "cpu_offload": False,
        }

    training_args = TrainingArguments(**training_kwargs)


    # for fair train and eval loss curve comparison
    train_len = len(packed_train)
    n = max(1, int(0.01 * train_len))
    train_subset = packed_train.shuffle(seed=data_seed).select(range(n))

    if is_rank0():
        save_initial_checkpoint(model, tokenizer, str(output_path / "checkpoint-initial"))
    barrier()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=packed_train,
        eval_dataset={
            "val": packed_val,
            "train_subset": train_subset,
        },
        data_collator=default_data_collator,
    )
    if use_custom_save_callback:
        trainer.add_callback(SaveAtStepsCallback(save_schedule))

    trainer.train()
    fsdp_save(trainer, tokenizer, output_dir)

    try:
        if wandb is not None and wandb.run is not None:
            wandb.finish()
    except Exception:
        pass

    return trainer
