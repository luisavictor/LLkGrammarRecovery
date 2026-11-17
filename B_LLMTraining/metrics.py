"""Evaluation helpers for trained language models."""

import torch
import math
import os
import csv


def log_metrics_to_csv(metrics, filepath, run_name=None):
    if run_name is not None:
        metrics = {"run_name": run_name, **metrics}
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)


def evaluate_llm(dataset, tokenizer, model, device, batch_size, max_length):
    model.eval()
    model.to(device)
    if max_length < 2:
        raise ValueError("max_length must be >= 2 for causal perplexity.")
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = eos_id if eos_id is not None else 0
    ce = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    total_nll = 0.0
    total_tokens = 0
    token_buffer = []

    def process_chunks(chunks):
        nonlocal total_nll, total_tokens
        if not chunks:
            return

        input_ids = torch.tensor(chunks, dtype=torch.long, device=device)
        attention_mask = (input_ids != pad_id).long()
        labels = input_ids.masked_fill(attention_mask == 0, -100)

        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        loss_sum = ce(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        tok_count = int((shift_labels != -100).sum().item())

        total_nll += float(loss_sum.item())
        total_tokens += tok_count

    print(f"Dataset size: {len(dataset)}")
    chunk_batch = []

    for i in range(0, len(dataset), batch_size):
        batch = dataset[i:i + batch_size]
        texts = [t for t in batch["text"] if t]
        if not texts:
            continue

        enc = tokenizer(
            texts,
            add_special_tokens=False,
            truncation=True,
            max_length=max_length - 1 if eos_id is not None else max_length,
        )
        for ids in enc["input_ids"]:
            if eos_id is not None and (len(ids) == 0 or ids[-1] != eos_id):
                ids = ids + [eos_id]
            token_buffer.extend(ids)

            while len(token_buffer) >= max_length:
                chunk_batch.append(token_buffer[:max_length])
                del token_buffer[:max_length]
                if len(chunk_batch) >= batch_size:
                    process_chunks(chunk_batch)
                    chunk_batch = []

    if token_buffer:
        last = token_buffer + [pad_id] * (max_length - len(token_buffer))
        chunk_batch.append(last)
    process_chunks(chunk_batch)

    if total_tokens == 0:
        raise ValueError("No valid tokens found for perplexity computation.")

    metrics = {
        "perplexity": math.exp(total_nll / total_tokens),
    }

    print(f"Total tokens (incl. EOS): {total_tokens}")
    print(f"Total NLL: {total_nll}")
    print(f"Perplexity: {metrics['perplexity']:.4f}")

    return metrics
