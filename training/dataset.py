"""Tokenize SFT splits with loss applied only to the assistant (code) turn."""

import hashlib
import re

from datasets import Dataset, load_dataset

from data.prompts import build_training_messages, interface_prose
from eval.ports import PortError, extract_ports

IGNORE_INDEX = -100
_NO_PROSE = re.compile(r"#\s*\(|\bparameter\b|\bsigned\b")


def prose_for(row: dict) -> str | None:
    """Prose port list for `row`, or None when prose would lose information.

    Parameterized and signed headers keep the code block: a flat port list can't
    express parameters or signedness.
    """
    if _NO_PROSE.search(row["ioheader"]):
        return None
    try:
        ports = extract_ports(row["ioheader"] + "\nendmodule\n", row["module"])
    except PortError:
        return None
    if not ports or any(p.unpacked or p.direction == "inout" for p in ports):
        return None
    return interface_prose(row["module"], [(p.direction, p.name, p.width) for p in ports])


def use_prose(row_id: str, fraction: float, seed: int) -> bool:
    """Deterministic per-row coin flip, so a run always sees the same mix."""
    if fraction <= 0:
        return False
    bucket = int(hashlib.md5(f"{seed}:{row_id}".encode()).hexdigest(), 16) % 10_000
    return bucket < fraction * 10_000


def tokenize_example(row: dict, tokenizer, max_seq_len: int, prose_fraction: float = 0.0, seed: int = 0) -> dict:
    prose = prose_for(row) if prose_fraction > 0 and use_prose(row["id"], prose_fraction, seed) else None
    messages = build_training_messages(row, prose)
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=False)
    if not full.startswith(prompt):
        raise ValueError("chat template prompt is not a prefix of the full conversation")

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"][:max_seq_len]
    labels = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    labels = labels[: len(full_ids)]
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "length": len(full_ids),
    }


def load_sft_dataset(
    path: str, tokenizer, max_seq_len: int, prose_fraction: float = 0.0, seed: int = 0, num_proc: int = 4
) -> Dataset:
    ds = load_dataset("json", data_files=path, split="train")
    ds = ds.map(
        tokenize_example,
        fn_kwargs={"tokenizer": tokenizer, "max_seq_len": max_seq_len, "prose_fraction": prose_fraction, "seed": seed},
        remove_columns=ds.column_names,
        num_proc=num_proc,
        desc=f"tokenizing {path}",
    )
    # A prompt longer than max_seq_len leaves no supervised tokens and a NaN loss.
    return ds.filter(lambda ex: any(label != IGNORE_INDEX for label in ex["labels"]), num_proc=num_proc)
