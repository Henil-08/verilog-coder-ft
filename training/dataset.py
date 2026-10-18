"""Tokenize SFT splits with loss applied only to the assistant (code) turn."""

from datasets import Dataset, load_dataset

from data.prompts import build_training_messages

IGNORE_INDEX = -100


def tokenize_example(row: dict, tokenizer, max_seq_len: int) -> dict:
    messages = build_training_messages(row)
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


def load_sft_dataset(path: str, tokenizer, max_seq_len: int, num_proc: int = 4) -> Dataset:
    ds = load_dataset("json", data_files=path, split="train")
    ds = ds.map(
        tokenize_example,
        fn_kwargs={"tokenizer": tokenizer, "max_seq_len": max_seq_len},
        remove_columns=ds.column_names,
        num_proc=num_proc,
        desc=f"tokenizing {path}",
    )
    # A prompt longer than max_seq_len leaves no supervised tokens and a NaN loss.
    return ds.filter(lambda ex: any(label != IGNORE_INDEX for label in ex["labels"]), num_proc=num_proc)
