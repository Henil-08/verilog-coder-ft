"""Filter OpenRTLSet and write held-out test, validation, and training splits.

Usage:
    uv run python -m data.split [--config configs/data.yaml] [--no-lint]
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from data.filters import CATEGORIES, categorize, dedup_key, is_standalone, module_name
from data.load import CONFIG_PATH, load_config, load_raw
from data.prompts import build_training_messages
from eval.verilator import lint, verilator_available

COLUMNS = ["id", "category", "repo", "module", "ioheader", "description", "code", "n_tokens"]


def filter_rows(df: pd.DataFrame, cfg: dict, stats: dict) -> pd.DataFrame:
    f = cfg["filters"]
    stats["raw"] = len(df)

    df = df[~df["repo"].isin(f["drop_repos"])]
    stats["after_drop_repos"] = len(df)

    df = df[(df["code"].str.len() >= f["min_code_chars"]) & (df["description"].str.len() > 0)]
    stats["after_min_length"] = len(df)

    df = df.assign(dedup=df["code"].map(dedup_key)).drop_duplicates("dedup").drop(columns="dedup")
    stats["after_dedup"] = len(df)
    return df


def count_tokens(df: pd.DataFrame, tokenizer_name: str, batch_size: int = 1000) -> list[int]:
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    counts = []
    rows = df.to_dict("records")
    for i in range(0, len(rows), batch_size):
        texts = [
            tok.apply_chat_template(build_training_messages(r), tokenize=False)
            for r in rows[i : i + batch_size]
        ]
        counts.extend(len(ids) for ids in tok(texts, add_special_tokens=False)["input_ids"])
    return counts


def test_quotas(counts: dict[str, int], test_size: int, min_per_category: int) -> dict[str, int]:
    """Proportional allocation with a floor for small categories; the largest absorbs the rest."""
    total = sum(counts.values())
    largest = max(counts, key=counts.get)
    quotas = {}
    for cat, n in counts.items():
        if cat == largest:
            continue
        proportional = round(test_size * n / total)
        quotas[cat] = max(proportional, min(min_per_category, n // 2))
    quotas[largest] = test_size - sum(quotas.values())
    return quotas


def _passes_lint(row: dict) -> bool:
    return lint(row["code"], top=row["module"], systemverilog=row["category"] != "verilog").ok


def pick_test(pool: pd.DataFrame, quota: int, require_lint: bool, seed: int) -> pd.DataFrame:
    """Sample `quota` standalone rows, linting candidates in parallel until the quota is met."""
    candidates = pool[pool["standalone"] & pool["module"].notna()].sample(frac=1, random_state=seed)
    if not require_lint:
        return candidates.head(quota)

    picked = []
    chunk = max(quota * 2, 32)
    with ThreadPoolExecutor(os.cpu_count()) as ex:
        for start in range(0, len(candidates), chunk):
            batch = candidates.iloc[start : start + chunk]
            ok = list(ex.map(_passes_lint, batch.to_dict("records")))
            picked.append(batch[ok])
            if sum(len(p) for p in picked) >= quota:
                break
    return pd.concat(picked).head(quota) if picked else candidates.head(0)


def write_jsonl(df: pd.DataFrame, path: Path) -> None:
    with open(path, "w") as f:
        for row in df[COLUMNS].to_dict("records"):
            f.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=CONFIG_PATH)
    parser.add_argument("--no-lint", action="store_true", help="skip Verilator lint on test candidates")
    args = parser.parse_args()

    cfg = load_config(args.config)
    split_cfg = cfg["split"]
    seed = split_cfg["seed"]
    require_lint = split_cfg["test_require_lint"] and not args.no_lint
    if require_lint and not verilator_available():
        raise SystemExit("verilator not found on PATH (install it or pass --no-lint)")

    stats: dict = {}
    df = filter_rows(load_raw(cfg).to_pandas(), cfg, stats)

    df["n_tokens"] = count_tokens(df, cfg["tokenizer"])
    df = df[df["n_tokens"] <= cfg["filters"]["max_tokens"]].copy()
    stats["after_max_tokens"] = len(df)

    df["category"] = df["code"].map(categorize)
    df["module"] = df["ioheader"].map(module_name)
    df["standalone"] = [is_standalone(c, h) for c, h in zip(df["code"], df["has_children"])]
    counts = {c: int((df["category"] == c).sum()) for c in CATEGORIES if (df["category"] == c).any()}
    stats["category_counts"] = counts

    quotas = test_quotas(counts, split_cfg["test_size"], split_cfg["min_per_category"])
    test_parts = []
    for cat, quota in quotas.items():
        # UVM code depends on the UVM library, which a bare Verilator lint can't resolve.
        part = pick_test(df[df["category"] == cat], quota, require_lint and cat != "uvm", seed)
        if len(part) < quota:
            print(f"warning: only {len(part)}/{quota} eligible test rows for '{cat}'")
        test_parts.append(part)
    test = pd.concat(test_parts)

    rest = df.drop(test.index).sample(frac=1, random_state=seed)
    val = rest.head(split_cfg["val_size"])
    train = rest.iloc[split_cfg["val_size"] :]

    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    splits = {"test": test, "val": val, "train_full": train}
    for name, size in split_cfg["train_subsets"].items():
        if size > len(train):
            print(f"warning: {name} wants {size} rows, only {len(train)} available")
        splits[name] = train.head(size)

    stats["splits"] = {}
    for name, part in splits.items():
        write_jsonl(part, out / f"{name}.jsonl")
        stats["splits"][name] = {
            "rows": len(part),
            "tokens": int(part["n_tokens"].sum()),
            "by_category": part["category"].value_counts().to_dict(),
        }

    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
