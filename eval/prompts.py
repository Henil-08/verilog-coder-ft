"""Evaluation prompts for both benchmarks, in the same chat format used for training."""

import json

from data.prompts import build_messages
from eval.verilog_eval import load_problems


def openrtl_items(test_file: str = "data/processed/test.jsonl") -> list[dict]:
    items = []
    for line in open(test_file):
        row = json.loads(line)
        items.append({
            "benchmark": "openrtl",
            "id": row["id"],
            "messages": build_messages(row["description"], row["ioheader"]),
        })
    return items


def verilog_eval_items(root=None) -> list[dict]:
    problems = load_problems(root) if root else load_problems()
    # The VerilogEval prompt already states the TopModule interface.
    return [
        {"benchmark": "verilog_eval", "id": p["id"], "messages": build_messages(p["prompt"])}
        for p in problems
    ]


def all_items() -> list[dict]:
    return openrtl_items() + verilog_eval_items()
