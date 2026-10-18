"""Evaluation prompts for both benchmarks, in the same chat format used for training."""

import json
import re

from data.prompts import build_messages
from eval.verilog_eval import load_problems


def openrtl_items(split_file: str = "data/processed/test.jsonl", benchmark: str = "openrtl") -> list[dict]:
    items = []
    for line in open(split_file):
        row = json.loads(line)
        items.append({
            "benchmark": benchmark,
            "id": row["id"],
            "messages": build_messages(row["description"], row["ioheader"]),
        })
    return items


_REF_HEADER = re.compile(r"module\s+RefModule\b.*?\);", re.DOTALL)


def interface_header(problem: dict) -> str:
    """The reference's port declaration, renamed to TopModule."""
    return _REF_HEADER.search(problem["ref"]).group(0).replace("RefModule", "TopModule", 1)


def verilog_eval_items(root=None, with_interface: bool = False) -> list[dict]:
    """VerilogEval prompts as published, or with the interface also given as a code block.

    The published prompts describe the TopModule interface in prose. Training prompts
    always carried a `Module interface:` block, so the `_iface` variant restates the
    same ports in that form (taken from the reference header; some headers say
    `output reg`, a mild implementation hint).
    """
    problems = load_problems(root) if root else load_problems()
    benchmark = "verilog_eval_iface" if with_interface else "verilog_eval"
    return [
        {
            "benchmark": benchmark,
            "id": p["id"],
            "messages": build_messages(p["prompt"], interface_header(p) if with_interface else None),
        }
        for p in problems
    ]


# openrtl_val (validation split) is used only to choose between fine-tuned runs.
BENCHMARKS = ("openrtl", "openrtl_val", "verilog_eval", "verilog_eval_iface")


def all_items(benchmarks=BENCHMARKS) -> list[dict]:
    items = []
    if "openrtl" in benchmarks:
        items += openrtl_items()
    if "openrtl_val" in benchmarks:
        items += openrtl_items("data/processed/val.jsonl", "openrtl_val")
    if "verilog_eval" in benchmarks:
        items += verilog_eval_items()
    if "verilog_eval_iface" in benchmarks:
        items += verilog_eval_items(with_interface=True)
    return items
