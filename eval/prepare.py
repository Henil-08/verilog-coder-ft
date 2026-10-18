"""Decide which held-out test rows the diff-sim harness can score fairly.

A row is eligible when
  1. the reference passes against itself (harness and design are deterministic), and
  2. the reference fails against a stub that ties every output to 0, so random
     stimulus can actually tell a wrong design from the right one.

Usage:
    uv run python -m eval.prepare   # writes data/processed/test_eligibility.jsonl
"""

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from eval.diffsim import diff_sim
from eval.ports import PortError, extract_ports


def zero_stub(code: str, top: str) -> str | None:
    try:
        ports = extract_ports(code, top)
    except PortError:
        return None
    decls, assigns = [], []
    for p in ports:
        rng = f"[{p.width - 1}:0] " if p.width > 1 else ""
        decls.append(f"  {p.direction} {'logic ' if p.direction == 'output' else ''}{rng}{p.name}")
        if p.direction == "output":
            assigns.append(f"  assign {p.name} = '0;")
    return f"module {top} (\n" + ",\n".join(decls) + "\n);\n" + "\n".join(assigns) + "\nendmodule\n"


def check_row(row: dict) -> dict:
    top, ref = row["module"], row["code"]
    self_check = diff_sim(ref, ref, top)
    result = {"id": row["id"], "category": row["category"], "self_check": self_check.status}
    if not self_check.passed:
        result.update(eligible=False, reason=f"self_check:{self_check.status}", log=self_check.log[-500:])
        return result

    stub = zero_stub(ref, top)
    stub_check = diff_sim(ref, stub, top) if stub else None
    result["zero_stub_check"] = stub_check.status if stub_check else "no_stub"
    discriminative = stub_check is not None and stub_check.status == "mismatch"
    result.update(eligible=discriminative, reason=None if discriminative else "not_discriminative")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-file", default="data/processed/test.jsonl")
    parser.add_argument("--out", default="data/processed/test_eligibility.jsonl")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    args = parser.parse_args()

    rows = [json.loads(line) for line in open(args.test_file)]
    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(check_row, rows))

    Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in results))
    eligible = [r for r in results if r["eligible"]]
    print(f"eligible: {len(eligible)}/{len(results)}")
    print("by category:", dict(Counter(r["category"] for r in eligible)))
    print("excluded reasons:", dict(Counter(r["reason"] for r in results if not r["eligible"])))


if __name__ == "__main__":
    main()
