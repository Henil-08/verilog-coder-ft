"""Decide which held-out test rows the diff-sim harness can score fairly.

OpenRTLSet: a row is eligible when
  1. the reference passes against itself (harness and design are deterministic), and
  2. the reference fails against a stub that ties every output to 0, so random
     stimulus can actually tell a wrong design from the right one.

VerilogEval: a problem is valid when its reference, dropped in as TopModule, passes
its own testbench under the official harness.

Usage:
    uv run python -m eval.prepare   # writes test_eligibility.jsonl, verilog_eval_invalid.json
"""

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from eval.diffsim import diff_sim
from eval.ports import PortError, extract_ports
from eval.verilog_eval import load_problems
from eval.verilog_eval import score as score_verilog_eval


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


def invalid_verilog_eval_problems(workers: int) -> list[str]:
    problems = load_problems()

    def ref_status(p: dict) -> str:
        return score_verilog_eval(p, p["ref"].replace("module RefModule", "module TopModule"))["status"]

    with ThreadPoolExecutor(workers) as ex:
        statuses = list(ex.map(ref_status, problems))
    return [p["id"] for p, s in zip(problems, statuses) if s != "pass"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test-file", default="data/processed/test.jsonl")
    parser.add_argument("--out", default="data/processed/test_eligibility.jsonl")
    parser.add_argument("--ve-out", default="data/processed/verilog_eval_invalid.json")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--skip-openrtl", action="store_true")
    args = parser.parse_args()

    invalid = invalid_verilog_eval_problems(args.workers)
    Path(args.ve_out).write_text(json.dumps(invalid, indent=2))
    print(f"verilog_eval: {len(invalid)} problems fail with their own reference: {invalid}")
    if args.skip_openrtl:
        return

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
