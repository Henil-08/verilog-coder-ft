"""Score generated code on both benchmarks and write per-example results.

    uv run python -m eval.score outputs/generations/*.jsonl

Generation files for the same model (e.g. L1.jsonl and L1.iface.jsonl) are merged.
Code is re-extracted from the raw response, so extraction fixes apply without
regenerating. For each model this writes results/<model>.jsonl, one row per item:
  - compiles:  generated module passes `verilator --lint-only` (OpenRTLSet) or
               compiles under iverilog with the VerilogEval testbench
  - status:    diff-sim / testbench outcome (see eval.diffsim.STATUSES)
  - passed:    functional pass (counted toward pass@1)
  - eligible:  OpenRTLSet rows only; ineligible rows are excluded from pass@1
"""

import argparse
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from data.prompts import extract_code
from eval.diffsim import diff_sim
from eval.verilator import lint
from eval.verilog_eval import load_problems
from eval.verilog_eval import score as score_verilog_eval

LOG_CHARS = 400


def load_jsonl(path) -> list[dict]:
    return [json.loads(line) for line in open(path)]


def score_openrtl(gen: dict, row: dict, eligible: bool) -> dict:
    compiles = lint(gen["code"], top=row["module"]).ok
    result = {"category": row["category"], "eligible": eligible, "compiles": compiles}
    if not eligible:
        return result | {"status": "ineligible", "passed": None}
    sim = diff_sim(row["code"], gen["code"], row["module"])
    return result | {
        "status": sim.status,
        "passed": sim.passed,
        "mismatches": sim.mismatches,
        "samples": sim.samples,
        "first_mismatch": sim.first_mismatch,
        "log": sim.log[-LOG_CHARS:] or None,
    }


def score_ve(gen: dict, problem: dict, valid: bool, clean: bool) -> dict:
    res = score_verilog_eval(problem, gen["code"])
    return {
        "category": gen["benchmark"],
        "valid_problem": valid,
        "clean": clean,
        "compiles": res["status"] not in ("compile_error",),
        "status": res["status"],
        "passed": res["status"] == "pass",
        "mismatches": res.get("mismatches"),
        "samples": res.get("samples"),
        "log": res.get("log", "")[-LOG_CHARS:] or None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("generations", nargs="+")
    parser.add_argument("--test-file", default="data/processed/test.jsonl")
    parser.add_argument("--eligibility", default="data/processed/test_eligibility.jsonl")
    parser.add_argument("--ve-invalid", default="data/processed/verilog_eval_invalid.json")
    parser.add_argument("--contamination", default="data/processed/contamination.json")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--force", action="store_true", help="rescore even if results exist")
    args = parser.parse_args()

    rows = {r["id"]: r for r in load_jsonl(args.test_file)}
    eligible = {r["id"]: r["eligible"] for r in load_jsonl(args.eligibility)}
    problems = {p["id"]: p for p in load_problems()}
    # Problems whose reference fails its own testbench under the official harness.
    invalid = set(json.loads(Path(args.ve_invalid).read_text()))
    flagged = {f["problem"] for f in json.load(open(args.contamination))["flagged"]}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    by_model: dict[str, list[dict]] = defaultdict(list)
    for path in args.generations:
        for gen in load_jsonl(path):
            by_model[gen["model"]].append(gen | {"code": extract_code(gen["response"])})

    for model, gens in by_model.items():
        out_path = out_dir / f"{model}.jsonl"
        if out_path.exists() and not args.force:
            print(f"skip {model} (exists)")
            continue

        def work(gen: dict) -> dict:
            if gen["benchmark"] == "openrtl":
                scored = score_openrtl(gen, rows[gen["id"]], eligible[gen["id"]])
            else:
                pid = gen["id"]
                scored = score_ve(gen, problems[pid], pid not in invalid, pid not in flagged)
            keep = ("model", "benchmark", "id", "output_tokens", "finish_reason", "code")
            return {k: gen[k] for k in keep} | scored

        with ThreadPoolExecutor(args.workers) as ex:
            results = list(ex.map(work, gens))
        out_path.write_text("".join(json.dumps(r) + "\n" for r in results))
        ort = [r for r in results if r["benchmark"] == "openrtl" and r["eligible"]]
        line = f"{model:<32} openrtl {sum(r['passed'] for r in ort)}/{len(ort)}"
        for bench in ("verilog_eval", "verilog_eval_iface"):
            ve = [r for r in results if r["benchmark"] == bench]
            if ve:
                line += f"   {bench} {sum(r['passed'] for r in ve)}/{len(ve)}"
        print(line)


if __name__ == "__main__":
    main()
