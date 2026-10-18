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


SPLITS = {"openrtl": "test", "openrtl_val": "val"}
KEEP = ("model", "benchmark", "id", "output_tokens", "finish_reason", "code")


def load_context(processed_dir: str | Path = "data/processed") -> dict:
    """Everything needed to score any item: references, eligibility, VerilogEval flags."""
    proc = Path(processed_dir)
    rows, eligible = {}, {}
    for bench, split in SPLITS.items():
        if (proc / f"{split}_eligibility.jsonl").exists():
            rows[bench] = {r["id"]: r for r in load_jsonl(proc / f"{split}.jsonl")}
            eligible[bench] = {r["id"]: r["eligible"] for r in load_jsonl(proc / f"{split}_eligibility.jsonl")}
    return {
        "rows": rows,
        "eligible": eligible,
        "problems": {p["id"]: p for p in load_problems()},
        # Problems whose reference fails its own testbench under the official harness.
        "invalid": set(json.loads((proc / "verilog_eval_invalid.json").read_text())),
        "flagged": {f["problem"] for f in json.load(open(proc / "contamination.json"))["flagged"]},
    }


def score_item(gen: dict, ctx: dict) -> dict:
    gen = gen | {"code": extract_code(gen["response"])}
    bench = gen["benchmark"]
    if bench in SPLITS:
        scored = score_openrtl(gen, ctx["rows"][bench][gen["id"]], ctx["eligible"][bench][gen["id"]])
    else:
        pid = gen["id"]
        scored = score_ve(gen, ctx["problems"][pid], pid not in ctx["invalid"], pid not in ctx["flagged"])
    return {k: gen[k] for k in KEEP} | scored


def pending(generation_files: list[str], out_dir: str | Path, force: bool = False) -> dict[str, tuple[list, list]]:
    """Per model: (already-scored results, generations still to score)."""
    by_model: dict[str, list[dict]] = defaultdict(list)
    for path in generation_files:
        for gen in load_jsonl(path):
            by_model[gen["model"]].append(gen)
    out = {}
    for model, gens in by_model.items():
        out_path = Path(out_dir) / f"{model}.jsonl"
        previous = [] if force or not out_path.exists() else load_jsonl(out_path)
        done = {(r["benchmark"], r["id"]) for r in previous}
        out[model] = (previous, [g for g in gens if (g["benchmark"], g["id"]) not in done])
    return out


def write_results(model: str, results: list[dict], out_dir: str | Path) -> None:
    Path(out_dir).mkdir(exist_ok=True)
    Path(out_dir, f"{model}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in results))
    line = f"{model:<32}"
    for bench in ("openrtl", "openrtl_val", "verilog_eval", "verilog_eval_iface"):
        scored = [r for r in results if r["benchmark"] == bench and r.get("eligible", True)]
        if scored:
            line += f"   {bench} {sum(bool(r['passed']) for r in scored)}/{len(scored)}"
    print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("generations", nargs="+")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--force", action="store_true", help="rescore items that already have results")
    args = parser.parse_args()

    ctx = load_context(args.processed_dir)
    for model, (previous, todo) in pending(args.generations, args.out_dir, args.force).items():
        if not todo:
            print(f"skip {model} (all {len(previous)} items scored)")
            continue
        with ThreadPoolExecutor(args.workers) as ex:
            results = previous + list(ex.map(lambda g: score_item(g, ctx), todo))
        write_results(model, results, args.out_dir)


if __name__ == "__main__":
    main()
