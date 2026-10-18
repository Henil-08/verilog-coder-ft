"""Aggregate per-example results into the base-vs-fine-tuned comparison table.

    uv run python -m eval.report --baseline base-Qwen2.5-Coder-1.5B-Instruct

Writes results/summary.json and results/summary.md. Each pass@1 comes with a 95%
Wilson interval, and every model is compared with the baseline on the same items
using an exact McNemar test (two-sided), since all models see identical prompts.
"""

import argparse
import json
import math
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant pair counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


SUBSETS = {
    "openrtl": lambda r: r["benchmark"] == "openrtl" and r["eligible"],
    "openrtl_val": lambda r: r["benchmark"] == "openrtl_val" and r["eligible"],
    "openrtl_verilog": lambda r: r["benchmark"] == "openrtl" and r["eligible"] and r["category"] == "verilog",
    "openrtl_systemverilog": lambda r: r["benchmark"] == "openrtl" and r["eligible"] and r["category"] == "systemverilog",
    "verilog_eval": lambda r: r["benchmark"] == "verilog_eval",
    "verilog_eval_valid": lambda r: r["benchmark"] == "verilog_eval" and r["valid_problem"],
    "verilog_eval_clean": lambda r: r["benchmark"] == "verilog_eval" and r["valid_problem"] and r["clean"],
    "verilog_eval_iface": lambda r: r["benchmark"] == "verilog_eval_iface",
    "verilog_eval_iface_clean": lambda r: r["benchmark"] == "verilog_eval_iface" and r["valid_problem"] and r["clean"],
}
COMPILE_SUBSETS = {
    "openrtl_compile": lambda r: r["benchmark"] == "openrtl",
    "verilog_eval_compile": lambda r: r["benchmark"] == "verilog_eval",
    "verilog_eval_iface_compile": lambda r: r["benchmark"] == "verilog_eval_iface",
}


def summarize(results: list[dict], baseline: list[dict] | None) -> dict:
    out = {}
    base_by_key = {(r["benchmark"], r["id"]): r for r in baseline} if baseline else {}
    for name, keep in SUBSETS.items():
        rows = [r for r in results if keep(r)]
        k = sum(bool(r["passed"]) for r in rows)
        lo, hi = wilson(k, len(rows))
        entry = {"passed": k, "n": len(rows), "pass@1": round(k / len(rows), 4) if rows else None,
                 "ci95": [round(lo, 4), round(hi, 4)]}
        if base_by_key:
            pairs = [(bool(base_by_key[(r["benchmark"], r["id"])]["passed"]), bool(r["passed"])) for r in rows]
            gained = sum(1 for b, m in pairs if m and not b)
            lost = sum(1 for b, m in pairs if b and not m)
            entry |= {"gained_vs_baseline": gained, "lost_vs_baseline": lost,
                      "mcnemar_p": round(mcnemar_exact(gained, lost), 4)}
        out[name] = entry
    for name, keep in COMPILE_SUBSETS.items():
        rows = [r for r in results if keep(r)]
        out[name] = {"compiles": sum(r["compiles"] for r in rows), "n": len(rows),
                     "rate": round(sum(r["compiles"] for r in rows) / len(rows), 4) if rows else None}
    truncated = sum(r["finish_reason"] == "length" for r in results)
    out["truncated_outputs"] = truncated
    return out


def pct(entry: dict) -> str:
    return f"{100 * entry['pass@1']:.1f}" if entry["pass@1"] is not None else "-"


def to_markdown(summary: dict, baseline: str, training: dict) -> str:
    lines = [
        "| Model | OpenRTLSet pass@1 | VerilogEval pass@1 | VerilogEval + interface | Compile (ORT / VE / VE+if) "
        "| vs base, ORT: +/- (p) | vs base, VE+if: +/- (p) | Train cost |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for model, s in summary.items():
        o, v, vi = s["openrtl"], s["verilog_eval"], s["verilog_eval_iface"]

        def delta(e: dict) -> str:
            if model == baseline or "gained_vs_baseline" not in e:
                return "baseline"
            return f"+{e['gained_vs_baseline']}/-{e['lost_vs_baseline']} (p={e['mcnemar_p']})"

        def rate(name: str) -> str:
            r = s[name]["rate"]
            return f"{100 * r:.0f}%" if r is not None else "-"

        cost = training.get(model, {}).get("container_cost_usd")
        lines.append(
            f"| {model} | {pct(o)} ({o['passed']}/{o['n']}) | {pct(v)} ({v['passed']}/{v['n']}) | "
            f"{pct(vi)} ({vi['passed']}/{vi['n']}) | "
            f"{rate('openrtl_compile')} / {rate('verilog_eval_compile')} / {rate('verilog_eval_iface_compile')} | "
            f"{delta(o)} | {delta(vi)} | {f'${cost:.2f}' if cost else '-'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--runs-dir", default="outputs/runs", help="training summaries (optional)")
    parser.add_argument("--baseline", required=True, help="model tag to compare against")
    parser.add_argument("--models", default="", help="comma-separated tags in table order (default: all)")
    parser.add_argument("--name", default="summary", help="output file stem")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    files = {p.stem: p for p in results_dir.glob("*.jsonl")}
    order = args.models.split(",") if args.models else sorted(files)
    load = lambda tag: [json.loads(line) for line in open(files[tag])]
    baseline = load(args.baseline)

    summary = {tag: summarize(load(tag), None if tag == args.baseline else baseline) for tag in order}
    training = {}
    for tag in order:
        path = Path(args.runs_dir) / tag / "summary.json"
        if path.exists():
            training[tag] = json.loads(path.read_text())

    (results_dir / f"{args.name}.json").write_text(json.dumps(summary, indent=2))
    md = to_markdown(summary, args.baseline, training)
    (results_dir / f"{args.name}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
