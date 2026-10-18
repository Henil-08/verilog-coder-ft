"""Check whether VerilogEval reference solutions leak into the training data.

VerilogEval problems come from HDLBits, whose solutions are common on GitHub.
Candidates are training rows named `top_module` (HDLBits' convention) or whose port
names match the problem's; each candidate is scored by token-set Jaccard against the
reference. This catches copies and light edits, not paraphrased re-implementations.

    uv run python -m eval.contamination
"""

import argparse
import json
import re

from data.filters import strip_comments
from eval.ports import PortError, extract_ports
from eval.verilog_eval import load_problems

_TOKEN = re.compile(r"\w+|[^\w\s]")
_IDENT = re.compile(r"\b[A-Za-z_]\w*\b")


def tokens(code: str) -> set[str]:
    return set(_TOKEN.findall(strip_comments(code)))


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-file", default="data/processed/train_full.jsonl")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--out", default="data/processed/contamination.json")
    args = parser.parse_args()

    train = [json.loads(line) for line in open(args.train_file)]
    header_idents = [set(_IDENT.findall(r["ioheader"])) for r in train]

    report = []
    for problem in load_problems():
        try:
            port_names = {p.name for p in extract_ports(problem["ref"], "RefModule")}
        except PortError:
            port_names = set()
        candidates = [
            r for r, idents in zip(train, header_idents)
            if r["module"] == "top_module" or (port_names and port_names <= idents)
        ]
        ref_tokens = tokens(problem["ref"].replace("RefModule", "top_module"))
        best = max(((jaccard(ref_tokens, tokens(r["code"])), r["id"]) for r in candidates), default=(0.0, None))
        report.append({"problem": problem["id"], "candidates": len(candidates),
                       "best_jaccard": round(best[0], 3), "best_train_id": best[1]})

    flagged = [r for r in report if r["best_jaccard"] >= args.threshold]
    json.dump({"threshold": args.threshold, "flagged": flagged, "problems": report}, open(args.out, "w"), indent=2)
    print(f"{len(flagged)}/{len(report)} problems have a training row with token Jaccard >= {args.threshold}")
    for r in sorted(report, key=lambda r: -r["best_jaccard"])[:5]:
        print(f"  {r['problem']:<32} best={r['best_jaccard']:.3f} candidates={r['candidates']}")


if __name__ == "__main__":
    main()
