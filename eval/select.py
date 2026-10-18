"""Pick the best fine-tuned run per model size on the validation split.

Selection uses only `openrtl_val` (diff-sim pass@1 on held-out validation modules),
so the test benchmarks stay untouched by the choice. Ties go to the cheaper run.

    uv run python -m eval.select --sizes 1.5B,3B,7B
"""

import argparse
import json
from pathlib import Path

from training.config import list_runs


def val_score(results_path: Path) -> tuple[int, int] | None:
    if not results_path.exists():
        return None
    rows = [json.loads(line) for line in open(results_path)]
    val = [r for r in rows if r["benchmark"] == "openrtl_val" and r["eligible"]]
    return (sum(bool(r["passed"]) for r in val), len(val)) if val else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", default="1.5B,3B,7B")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--runs-dir", default="outputs/runs")
    parser.add_argument("--out", default="results/selection.json")
    args = parser.parse_args()

    selection = {}
    for size in args.sizes.split(","):
        candidates = []
        for run in list_runs(grid_size=size):
            score = val_score(Path(args.results_dir) / f"{run}.jsonl")
            if score is None:
                continue
            summary = Path(args.runs_dir) / run / "summary.json"
            cost = json.loads(summary.read_text()).get("container_cost_usd", 0.0) if summary.exists() else 0.0
            candidates.append({"run": run, "val_passed": score[0], "val_n": score[1], "cost_usd": cost})
        if not candidates:
            print(f"{size}: no scored runs yet")
            continue
        candidates.sort(key=lambda c: (-c["val_passed"], c["cost_usd"]))
        selection[size] = {"best": candidates[0]["run"], "candidates": candidates}
        missing = 9 - len(candidates)
        print(f"{size}: best {candidates[0]['run']} ({candidates[0]['val_passed']}/{candidates[0]['val_n']} val)"
              + (f"  [{missing} runs not scored yet]" if missing else ""))

    Path(args.out).write_text(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
