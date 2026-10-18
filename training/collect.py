"""Copy run summaries and generations from the Modal volume into outputs/.

Talks to the volume directly (no app, no upload), so it is quick to rerun while jobs
are still going. Uses the active Modal profile (or $MODAL_PROFILE).

    uv run python -m training.collect --runs grid:3B,7B-L3
"""

import argparse
import json
from pathlib import Path

import modal

from training.config import list_runs

ROOT = Path(__file__).resolve().parents[1]


def expand_runs(runs: str) -> list[str]:
    names = []
    for item in runs.split(","):
        names += list_runs(grid_size=item.removeprefix("grid:")) if item.startswith("grid:") else [item]
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--volume", default="verilog-ft-runs")
    args = parser.parse_args()

    volume = modal.Volume.from_name(args.volume)

    def names(path: str) -> set[str]:
        try:
            return {Path(e.path).name for e in volume.listdir(path)}
        except Exception:  # directory doesn't exist yet
            return set()

    generations = names("generations")
    for name in expand_runs(args.runs):
        status = []
        if "summary.json" in names(f"runs/{name}"):
            summary = json.loads(b"".join(volume.read_file(f"runs/{name}/summary.json")))
            local = ROOT / "outputs" / "runs" / name / "summary.json"
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_text(json.dumps(summary | {"modal_profile": modal.config._profile}, indent=2))
            status.append("summary ok")
        else:
            status.append("summary pending")
        if f"{name}.jsonl" in generations:
            local = ROOT / "outputs" / "generations" / f"{name}.jsonl"
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(b"".join(volume.read_file(f"generations/{name}.jsonl")))
            status.append("generations ok")
        else:
            status.append("generations pending")
        print(f"{name:<9} " + ", ".join(status))


if __name__ == "__main__":
    main()
