"""Score generations on Modal CPU containers instead of locally.

Same logic as eval.score (same per-item function), fanned out over many containers.
The image builds the simulator versions used to prepare the test sets: Verilator
5.052 for diff-sim and Icarus Verilog v12 for VerilogEval.

    uv run modal run eval/modal_score.py --files "outputs/generations/*.jsonl"
"""

import glob
import os
import sys
from collections import defaultdict
from pathlib import Path

import modal

PROJECT_DIR = Path(__file__).resolve().parents[1]
REMOTE_DIR = "/root/project"
VERILATOR_TAG = "v5.052"
WORKERS_PER_CONTAINER = 8

app = modal.App("verilog-coder-ft-score")

image = (
    modal.Image.from_registry("ubuntu:24.04", add_python="3.11")
    .apt_install(
        "ca-certificates", "python3", "git", "make", "autoconf", "g++", "flex", "bison", "help2man", "perl", "gperf",
        "libfl2", "libfl-dev", "zlib1g", "zlib1g-dev", "ccache",
    )
    .run_commands(
        f"git clone --depth 1 --branch {VERILATOR_TAG} https://github.com/verilator/verilator /tmp/verilator",
        "cd /tmp/verilator && autoconf && ./configure && make -j$(nproc) && make install && rm -rf /tmp/verilator",
    )
    .add_local_file(PROJECT_DIR / "scripts" / "install_iverilog12.sh", "/tmp/install_iverilog12.sh", copy=True)
    .run_commands("PREFIX=/opt/iverilog-12 bash /tmp/install_iverilog12.sh")
    .uv_sync(str(PROJECT_DIR))
    .env({"IVERILOG_BIN_DIR": "/opt/iverilog-12/bin"})
    .add_local_dir(PROJECT_DIR / "data" / "processed", f"{REMOTE_DIR}/data/processed",
                   ignore=["train_*.jsonl"])
    .add_local_dir(PROJECT_DIR / "data" / "raw" / "verilog-eval" / "dataset_spec-to-rtl",
                   f"{REMOTE_DIR}/data/raw/verilog-eval/dataset_spec-to-rtl")
    .add_local_python_source("data", "eval", "training")
)

_ctx = None


@app.function(image=image, cpu=WORKERS_PER_CONTAINER, memory=8192, timeout=3600)
def score_batch(gens: list[dict]) -> list[dict]:
    from concurrent.futures import ThreadPoolExecutor

    from eval.score import load_context, score_item

    global _ctx
    os.chdir(REMOTE_DIR)
    if _ctx is None:
        _ctx = load_context()
    with ThreadPoolExecutor(WORKERS_PER_CONTAINER) as ex:
        return list(ex.map(lambda g: score_item(g, _ctx), gens))


@app.local_entrypoint()
def main(files: str, out_dir: str = "results", force: bool = False, batch_size: int = 16):
    sys.path.insert(0, str(PROJECT_DIR))
    from eval.score import pending, write_results

    paths = sorted({p for pattern in files.split(",") for p in glob.glob(pattern)})
    work = pending(paths, out_dir, force)
    todo = [g for _, (_, gens) in work.items() for g in gens]
    print(f"{len(todo)} items to score across {sum(1 for _, (_, g) in work.items() if g)} models")
    if not todo:
        return

    batches = [todo[i : i + batch_size] for i in range(0, len(todo), batch_size)]
    scored: dict[str, list[dict]] = defaultdict(list)
    for results in score_batch.map(batches):
        for r in results:
            scored[r["model"]].append(r)

    for model, (previous, _) in work.items():
        if scored.get(model):
            write_results(model, previous + scored[model], out_dir)
