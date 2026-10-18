"""Launch training runs on Modal GPUs.

    uv run modal run training/modal_app.py --runs L2 --max-steps 20          # pilot
    uv run modal run training/modal_app.py --runs L1,L2,L3,Q1,Q2,Q3,F1,F2,F3  # ablation grid, in parallel
    uv run modal run training/modal_app.py --runs F2 --n-gpus 2               # override GPU count

Outputs land on the `verilog-ft-runs` volume:
    uv run modal volume get verilog-ft-runs runs/L2/summary.json .
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import modal

PROJECT_DIR = Path(__file__).resolve().parents[1]
REMOTE_DIR = "/root/project"
RUNS_DIR = "/vol/runs"

app = modal.App("verilog-coder-ft-train")
runs_volume = modal.Volume.from_name("verilog-ft-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("verilog-ft-hf-cache", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.11")
    .entrypoint([])
    .uv_sync(str(PROJECT_DIR), extras=["train"])
    .env({"HF_HOME": "/cache/hf", "TOKENIZERS_PARALLELISM": "false"})
    .add_local_dir(
        PROJECT_DIR,
        REMOTE_DIR,
        ignore=[".venv", ".git", "outputs", "data/raw", "**/__pycache__", "data/processed/train_full.jsonl"],
    )
)


@app.function(
    image=image,
    gpu="H100",
    timeout=4 * 3600,
    volumes={"/vol": runs_volume, "/cache": hf_cache},
    secrets=[modal.Secret.from_name("wandb-secret"), modal.Secret.from_name("huggingface-secret")],
)
def train(run: str, gpu: str, n_gpus: int, max_steps: int = -1) -> dict:
    sys.path.insert(0, REMOTE_DIR)
    from training.config import estimate_cost

    start = time.time()
    cmd = [
        "torchrun", "--standalone", f"--nproc_per_node={n_gpus}",
        "-m", "training.train",
        "--experiment", run,
        "--output-root", RUNS_DIR,
        "--max-steps", str(max_steps),
    ]
    proc = subprocess.run(cmd, cwd=REMOTE_DIR)
    hf_cache.commit()
    if proc.returncode != 0:
        runs_volume.commit()
        raise RuntimeError(f"run {run} exited with code {proc.returncode}")

    # Container time covers model download and tokenization too, so it's the honest cost.
    summary_path = Path(RUNS_DIR) / run / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["container_seconds"] = round(time.time() - start, 1)
    summary["container_cost_usd"] = estimate_cost(gpu, n_gpus, summary["container_seconds"])
    summary_path.write_text(json.dumps(summary, indent=2))
    runs_volume.commit()
    return summary


@app.local_entrypoint()
def main(runs: str, max_steps: int = -1, gpu: str = "", n_gpus: int = 0):
    from training.config import load_run

    calls = []
    for name in runs.split(","):
        cfg = load_run(name)
        run_gpu, run_n = gpu or cfg["gpu"], n_gpus or cfg["n_gpus"]
        spec = run_gpu if run_n == 1 else f"{run_gpu}:{run_n}"
        print(f"launching {name} on {spec}" + (f" (max_steps={max_steps})" if max_steps > 0 else ""))
        calls.append((name, train.with_options(gpu=spec).spawn(name, run_gpu, run_n, max_steps)))

    total = 0.0
    print(f"\n{'run':<6} {'method':<6} {'steps':>5} {'tok/s':>9} {'mem GB':>7} {'eval loss':>9} {'cost $':>7}")
    for name, call in calls:
        try:
            s = call.get()
        except Exception as exc:  # keep collecting the other runs
            print(f"{name:<6} FAILED: {exc}")
            continue
        total += s["container_cost_usd"]
        print(
            f"{name:<6} {s['method']:<6} {s['steps']:>5} {s['tokens_per_second']:>9,.0f} "
            f"{s['peak_gpu_mem_gb']:>7} {s['final_eval_loss']:>9} {s['container_cost_usd']:>7.2f}"
        )
    print(f"\ntotal GPU cost (est., excludes CPU/RAM): ${total:.2f}")
