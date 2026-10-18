"""Launch training runs on Modal GPUs.

    uv run modal run training/modal_app.py::main --runs L2 --max-steps 20          # pilot
    uv run modal run training/modal_app.py::main --runs L1,L2,L3,Q1,Q2,Q3,F1,F2,F3  # ablation grid, in parallel
    uv run modal run training/modal_app.py::main --runs F2 --n-gpus 2               # override GPU count

    # long jobs: detach so they survive the local client disconnecting, then collect
    uv run modal deploy eval/modal_generate.py                                 # once per workspace
    uv run modal run --detach training/modal_app.py::main --runs grid:3B --evaluate --budget 28
    uv run modal run training/modal_app.py::collect --runs grid:3B

--budget refuses to launch if this month's metered spend plus the estimated cost of
the queued runs would exceed it. --evaluate makes each training container hand its
model to the deployed generation app when it finishes, so evaluation needs no local
process either. `collect` copies run summaries and generations from the volume to
outputs/; adapters and weights stay on the `verilog-ft-runs` volume:
    uv run modal volume get verilog-ft-runs runs/L2/model ./L2-adapter
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
def train(run: str, gpu: str, n_gpus: int, max_steps: int = -1, evaluate: bool = False) -> dict:
    sys.path.insert(0, REMOTE_DIR)
    from training.config import estimate_cost, load_run

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
    runs_volume.commit()

    if evaluate:
        # Hand off to the deployed generation app (modal deploy eval/modal_generate.py);
        # it runs independently of this container and of the local client.
        cfg = load_run(run)
        model_dir = f"{RUNS_DIR}/{run}/model"
        model, adapter = (model_dir, None) if cfg["method"] == "full" else (cfg["base_model"], model_dir)
        items = [json.loads(line) for line in open(Path(REMOTE_DIR) / "data/processed/eval_items.jsonl")]
        generate = modal.Function.from_name("verilog-coder-ft-generate", "generate")
        call = generate.spawn(model, [{"tag": run, "adapter": adapter}], items)
        summary["generation_call_id"] = call.object_id
    summary_path.write_text(json.dumps(summary, indent=2))
    runs_volume.commit()
    return summary


def expand_runs(runs: str) -> list[str]:
    from training.config import list_runs

    names = []
    for item in runs.split(","):
        names += list_runs(grid_size=item.removeprefix("grid:")) if item.startswith("grid:") else [item]
    return names


@app.local_entrypoint()
def main(runs: str, max_steps: int = -1, gpu: str = "", n_gpus: int = 0, evaluate: bool = False,
         budget: float = 0.0):
    from training.budget import estimate_eval_cost, estimate_run_cost, metered_cost
    from training.config import load_run

    names = expand_runs(runs)
    configs = {name: load_run(name) for name in names}
    for cfg in configs.values():
        cfg["gpu"], cfg["n_gpus"] = gpu or cfg["gpu"], n_gpus or cfg["n_gpus"]

    if budget and max_steps < 0:
        estimates = {n: estimate_run_cost(c) + (estimate_eval_cost(c["base_model"]) if evaluate else 0.0)
                     for n, c in configs.items()}
        spent = metered_cost()
        if spent is None:
            raise SystemExit("could not read `modal billing summary`; rerun without --budget to skip the check")
        planned = sum(estimates.values())
        print(f"spent this month ${spent:.2f} + planned ${planned:.2f} (est.) vs budget ${budget:.2f}")
        if spent + planned > budget:
            room, fits = budget - spent, []
            for n, cost in estimates.items():
                if cost <= room:
                    fits.append(n)
                    room -= cost
            raise SystemExit(f"over budget; runs that fit: {','.join(fits) or 'none'}")

    calls = []
    for name, cfg in configs.items():
        spec = cfg["gpu"] if cfg["n_gpus"] == 1 else f"{cfg['gpu']}:{cfg['n_gpus']}"
        print(f"launching {name} on {spec}" + (f" (max_steps={max_steps})" if max_steps > 0 else ""))
        calls.append((name, train.with_options(gpu=spec).spawn(name, cfg["gpu"], cfg["n_gpus"], max_steps, evaluate)))

    total, done = 0.0, []
    print(f"\n{'run':<9} {'method':<6} {'steps':>5} {'tok/s':>9} {'mem GB':>7} {'eval loss':>9} {'cost $':>7}")
    for name, call in calls:
        try:
            s = call.get()
        except Exception as exc:  # keep collecting the other runs
            print(f"{name:<9} FAILED: {exc}")
            continue
        total += s["container_cost_usd"]
        done.append(name)
        local = PROJECT_DIR / "outputs" / "runs" / name
        local.mkdir(parents=True, exist_ok=True)
        (local / "summary.json").write_text(json.dumps(s | {"modal_profile": modal.config._profile}, indent=2))
        print(
            f"{name:<9} {s['method']:<6} {s['steps']:>5} {s['tokens_per_second']:>9,.0f} "
            f"{s['peak_gpu_mem_gb']:>7} {s['final_eval_loss']:>9} {s['container_cost_usd']:>7.2f}"
        )
    print(f"\ntotal GPU cost (est., excludes CPU/RAM): ${total:.2f}")

    if evaluate and done:
        print("generation was handed to the deployed app; fetch everything later with "
              f"`modal run training/modal_app.py::collect --runs {','.join(done)}`")


@app.local_entrypoint()
def collect(runs: str):
    """Copy summaries and generations for `runs` from the volume into outputs/."""
    for name in expand_runs(runs):
        files = {f"runs/{name}/summary.json": PROJECT_DIR / "outputs" / "runs" / name / "summary.json",
                 f"generations/{name}.jsonl": PROJECT_DIR / "outputs" / "generations" / f"{name}.jsonl"}
        status = []
        for remote, local in files.items():
            try:
                data = b"".join(runs_volume.read_file(remote))
            except Exception:  # not written yet (still running) or the run failed
                status.append(f"{Path(remote).name}: missing")
                continue
            local.parent.mkdir(parents=True, exist_ok=True)
            if local.suffix == ".json":
                data = json.dumps(json.loads(data) | {"modal_profile": modal.config._profile}, indent=2).encode()
            local.write_bytes(data)
            status.append(f"{Path(remote).name}: ok")
        print(f"{name:<9} " + ", ".join(status))
