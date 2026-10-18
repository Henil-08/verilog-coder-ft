"""Resolve a named run from configs/experiments.yaml into a flat dict."""

from pathlib import Path

import yaml

EXPERIMENTS_PATH = Path(__file__).resolve().parents[1] / "configs" / "experiments.yaml"

# On-demand Modal prices (USD per GPU-hour), used only for cost estimates in run summaries.
GPU_HOURLY_USD = {
    "B200": 6.25,
    "H200": 4.54,
    "H100": 3.95,
    "A100-80GB": 2.50,
    "A100-40GB": 2.10,
    "L40S": 1.95,
    "A10": 1.10,
    "L4": 0.80,
    "T4": 0.59,
}


def list_runs(path: str | Path = EXPERIMENTS_PATH) -> list[str]:
    with open(path) as f:
        return list(yaml.safe_load(f)["runs"])


def load_run(name: str, path: str | Path = EXPERIMENTS_PATH) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if name not in cfg["runs"]:
        raise KeyError(f"unknown run '{name}'; choose from {list(cfg['runs'])}")
    run = {**cfg["defaults"], **cfg["runs"][name], "run_name": name}
    if run["method"] not in ("lora", "qlora", "full"):
        raise ValueError(f"run '{name}': unknown method '{run['method']}'")
    return run


def estimate_cost(gpu: str, n_gpus: int, seconds: float) -> float:
    return round(GPU_HOURLY_USD.get(gpu, 0.0) * n_gpus * seconds / 3600, 3)
