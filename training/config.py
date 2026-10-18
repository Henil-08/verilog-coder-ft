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


def _grid_names(cfg: dict) -> list[str]:
    grid = cfg.get("grid", {})
    return [f"{size}-{method}" for size in grid.get("sizes", {}) for method in grid.get("methods", [])]


def list_runs(path: str | Path = EXPERIMENTS_PATH, grid_size: str | None = None) -> list[str]:
    """All run names, or just one grid size's runs (e.g. grid_size="3B")."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if grid_size:
        return [n for n in _grid_names(cfg) if n.startswith(f"{grid_size}-")]
    return list(cfg["runs"]) + _grid_names(cfg)


def _resolve_grid(name: str, cfg: dict) -> dict:
    size, method = name.rsplit("-", 1)
    grid = cfg["grid"]
    size_cfg = dict(grid["sizes"][size])
    full_overrides = size_cfg.pop("full", {})
    run = {**cfg["defaults"], **cfg["runs"][method], **grid["common"], **size_cfg}
    if run["method"] == "full":
        run.update(full_overrides)
    return run


def load_run(name: str, path: str | Path = EXPERIMENTS_PATH) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if name in cfg["runs"]:
        run = {**cfg["defaults"], **cfg["runs"][name]}
    elif name in _grid_names(cfg):
        run = _resolve_grid(name, cfg)
    else:
        raise KeyError(f"unknown run '{name}'; choose from {list_runs(path)}")
    run["run_name"] = name
    if run["method"] not in ("lora", "qlora", "full"):
        raise ValueError(f"run '{name}': unknown method '{run['method']}'")
    return run


def estimate_cost(gpu: str, n_gpus: int, seconds: float) -> float:
    return round(GPU_HOURLY_USD.get(gpu, 0.0) * n_gpus * seconds / 3600, 3)
