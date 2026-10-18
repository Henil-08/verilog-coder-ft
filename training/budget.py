"""Cost estimates for queued runs and a spend check against the Modal workspace.

Estimates scale the decode-free training throughput measured on the 1.5B ablation
(H100, tokens/s) by parameter count and GPU type. They are rough (±30%) and exist to
stop a launch that would obviously overrun a budget, not to bill anyone.
"""

import json
import re
import subprocess
from pathlib import Path

from training.config import GPU_HOURLY_USD

# Measured on Qwen2.5-Coder-1.5B, one H100, micro batch 8, length-grouped batches.
MEASURED_TPS_1P5B = {
    "L1": 20_920, "L2": 15_344, "L3": 16_424,
    "Q1": 17_331, "Q2": 13_480, "Q3": 13_258,
    "F1": 20_934, "F2": 18_864, "F3": 21_229,
}
# Throughput relative to the matching 1.5B run, measured in 10-step pilots on the grid's
# GPU setup for that size (3B full: 2xH100; 7B full: 2xH200). Families are L/Q/F.
MEASURED_SIZE_FACTOR = {
    ("3B", "L"): 8_694 / 16_424, ("3B", "Q"): 8_694 / 16_424, ("3B", "F"): 12_608 / 20_934,
    ("7B", "L"): 5_481 / 16_424, ("7B", "Q"): 4_293 / 13_258, ("7B", "F"): 7_493 / 20_934,
}
PARAMS_B = {"0.5B": 0.49, "1.5B": 1.54, "3B": 3.09, "7B": 7.62, "14B": 14.8}
GPU_SPEED = {"H100": 1.0, "H200": 1.2, "A100-80GB": 0.5, "A100-40GB": 0.45, "L40S": 0.4}
MULTI_GPU_EFFICIENCY = 0.85
OVERHEAD_SECONDS = 150  # image start, model download, tokenization
MARGIN = 1.0  # with the overhead above, estimates still run ~10% over actual cost
# Generation for all benchmarks on one L40S, per model (roughly scales with size).
EVAL_COST_USD = {"0.5B": 0.2, "1.5B": 0.3, "3B": 0.5, "7B": 1.0, "14B": 1.8}


def _size(base_model: str) -> str:
    match = re.search(r"(\d+(?:\.\d+)?)B", base_model)
    return f"{match.group(1)}B" if match else "7B"


def train_tokens(train_file: str) -> int:
    split = Path(train_file).stem
    stats = json.loads((Path(train_file).parent / "stats.json").read_text())
    return stats["splits"][split]["tokens"]


def estimate_run_cost(cfg: dict) -> float:
    method_key = cfg["run_name"].rsplit("-", 1)[-1]
    base_tps = MEASURED_TPS_1P5B.get(method_key, 15_000)
    size, n = _size(cfg["base_model"]), cfg["n_gpus"]
    factor = MEASURED_SIZE_FACTOR.get((size, method_key[0]))
    if factor is not None:
        tps = base_tps * factor
    else:  # no pilot for this size: scale by parameter count and GPU setup
        tps = base_tps * PARAMS_B["1.5B"] / PARAMS_B.get(size, 7.62)
        tps *= GPU_SPEED.get(cfg["gpu"], 1.0) * n * (MULTI_GPU_EFFICIENCY if n > 1 else 1.0)
    seconds = train_tokens(cfg["train_file"]) * cfg["epochs"] / tps + OVERHEAD_SECONDS
    return round(seconds / 3600 * GPU_HOURLY_USD[cfg["gpu"]] * n * MARGIN, 2)


def estimate_eval_cost(base_model: str) -> float:
    return EVAL_COST_USD.get(_size(base_model), 1.0)


def metered_cost() -> float | None:
    """This month's metered spend for the active Modal profile (None if unavailable)."""
    try:
        out = subprocess.run(["modal", "billing", "summary"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    match = re.search(r"Metered Cost:\s*\$?([\d.]+)", out)
    return float(match.group(1)) if match else None
