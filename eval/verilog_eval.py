"""VerilogEval v2 spec-to-RTL scoring, mirroring the official Icarus Verilog harness.

Each problem ships a natural-language prompt, a reference (`RefModule`), and a
testbench that drives both modules and prints `Mismatches: N in M samples`.
Compile flags and the pass rule (no compile errors, zero mismatches) follow
NVlabs/verilog-eval's Makefile.in and scripts/sv-iv-analyze.

    git clone https://github.com/NVlabs/verilog-eval data/raw/verilog-eval
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path

DEFAULT_ROOT = Path("data/raw/verilog-eval/dataset_spec-to-rtl")
TOP = "TopModule"
# The testbenches need iverilog v12 (v13 rejects them); see scripts/install_iverilog12.sh.
_BIN_DIR = os.environ.get("IVERILOG_BIN_DIR", "")
IVERILOG_BIN = os.path.join(_BIN_DIR, "iverilog") if _BIN_DIR else "iverilog"
VVP_BIN = os.path.join(_BIN_DIR, "vvp") if _BIN_DIR else "vvp"
IVERILOG = [IVERILOG_BIN, "-Wall", "-Winfloop", "-Wno-timescale", "-g2012", "-s", "tb"]
_MISMATCHES = re.compile(r"^Mismatches: (\d+) in (\d+) samples$", re.MULTILINE)


def load_problems(root: str | Path = DEFAULT_ROOT) -> list[dict]:
    root = Path(root)
    names = (root / "problems.txt").read_text().split()
    return [
        {
            "id": name,
            "prompt": (root / f"{name}_prompt.txt").read_text().strip(),
            "ref": (root / f"{name}_ref.sv").read_text(),
            "test": (root / f"{name}_test.sv").read_text(),
        }
        for name in names
    ]


def score(problem: dict, code: str, timeout: int = 30) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        files = {"dut.sv": code, "test.sv": problem["test"], "ref.sv": problem["ref"]}
        for name, text in files.items():
            Path(tmp, name).write_text(text)
        try:
            build = subprocess.run(
                IVERILOG + ["-o", "sim", *files], cwd=tmp, capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "log": "compile timeout"}
        log = build.stdout + build.stderr
        if build.returncode != 0 or "error" in log:
            return {"status": "compile_error", "log": log[-2000:]}

        try:
            run = subprocess.run([VVP_BIN, "-n", "sim"], cwd=tmp, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "log": "simulation timeout"}

    out = run.stdout + run.stderr
    if "TIMEOUT" in out:
        return {"status": "timeout", "log": out[-1000:]}
    match = _MISMATCHES.search(out)
    if not match:
        return {"status": "no_result", "log": out[-1000:]}
    mismatches, samples = int(match.group(1)), int(match.group(2))
    return {"status": "pass" if mismatches == 0 else "mismatch", "mismatches": mismatches, "samples": samples}
