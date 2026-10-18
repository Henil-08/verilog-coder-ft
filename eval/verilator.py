"""Thin wrappers around the Verilator CLI."""

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LintResult:
    ok: bool
    log: str


def verilator_available() -> bool:
    return shutil.which("verilator") is not None


def lint(code: str, top: str | None = None, systemverilog: bool = True, timeout: int = 20) -> LintResult:
    """Run `verilator --lint-only` on a single source string.

    Warnings are allowed (-Wno-fatal); only errors count as failure.
    """
    suffix = ".sv" if systemverilog else ".v"
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / f"design{suffix}"
        src.write_text(code)
        cmd = ["verilator", "--lint-only", "-Wno-fatal", "-Wno-lint", "-Wno-style"]
        if top:
            cmd += ["--top-module", top]
        cmd.append(str(src))
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=tmp)
        except subprocess.TimeoutExpired:
            return LintResult(False, f"timeout after {timeout}s")
        return LintResult(proc.returncode == 0, proc.stderr)
