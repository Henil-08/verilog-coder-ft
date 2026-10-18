"""Differential simulation: does a generated module behave like the reference?

Both modules are instantiated side by side in a generated SystemVerilog testbench,
driven with identical random stimulus, and their outputs compared every half cycle.
Verilator builds and runs the testbench.

Limits worth knowing when reading results:
- Random stimulus is evidence of equivalence, not proof (deep states may go unvisited).
- Registers start at 0 (--x-initial 0) in both models, so designs that differ only in
  their behaviour from an uninitialised state are treated as equivalent.
- Clocks and resets are found from port names; a misnamed clock is driven as data.
"""

import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from eval.ports import Port, PortError, extract_ports
from eval.verilator import lint

RESET_CYCLES = 4
RESET_REASSERT_PCT = 2  # chance per cycle of pulsing reset again after the initial reset

STATUSES = (
    "pass",            # outputs matched for every compared sample
    "mismatch",        # simulated, outputs differed
    "compile_error",   # generated code fails Verilator lint on its own
    "port_mismatch",   # generated interface differs from the reference
    "build_error",     # testbench + both modules failed to build
    "timeout",
    "unsupported",     # reference uses ports this harness can't drive (inout, unpacked)
)


@dataclass
class SimResult:
    status: str
    mismatches: int | None = None
    samples: int | None = None
    first_mismatch: str | None = None
    log: str = ""
    details: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.status == "pass"


def rename_module(code: str, old: str, new: str) -> str:
    """Rename the declaration of module `old`; signals that share its name are left alone."""
    return re.sub(rf"\b(module|macromodule)(\s+){re.escape(old)}\b", rf"\1\2{new}", code)


def _random_expr(width: int) -> str:
    words = (width + 31) // 32
    return "{" + ", ".join(["$urandom"] * words) + "}" if words > 1 else "$urandom"


def build_testbench(ports: list[Port], ref: str, dut: str, cycles: int, seed: int) -> str:
    clocks = [p for p in ports if p.is_clock]
    resets = [p for p in ports if p.is_reset]
    data_inputs = [p for p in ports if p.direction == "input" and p not in clocks and p not in resets]
    outputs = [p for p in ports if p.direction == "output"]

    def decl(p: Port, prefix: str = "") -> str:
        rng = f"[{p.width - 1}:0] " if p.width > 1 else ""
        return f"  logic {rng}{prefix}{p.name};"

    lines = ["`timescale 1ns/1ps", "module tb;"]
    lines += [decl(p) for p in ports if p.direction == "input"]
    lines += [decl(p, "ref_") for p in outputs] + [decl(p, "dut_") for p in outputs]

    def conns(prefix: str) -> str:
        parts = [f".{p.name}({p.name})" for p in ports if p.direction == "input"]
        parts += [f".{p.name}({prefix}{p.name})" for p in outputs]
        return ", ".join(parts)

    lines.append(f"  {ref} u_ref ({conns('ref_')});")
    lines.append(f"  {dut} u_dut ({conns('dut_')});")
    lines += ["  int mismatches = 0;", "  int samples = 0;", "  string first = \"\";"]

    def assert_reset(on: bool) -> list[str]:
        return [f"    {r.name} = 1'b{int(on != r.reset_active_low)};" for r in resets]

    lines.append("  task automatic check(input int cycle);")
    lines.append("    samples++;")
    for p in outputs:
        lines += [
            f"    if (ref_{p.name} !== dut_{p.name}) begin",
            "      mismatches++;",
            f"      if (first == \"\") first = $sformatf(\"cycle %0d {p.name}: ref=%h dut=%h\", "
            f"cycle, ref_{p.name}, dut_{p.name});",
            "    end",
        ]
    lines.append("  endtask")

    drive_data = [f"    {p.name} = {_random_expr(p.width)};" for p in data_inputs]
    lines += ["  initial begin", f"    void'($urandom({seed}));"]
    lines += [f"    {c.name} = 1'b0;" for c in clocks] + [f"    {p.name} = '0;" for p in data_inputs]
    lines += [f"    for (int i = 0; i < {cycles}; i++) begin"]
    if resets:
        lines += [f"      if (i < {RESET_CYCLES} || ($urandom % 100) < {RESET_REASSERT_PCT}) begin"]
        lines += ["  " + s for s in assert_reset(True)]
        lines += ["      end else begin"] + ["  " + s for s in assert_reset(False)] + ["      end"]
    lines += ["  " + s for s in drive_data]
    lines += ["      #1;", "      check(i);"]
    if clocks:
        lines += [f"      {c.name} = 1'b1;" for c in clocks] + ["      #1;", "      check(i);"]
        lines += [f"      {c.name} = 1'b0;" for c in clocks] + ["      #1;"]
    lines += [
        "    end",
        "    $display(\"RESULT mismatches=%0d samples=%0d\", mismatches, samples);",
        "    if (first != \"\") $display(\"FIRST %s\", first);",
        "    $finish;",
        "  end",
        "endmodule",
    ]
    return "\n".join(lines) + "\n"


def _run(cmd: list[str], cwd: str, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def diff_sim(
    ref_code: str,
    dut_code: str,
    top: str,
    cycles: int = 500,
    seed: int = 1,
    build_timeout: int = 180,
    run_timeout: int = 30,
) -> SimResult:
    """Compare `dut_code` against `ref_code`, both expected to define module `top`."""
    try:
        ref_ports = extract_ports(ref_code, top)
    except PortError as exc:
        return SimResult("unsupported", log=f"reference: {exc}")
    if any(p.direction == "inout" or p.unpacked or p.width < 1 for p in ref_ports):
        return SimResult("unsupported", log="reference has inout/unpacked/zero-width ports")

    lint_result = lint(dut_code, top=top)
    if not lint_result.ok:
        return SimResult("compile_error", log=lint_result.log[-2000:])

    try:
        dut_ports = extract_ports(dut_code, top)
    except PortError as exc:
        return SimResult("port_mismatch", log=str(exc))
    if sorted(ref_ports, key=lambda p: p.name) != sorted(dut_ports, key=lambda p: p.name):
        diff = set(ref_ports) ^ set(dut_ports)
        return SimResult("port_mismatch", log="; ".join(sorted(str(p) for p in diff)))

    ref_name, dut_name = f"ref_{top}", f"dut_{top}"
    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp, "ref.sv").write_text(rename_module(ref_code, top, ref_name))
        Path(tmp, "dut.sv").write_text(rename_module(dut_code, top, dut_name))
        Path(tmp, "tb.sv").write_text(build_testbench(ref_ports, ref_name, dut_name, cycles, seed))
        build = [
            "verilator", "--binary", "--timing", "-j", "1",
            "-Wno-fatal", "-Wno-lint", "-Wno-style",
            "--x-initial", "0", "--x-assign", "0",
            "-O0", "-CFLAGS", "-O0",
            "--top-module", "tb", "-o", "sim",
            "tb.sv", "ref.sv", "dut.sv",
        ]
        try:
            proc = _run(build, tmp, build_timeout)
        except subprocess.TimeoutExpired:
            return SimResult("timeout", log="build timeout")
        if proc.returncode != 0:
            return SimResult("build_error", log=(proc.stdout + proc.stderr)[-2000:])

        try:
            proc = _run(["./obj_dir/sim"], tmp, run_timeout)
        except subprocess.TimeoutExpired:
            return SimResult("timeout", log="simulation timeout")

    out = proc.stdout
    match = re.search(r"RESULT mismatches=(\d+) samples=(\d+)", out)
    if not match:
        return SimResult("build_error", log=(out + proc.stderr)[-2000:])
    mismatches, samples = int(match.group(1)), int(match.group(2))
    first = re.search(r"FIRST (.*)", out)
    return SimResult(
        "pass" if mismatches == 0 else "mismatch",
        mismatches=mismatches,
        samples=samples,
        first_mismatch=first.group(1) if first else None,
    )
