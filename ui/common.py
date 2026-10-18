"""Helpers shared by the CLI and the Streamlit app."""

import re

from eval.verilator import LintResult, lint, verilator_available

_MODULE = re.compile(r"\bmodule\s+(\w+)")


def compile_check(code: str) -> LintResult | None:
    """Verilator lint of generated code; None when Verilator isn't installed."""
    if not verilator_available():
        return None
    match = _MODULE.search(code)
    return lint(code, top=match.group(1) if match else None)
