"""Category detection, deduplication keys, and structural checks for RTL rows."""

import hashlib
import re

_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT = re.compile(r"//[^\n]*")
_WHITESPACE = re.compile(r"\s+")

_UVM = re.compile(r"\buvm_\w+|`uvm_\w+")
_SYSTEMVERILOG = re.compile(
    r"\b(always_ff|always_comb|always_latch|logic|typedef|enum|struct|"
    r"endinterface|endpackage|endclass|modport|unique\s+case|priority\s+case)\b"
    r"|\bimport\s+\w+::"
)
_MODULE_DECL = re.compile(r"\bmodule\s+(\w+)")
_INCLUDE = re.compile(r"`include\b")

CATEGORIES = ("uvm", "systemverilog", "verilog")


def strip_comments(code: str) -> str:
    return _LINE_COMMENT.sub("", _BLOCK_COMMENT.sub("", code))


def categorize(code: str) -> str:
    """Label a row as uvm, systemverilog, or verilog based on the constructs it uses.

    A "uvm" label means the code references UVM library identifiers; it says
    nothing about the quality or completeness of that testbench code.
    """
    body = strip_comments(code)
    if _UVM.search(body):
        return "uvm"
    if _SYSTEMVERILOG.search(body):
        return "systemverilog"
    return "verilog"


def dedup_key(code: str) -> str:
    """Hash of the code with comments and whitespace removed."""
    normalized = _WHITESPACE.sub("", strip_comments(code))
    return hashlib.sha1(normalized.encode()).hexdigest()


def module_name(ioheader: str) -> str | None:
    match = _MODULE_DECL.search(ioheader)
    return match.group(1) if match else None


def is_standalone(code: str, has_children: bool) -> bool:
    """True if the row is a single module with no submodules or `include deps."""
    if has_children:
        return False
    body = strip_comments(code)
    return len(_MODULE_DECL.findall(body)) == 1 and not _INCLUDE.search(body)
