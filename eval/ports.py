"""Extract a module's port list with pyslang (elaborated widths, so parameters resolve)."""

import re
from dataclasses import dataclass

from pyslang import ast, syntax

_CLOCK = re.compile(r"^(clk|clock)|(clk|clock)$|_clk_", re.IGNORECASE)
_RESET = re.compile(r"rst|reset", re.IGNORECASE)
_ACTIVE_LOW = re.compile(r"(_n|n|_b|_l)$|^n", re.IGNORECASE)


@dataclass(frozen=True)
class Port:
    name: str
    direction: str  # "input" | "output" | "inout"
    width: int
    unpacked: bool = False

    @property
    def is_clock(self) -> bool:
        return self.direction == "input" and self.width == 1 and bool(_CLOCK.search(self.name))

    @property
    def is_reset(self) -> bool:
        return self.direction == "input" and self.width == 1 and bool(_RESET.search(self.name))

    @property
    def reset_active_low(self) -> bool:
        return bool(_ACTIVE_LOW.search(self.name))


_DIRECTIONS = {
    ast.ArgumentDirection.In: "input",
    ast.ArgumentDirection.Out: "output",
    ast.ArgumentDirection.InOut: "inout",
}


class PortError(ValueError):
    pass


def extract_ports(code: str, top: str) -> list[Port]:
    """Return the ports of module `top`, raising PortError if it can't be elaborated."""
    compilation = ast.Compilation()
    compilation.addSyntaxTree(syntax.SyntaxTree.fromText(code))
    instances = {inst.name: inst for inst in compilation.getRoot().topInstances}
    if top not in instances:
        raise PortError(f"module '{top}' not found (top-level: {sorted(instances)})")

    ports = []
    for port in instances[top].body.portList:
        direction = _DIRECTIONS.get(getattr(port, "direction", None))
        if direction is None:  # interface ports and other non-net ports
            raise PortError(f"unsupported port kind for '{port.name}'")
        ports.append(Port(port.name, direction, port.type.bitWidth, port.type.isUnpackedArray))
    return ports
