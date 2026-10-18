"""Chat prompt format shared by training, evaluation, and inference."""

import re

SYSTEM_PROMPT = (
    "You are a hardware design assistant. Write synthesizable Verilog or "
    "SystemVerilog that matches the given specification and module interface. "
    "Return the complete module in a single code block."
)

_CODE_BLOCK = re.compile(r"```(?:verilog|systemverilog|sv|v)?\s*\n(.*?)```", re.DOTALL)
_OPEN_FENCE = re.compile(r"```(?:verilog|systemverilog|sv|v)?\s*\n(.*)", re.DOTALL)
_MODULE_SPAN = re.compile(r"\bmodule\b.*\bendmodule\b", re.DOTALL)


def interface_prose(module: str, ports: list[tuple[str, str, int]]) -> str:
    """Describe a port list in words: ports are (direction, name, width) tuples."""
    lines = [f"Module name: `{module}`. Ports:"]
    for direction, name, width in ports:
        lines.append(f"- {direction} `{name}`" + (f" ({width} bits)" if width > 1 else ""))
    return "\n".join(lines)


def user_message(description: str, ioheader: str | None = None, prose_interface: str | None = None) -> str:
    msg = f"Specification:\n{description.strip()}"
    if prose_interface:
        msg += f"\n\n{prose_interface}"
    elif ioheader:
        msg += f"\n\nModule interface:\n```verilog\n{ioheader.strip()}\n```"
    return msg


def build_messages(description: str, ioheader: str | None = None, prose_interface: str | None = None) -> list[dict]:
    """Messages for generation (no assistant turn)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message(description, ioheader, prose_interface)},
    ]


def build_training_messages(row: dict, prose_interface: str | None = None) -> list[dict]:
    """Messages for supervised fine-tuning, with the reference code as the assistant turn.

    With `prose_interface`, the ports are described in words instead of the header
    code block, so the model also learns to write the module header itself.
    """
    return build_messages(row["description"], row["ioheader"], prose_interface) + [
        {"role": "assistant", "content": f"```verilog\n{row['code'].strip()}\n```"}
    ]


def extract_code(text: str) -> str:
    """Pull the first fenced code block out of a model response, else return it whole.

    A response cut off at the token limit has an opening fence but no closing one;
    everything after the opening fence is taken in that case. Unfenced responses
    (some published checkpoints answer in plain text) yield the span from the first
    `module` to the last `endmodule`.
    """
    match = _CODE_BLOCK.search(text) or _OPEN_FENCE.search(text)
    if match:
        return match.group(1).strip()
    span = _MODULE_SPAN.search(text)
    return (span.group(0) if span else text).strip()
