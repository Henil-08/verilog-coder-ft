"""Chat prompt format shared by training, evaluation, and inference."""

import re

SYSTEM_PROMPT = (
    "You are a hardware design assistant. Write synthesizable Verilog or "
    "SystemVerilog that matches the given specification and module interface. "
    "Return the complete module in a single code block."
)

_CODE_BLOCK = re.compile(r"```(?:verilog|systemverilog|sv|v)?\s*\n(.*?)```", re.DOTALL)


def user_message(description: str, ioheader: str | None = None) -> str:
    msg = f"Specification:\n{description.strip()}"
    if ioheader:
        msg += f"\n\nModule interface:\n```verilog\n{ioheader.strip()}\n```"
    return msg


def build_messages(description: str, ioheader: str | None = None) -> list[dict]:
    """Messages for generation (no assistant turn)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message(description, ioheader)},
    ]


def build_training_messages(row: dict) -> list[dict]:
    """Messages for supervised fine-tuning, with the reference code as the assistant turn."""
    return build_messages(row["description"], row["ioheader"]) + [
        {"role": "assistant", "content": f"```verilog\n{row['code'].strip()}\n```"}
    ]


def extract_code(text: str) -> str:
    """Pull the first fenced code block out of a model response, else return it whole."""
    match = _CODE_BLOCK.search(text)
    return (match.group(1) if match else text).strip()
