"""Generate Verilog/SystemVerilog from a spec in the terminal.

    uv run python -m ui.cli "8-bit up counter with synchronous active-high reset and enable"
    uv run python -m ui.cli --spec-file spec.txt --interface-file header.v --model ft-7b --check
    uv run python -m ui.cli "UART transmitter, 8N1" --model ft-3b --compare base-3b --quantized
    echo "4:1 mux" | uv run python -m ui.cli - --backend ollama --stream

Models are names from configs/models.yaml, Hub ids, or local paths.
"""

import argparse
import sys
from pathlib import Path

from data.prompts import build_messages
from inference import BACKENDS, load_backend
from ui.common import compile_check


def read_spec(args) -> str:
    if args.spec_file:
        return Path(args.spec_file).read_text()
    if args.spec in (None, "-"):
        return sys.stdin.read()
    return args.spec


def run(model: str, messages: list[dict], args) -> str:
    backend = load_backend(model, args.backend, args.quantized)
    print(f"--- {model} [{backend.name}{', 4-bit' if args.quantized else ''}]", file=sys.stderr)
    if args.stream:
        text = ""
        for chunk in backend.stream(messages, args.max_tokens, args.temperature):
            print(chunk, end="", flush=True)
            text += chunk
        print()
        from inference.base import Generation

        code = Generation(text, backend.name, model).code
    else:
        gen = backend.complete(messages, args.max_tokens, args.temperature)
        code = gen.code
        print(code)
        speed = f", {gen.tokens_per_second} tok/s" if gen.tokens_per_second else ""
        print(f"--- {gen.output_tokens} tokens{speed}, finish={gen.finish_reason}", file=sys.stderr)

    if args.check:
        result = compile_check(code)
        if result is None:
            print("--- compile check skipped: verilator not on PATH", file=sys.stderr)
        else:
            print(f"--- verilator lint: {'PASS' if result.ok else 'FAIL'}", file=sys.stderr)
            if not result.ok:
                print(result.log.strip()[:2000], file=sys.stderr)
    return code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("spec", nargs="?", help="module specification ('-' or omit to read stdin)")
    parser.add_argument("--spec-file")
    parser.add_argument("--interface-file", help="module header to implement (optional)")
    parser.add_argument("--model", default="ft-7b")
    parser.add_argument("--compare", metavar="MODEL", help="also generate with this model, e.g. base-7b")
    parser.add_argument("--backend", default="auto", choices=("auto",) + BACKENDS)
    parser.add_argument("--quantized", action="store_true", help="use the 4-bit build (MLX 4-bit / AWQ)")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--check", action="store_true", help="lint the result with Verilator")
    parser.add_argument("--out", help="write the (first model's) code to this file")
    args = parser.parse_args()

    interface = Path(args.interface_file).read_text() if args.interface_file else None
    messages = build_messages(read_spec(args), interface)
    code = run(args.model, messages, args)
    if args.compare:
        run(args.compare, messages, args)
    if args.out:
        Path(args.out).write_text(code + "\n")


if __name__ == "__main__":
    main()
