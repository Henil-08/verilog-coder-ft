"""Build deployable copies of a trained run: merged HF weights, MLX (fp16 + 4-bit), and GGUF for Ollama.

    # 1. fetch the adapter (or full weights) from the Modal volume of the account that trained it
    uv run modal volume get verilog-ft-runs runs/7B-L3/model outputs/runs/7B-L3/model
    # 2. build everything under models/
    uv run --extra train --extra mlx python -m inference.convert --run 7B-L3 --name ft-7b --all

Steps (each can run alone): --merge, --mlx, --gguf. GGUF needs llama.cpp: `brew install
llama.cpp` (for llama-quantize) plus a source checkout for convert_hf_to_gguf.py, found
via $LLAMA_CPP_DIR (default ~/.local/opt/llama.cpp).
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from data.prompts import SYSTEM_PROMPT
from training.config import load_run

ROOT = Path(__file__).resolve().parents[1]
LLAMA_CPP_DIR = Path(os.environ.get("LLAMA_CPP_DIR", Path.home() / ".local/opt/llama.cpp"))
GGUF_QUANT = "Q4_K_M"
CHATML_TEMPLATE = (
    "{{- if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}"
    "{{- range .Messages }}{{- if ne .Role \"system\" }}<|im_start|>{{ .Role }}\n{{ .Content }}<|im_end|>\n{{ end }}{{ end }}"
    "<|im_start|>assistant\n"
)


def merge(run: str, name: str, runs_dir: Path) -> Path:
    """Fold a LoRA/QLoRA adapter into its bf16 base; full fine-tunes are copied as-is."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = load_run(run)
    src = runs_dir / run / "model"
    out = ROOT / "models" / "merged" / name
    if out.exists():
        shutil.rmtree(out)
    if cfg["method"] == "full":
        shutil.copytree(src, out)
        return out

    from peft import PeftModel

    base = AutoModelForCausalLM.from_pretrained(cfg["base_model"], dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, str(src)).merge_and_unload()
    model.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(src if (src / "tokenizer_config.json").exists() else cfg["base_model"]).save_pretrained(out)
    return out


def to_mlx(merged: Path, name: str) -> list[Path]:
    from mlx_lm import convert

    outs = []
    for suffix, quantize in (("", False), ("-4bit", True)):
        out = ROOT / "models" / "mlx" / f"{name}{suffix}"
        if out.exists():
            shutil.rmtree(out)
        convert(str(merged), mlx_path=str(out), quantize=quantize, q_bits=4, q_group_size=64,
                dtype="float16")
        outs.append(out)
    return outs


def to_gguf(merged: Path, name: str, register: bool = True) -> Path:
    script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
    if not script.exists():
        raise SystemExit(f"{script} not found: git clone --depth 1 https://github.com/ggml-org/llama.cpp {LLAMA_CPP_DIR}")
    out_dir = ROOT / "models" / "gguf"
    out_dir.mkdir(parents=True, exist_ok=True)
    f16, quant = out_dir / f"{name}-f16.gguf", out_dir / f"{name}-{GGUF_QUANT}.gguf"

    env = os.environ | {"PYTHONPATH": str(LLAMA_CPP_DIR / "gguf-py")}
    subprocess.run([sys.executable, str(script), str(merged), "--outtype", "f16", "--outfile", str(f16)],
                   check=True, env=env)
    subprocess.run(["llama-quantize", str(f16), str(quant), GGUF_QUANT], check=True)
    f16.unlink()

    modelfile = out_dir / f"{name}.Modelfile"
    # Qwen ChatML, spelled out: older Ollama releases ignore the template in GGUF metadata.
    modelfile.write_text(
        f"FROM ./{quant.name}\n"
        f'SYSTEM """{SYSTEM_PROMPT}"""\n'
        f'TEMPLATE """{CHATML_TEMPLATE}"""\n'
        'PARAMETER stop "<|im_end|>"\n'
        "PARAMETER temperature 0\n"
        "PARAMETER num_ctx 4096\n"
    )
    if register and shutil.which("ollama"):
        tag = f"verilog-coder:{name.removeprefix('ft-')}"
        subprocess.run(["ollama", "create", tag, "-f", str(modelfile)], check=True, cwd=out_dir)
        print(f"registered Ollama model {tag}")
    return quant


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, help="training run, e.g. 7B-L3")
    parser.add_argument("--name", required=True, help="output name, e.g. ft-7b (see configs/models.yaml)")
    parser.add_argument("--runs-dir", default="outputs/runs", type=Path)
    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--mlx", action="store_true")
    parser.add_argument("--gguf", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--no-ollama", action="store_true", help="build the GGUF but don't `ollama create`")
    args = parser.parse_args()

    merged = ROOT / "models" / "merged" / args.name
    if args.merge or args.all:
        merged = merge(args.run, args.name, args.runs_dir)
        print(f"merged -> {merged}")
    if args.mlx or args.all:
        for out in to_mlx(merged, args.name):
            print(f"mlx -> {out}")
    if args.gguf or args.all:
        print(f"gguf -> {to_gguf(merged, args.name, register=not args.no_ollama)}")


if __name__ == "__main__":
    main()
