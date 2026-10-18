"""Decode-speed benchmark: the same prompts through each local build of a model.

    uv run --extra mlx python -m inference.bench --model ft-7b --variants mlx,mlx-4bit,ollama

Variants: mlx (fp16), mlx-4bit, ollama (GGUF Q4_K_M), vllm, vllm-awq, transformers.
Prompts are the first N VerilogEval problems, greedy decoding, fixed max tokens.
Writes results/inference_bench.json (appending) and prints a table.
"""

import argparse
import gc
import json
import platform
import statistics
from datetime import datetime, timezone
from pathlib import Path

from eval.prompts import verilog_eval_items
from inference import load_backend

VARIANTS = {
    "mlx": ("mlx", False),
    "mlx-4bit": ("mlx", True),
    "ollama": ("ollama", False),
    "vllm": ("vllm", False),
    "vllm-awq": ("vllm", True),
    "transformers": ("transformers", False),
}


def bench(model: str, variant: str, prompts: list[list[dict]], max_tokens: int) -> dict:
    backend_name, quantized = VARIANTS[variant]
    backend = load_backend(model, backend_name, quantized)
    backend.complete(prompts[0], 32)  # warm-up: weights into memory, kernels compiled
    gens = [backend.complete(p, max_tokens) for p in prompts]
    source = backend.model
    del backend  # free weights before the next variant loads
    gc.collect()
    if backend_name == "mlx":
        import mlx.core as mx

        mx.clear_cache()
    speeds = [g.tokens_per_second for g in gens if g.tokens_per_second]
    peaks = [g.peak_memory_gb for g in gens if g.peak_memory_gb]
    return {
        "model": model,
        "variant": variant,
        "source": source,
        "prompts": len(prompts),
        "max_tokens": max_tokens,
        "output_tokens": sum(g.output_tokens or 0 for g in gens),
        "decode_tok_s_median": round(statistics.median(speeds), 1) if speeds else None,
        "decode_tok_s_mean": round(statistics.mean(speeds), 1) if speeds else None,
        "peak_memory_gb": max(peaks) if peaks else None,
        "machine": f"{platform.system()} {platform.machine()} {platform.processor()}".strip(),
        "date": datetime.now(timezone.utc).date().isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--variants", default="mlx,mlx-4bit")
    parser.add_argument("--prompts", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--machine", default="", help="label for the hardware, e.g. 'M4 Pro 24GB'")
    parser.add_argument("--out", default="results/inference_bench.json")
    args = parser.parse_args()

    prompts = [it["messages"] for it in verilog_eval_items()[: args.prompts]]
    out = Path(args.out)
    rows = json.loads(out.read_text()) if out.exists() else []
    for variant in args.variants.split(","):
        row = bench(args.model, variant, prompts, args.max_tokens)
        if args.machine:
            row["machine"] = args.machine
        rows = [r for r in rows if (r["model"], r["variant"], r["machine"]) != (row["model"], variant, row["machine"])]
        rows.append(row)
        print(f"{args.model:<10} {variant:<12} {row['decode_tok_s_median']:>8} tok/s (median)  "
              f"peak {row['peak_memory_gb']} GB")
    out.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
