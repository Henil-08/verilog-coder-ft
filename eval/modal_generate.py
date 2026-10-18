"""Generate completions for both benchmarks with vLLM on Modal.

Models are named by run (from configs/experiments.yaml), `base` / `base:<size>`, or a Hub id:

    uv run modal run eval/modal_generate.py --models base,L1,L2,L3,Q1,Q2,Q3
    uv run modal run eval/modal_generate.py --models F1,F2,F3
    uv run modal run eval/modal_generate.py --models Qwen/Qwen2.5-Coder-14B-Instruct --gpu A100-80GB
    uv run modal run eval/modal_generate.py --models base,L1 --benchmarks verilog_eval_iface --suffix .iface

LoRA and QLoRA adapters share one vLLM instance of their bf16 base model (QLoRA
adapters are served on the bf16 base, the usual merge/deploy path). Decoding is
greedy, so pass@1 is deterministic. Outputs land in the `verilog-ft-runs` volume
under generations/ and are downloaded to outputs/generations/.
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import modal

PROJECT_DIR = Path(__file__).resolve().parents[1]
REMOTE_DIR = "/root/project"
VOL = "/vol"
MAX_NEW_TOKENS = 2048
MAX_MODEL_LEN = 4096

app = modal.App("verilog-coder-ft-generate")
runs_volume = modal.Volume.from_name("verilog-ft-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("verilog-ft-hf-cache", create_if_missing=True)

image = (
    # devel image: vLLM/FlashInfer JIT-compile some kernels and need nvcc
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.11")
    .entrypoint([])
    .uv_sync(str(PROJECT_DIR), extras=["vllm"])
    .env({
        "HF_HOME": "/cache/hf",
        "TOKENIZERS_PARALLELISM": "false",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",  # greedy decoding doesn't need it; skips a JIT build
    })
    .add_local_python_source("data", "eval", "training")
)


# Older Hub checkpoints whose tokenizer class transformers 5 resolves incorrectly
# (spaces dropped or inserted on encode and decode). They are served with a generic
# fast tokenizer built from their own tokenizer.json, which round-trips exactly.
GENERIC_TOKENIZER_MODELS = {"ishorn5/RTLCoder-Deepseek-v1.1", "yang-z/CodeV-QW-7B"}
# Byte-level BPE models whose text vLLM still returns as raw byte symbols (Ġ, Ċ, ...).
BYTE_LEVEL_TEXT_MODELS = {"ishorn5/RTLCoder-Deepseek-v1.1"}


def _byte_decoder() -> dict[str, int]:
    """Inverse of GPT-2's bytes_to_unicode table."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def byte_level_to_text(text: str) -> str:
    """Turn byte-level BPE symbols back into text (exact inverse of the byte mapping)."""
    table = _byte_decoder()
    return bytes(table[ch] for ch in text if ch in table).decode("utf-8", errors="replace")


def generic_tokenizer_dir(model: str) -> str:
    """Copy `model`'s tokenizer files, forcing the generic PreTrainedTokenizerFast class."""
    from huggingface_hub import hf_hub_download

    out = Path("/tmp/tokenizers") / model.replace("/", "--")
    out.mkdir(parents=True, exist_ok=True)
    for name in ("tokenizer.json", "special_tokens_map.json"):
        try:
            Path(out, name).write_bytes(Path(hf_hub_download(model, name)).read_bytes())
        except Exception:  # special_tokens_map.json is optional
            pass
    cfg = json.loads(Path(hf_hub_download(model, "tokenizer_config.json")).read_text())
    cfg["tokenizer_class"] = "PreTrainedTokenizerFast"
    cfg.pop("auto_map", None)
    Path(out, "tokenizer_config.json").write_text(json.dumps(cfg))
    return str(out)


@app.function(
    image=image,
    gpu="L40S",
    timeout=2 * 3600,
    volumes={VOL: runs_volume, "/cache": hf_cache},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def generate(model: str, variants: list[dict], items: list[dict], suffix: str = "", max_lora_rank: int = 64) -> list[dict]:
    """Run every variant (plain model or LoRA adapter on `model`) over `items`."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from data.prompts import extract_code

    runs_volume.reload()
    use_lora = any(v["adapter"] for v in variants)
    tokenizer = generic_tokenizer_dir(model) if model in GENERIC_TOKENIZER_MODELS else None
    llm = LLM(
        model=model,
        tokenizer=tokenizer,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        enable_lora=use_lora,
        max_lora_rank=max_lora_rank,
        seed=0,
    )
    params = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)
    conversations = [it["messages"] for it in items]

    stats = []
    out_dir = Path(VOL) / "generations"
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, variant in enumerate(variants, start=1):
        lora = LoRARequest(variant["tag"], i, variant["adapter"]) if variant["adapter"] else None
        start = time.time()
        outputs = llm.chat(conversations, params, lora_request=lora, use_tqdm=False)
        seconds = time.time() - start

        rows, total_tokens = [], 0
        for it, out in zip(items, outputs):
            completion = out.outputs[0]
            text = byte_level_to_text(completion.text) if model in BYTE_LEVEL_TEXT_MODELS else completion.text
            total_tokens += len(completion.token_ids)
            rows.append({
                "model": variant["tag"],
                "benchmark": it["benchmark"],
                "id": it["id"],
                "response": text,
                "code": extract_code(text),
                "output_tokens": len(completion.token_ids),
                "finish_reason": completion.finish_reason,
            })
        (out_dir / f"{variant['tag']}{suffix}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        stats.append({"model": variant["tag"], "file": f"{variant['tag']}{suffix}.jsonl", "seconds": round(seconds, 1), "output_tokens": total_tokens,
                      "batch_tokens_per_second": round(total_tokens / seconds, 1)})
        print(stats[-1])
    runs_volume.commit()
    hf_cache.commit()
    return stats


def resolve(name: str) -> tuple[str, dict]:
    """Map a model name to (vLLM model path, variant)."""
    from training.config import load_run

    if name == "base" or name.startswith("base:"):
        # "base" is the ablation's base model; "base:7B" is Qwen2.5-Coder-7B-Instruct.
        size = name.partition(":")[2]
        base = f"Qwen/Qwen2.5-Coder-{size}-Instruct" if size else load_run("L1")["base_model"]
        return base, {"tag": "base-" + base.split("/")[-1], "adapter": None}
    if "/" in name:
        return name, {"tag": name.split("/")[-1], "adapter": None}
    run = load_run(name)
    if run["method"] == "full":
        return f"{VOL}/runs/{name}/model", {"tag": name, "adapter": None}
    return run["base_model"], {"tag": name, "adapter": f"{VOL}/runs/{name}/model"}


@app.local_entrypoint()
def main(models: str, gpu: str = "", limit: int = 0, benchmarks: str = "", suffix: str = ""):
    """`--benchmarks verilog_eval_iface --suffix .iface` generates just that set into separate files."""
    sys.path.insert(0, str(PROJECT_DIR))
    from eval.prompts import BENCHMARKS, all_items

    items = all_items(benchmarks.split(",") if benchmarks else BENCHMARKS)
    if limit:
        by_bench: dict[str, list] = defaultdict(list)
        for it in items:
            by_bench[it["benchmark"]].append(it)
        items = [it for group in by_bench.values() for it in group[:limit]]
    print(f"{len(items)} prompts")

    # Adapters sharing a base go to one container; full models and Hub models get their own.
    groups: dict[str, list[dict]] = defaultdict(list)
    for name in models.split(","):
        path, variant = resolve(name)
        groups[path].append(variant)

    calls = []
    for path, variants in groups.items():
        fn = generate.with_options(gpu=gpu) if gpu else generate
        calls.append(fn.spawn(path, variants, items, suffix))
        print(f"launched {path}: {[v['tag'] for v in variants]}")

    local_dir = PROJECT_DIR / "outputs" / "generations"
    local_dir.mkdir(parents=True, exist_ok=True)
    for call in calls:
        for s in call.get():
            data = b"".join(runs_volume.read_file(f"generations/{s['file']}"))
            (local_dir / s["file"]).write_bytes(data)
            print(f"{s['model']:<32} {s['output_tokens']:>9,} tokens  {s['batch_tokens_per_second']:>8,.0f} tok/s")
    print(f"saved to {local_dir}")
