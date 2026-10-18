"""AWQ 4-bit quantization for vLLM, plus a vLLM decode-speed benchmark (fp16 vs AWQ), on Modal.

    # quantize a merged model from the Hub (or a volume path) and push the result
    uv run modal run inference/modal_awq.py::quantize --model henil08/verilog-coder-7b --push-to henil08/verilog-coder-7b-awq
    # single-stream decode speed on an L40S, full precision vs AWQ
    uv run modal run inference/modal_awq.py::bench --models henil08/verilog-coder-7b,henil08/verilog-coder-7b-awq

AWQ uses llm-compressor (W4A16, asymmetric, lm_head kept in 16-bit). Calibration uses
256 examples from the validation split, never test data. Output is a compressed-tensors
checkpoint, which vLLM loads directly.
"""

import json
import sys
from pathlib import Path

import modal

PROJECT_DIR = Path(__file__).resolve().parents[1]
VOL = "/vol"
CALIBRATION_SAMPLES = 256
CALIBRATION_MAX_LEN = 1024

app = modal.App("verilog-coder-ft-awq")
runs_volume = modal.Volume.from_name("verilog-ft-runs", create_if_missing=True)
hf_cache = modal.Volume.from_name("verilog-ft-hf-cache", create_if_missing=True)
secrets = [modal.Secret.from_name("huggingface-secret")]

# llm-compressor pins its own torch/transformers, so it gets its own image.
awq_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("llmcompressor==0.14.0", "datasets", "pyslang", "pyyaml", "hf-transfer")
    .env({"HF_HOME": "/cache/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_file(PROJECT_DIR / "data" / "processed" / "val.jsonl", "/root/val.jsonl")
    .add_local_python_source("data")
)
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.11")
    .entrypoint([])
    .uv_sync(str(PROJECT_DIR), extras=["vllm"])
    .env({"HF_HOME": "/cache/hf", "VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_python_source("data", "eval", "inference", "training")
    .add_local_dir(PROJECT_DIR / "data" / "processed", "/root/data/processed", ignore=["train_*.jsonl"])
    .add_local_dir(PROJECT_DIR / "data" / "raw" / "verilog-eval" / "dataset_spec-to-rtl",
                   "/root/data/raw/verilog-eval/dataset_spec-to-rtl")
)


@app.function(image=awq_image, gpu="H100", timeout=2 * 3600,
              volumes={VOL: runs_volume, "/cache": hf_cache}, secrets=secrets)
def quantize_model(model: str, out_name: str, push_to: str = "") -> str:
    from datasets import Dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.transform.awq import AWQModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from data.prompts import build_training_messages

    tokenizer = AutoTokenizer.from_pretrained(model)
    net = AutoModelForCausalLM.from_pretrained(model, dtype="auto")

    rows = [json.loads(line) for line in open("/root/val.jsonl")][:CALIBRATION_SAMPLES]
    texts = [tokenizer.apply_chat_template(build_training_messages(r), tokenize=False) for r in rows]
    calib = Dataset.from_dict(dict(tokenizer(texts, max_length=CALIBRATION_MAX_LEN, truncation=True)))

    recipe = [
        AWQModifier(duo_scaling="both"),
        QuantizationModifier(ignore=["lm_head"], scheme="W4A16_ASYM", targets=["Linear"]),
    ]
    oneshot(model=net, dataset=calib, recipe=recipe, max_seq_length=CALIBRATION_MAX_LEN,
            num_calibration_samples=len(rows))

    out = Path(VOL) / "awq" / out_name
    net.save_pretrained(out, save_compressed=True)
    tokenizer.save_pretrained(out)
    runs_volume.commit()
    if push_to:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(push_to, exist_ok=True)
        api.upload_folder(repo_id=push_to, folder_path=str(out), commit_message="Upload AWQ W4A16 build")
    return str(out)


@app.function(image=vllm_image, gpu="L40S", timeout=3600,
              volumes={VOL: runs_volume, "/cache": hf_cache}, secrets=secrets)
def bench_vllm(model: str, prompts: int = 10, max_tokens: int = 512) -> dict:
    import os
    import statistics

    os.chdir("/root")
    from eval.prompts import verilog_eval_items
    from inference.vllm_backend import VLLMBackend

    backend = VLLMBackend(model)
    items = [it["messages"] for it in verilog_eval_items()[:prompts]]
    backend.complete(items[0], 32)  # warm-up
    speeds = [backend.complete(m, max_tokens).tokens_per_second for m in items]
    return {"model": model, "variant": "vllm", "machine": "Modal L40S",
            "prompts": prompts, "max_tokens": max_tokens,
            "decode_tok_s_median": round(statistics.median(speeds), 1),
            "decode_tok_s_mean": round(statistics.mean(speeds), 1)}


@app.local_entrypoint()
def quantize(model: str, name: str = "ft-7b", push_to: str = ""):
    print(quantize_model.remote(model, name, push_to))


@app.local_entrypoint()
def bench(models: str, out: str = "results/inference_bench.json"):
    sys.path.insert(0, str(PROJECT_DIR))
    path = PROJECT_DIR / out
    rows = json.loads(path.read_text()) if path.exists() else []
    for row in bench_vllm.map(models.split(",")):
        rows = [r for r in rows if (r["model"], r.get("machine")) != (row["model"], row["machine"])]
        rows.append(row)
        print(f"{row['model']:<45} {row['decode_tok_s_median']:>8} tok/s (median, {row['machine']})")
    path.write_text(json.dumps(rows, indent=2))
