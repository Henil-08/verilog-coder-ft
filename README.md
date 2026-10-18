# verilog-coder-ft

Fine-tuning a small open coder LLM for Verilog/SystemVerilog code generation, evaluated by
functional correctness (Verilator compile + testbench pass@1), served through an
auto-detected MLX (Apple Silicon) or vLLM (CUDA) backend.

> Work in progress.

## Layout

```
data/        dataset loading + held-out split
training/    LoRA fine-tuning (Transformers + PEFT + DeepSpeed)
eval/        Verilator-based functional evaluation harness
inference/   backend abstraction (MLX / vLLM)
ui/          Streamlit app + CLI
```

## Setup

```bash
uv sync
```

Requires [Verilator](https://verilator.org) >= 5.0 on `PATH` for evaluation.

## Data

Training data comes from [OpenRTLSet](https://huggingface.co/datasets/ESCAD/OpenRTLSet)
(131k Verilog/SystemVerilog modules from GitHub, each paired with an LLM-written description).

```bash
uv run python -m data.split    # writes data/processed/*.jsonl + stats.json
```

The pipeline:

1. Drops the `lac-dcc/chimera` rows (54.6k randomly generated fuzzer modules with no design intent).
2. Dedupes on code with comments and whitespace stripped.
3. Keeps examples that fit in 2048 tokens once chat-formatted.
4. Labels each row `verilog`, `systemverilog`, or `uvm` by the constructs it uses.
5. Holds out a 500-row test set stratified by label. Test rows must be standalone modules
   (no submodules or `include`s) that pass `verilator --lint-only`, so the eval harness
   can simulate them.

| Stage | Rows |
|---|---|
| Raw | 131,309 |
| After dropping fuzzer rows | 76,668 |
| After dedup | 63,571 |
| After 2048-token cap | 54,732 |
| Test / val | 500 / 500 |
| Train (full / 40k / 20k subsets) | 53,732 / 40,000 / 20,000 |

On UVM: the raw corpus contains UVM testbench code in only 3 rows, and none of them survive
dedup and the length cap. The training data is effectively plain Verilog (~93%) and
SystemVerilog (~7%). Nothing in this project should be read as UVM capability.

## Training

Runs are defined in [configs/experiments.yaml](configs/experiments.yaml) and trained with
HF Transformers + PEFT + DeepSpeed ZeRO-2 ([configs/deepspeed/](configs/deepspeed/)). The loss
covers only the assistant (code) tokens. Metrics go to W&B.

The ablation compares three fine-tuning methods on Qwen2.5-Coder-1.5B-Instruct, using 20k
training examples (14.9M tokens) and 1 epoch:

| Run | Method | Adapter | LR |
|---|---|---|---|
| L1 / Q1 | LoRA / QLoRA (NF4) | r=8, attention projections | 2e-4 |
| L2 / Q2 | LoRA / QLoRA (NF4) | r=16, all linear layers | 2e-4 |
| L3 / Q3 | LoRA / QLoRA (NF4) | r=64, all linear layers | 1e-4 |
| F1 / F2 / F3 | Full fine-tune | — | 1e-5 / 2e-5 / 5e-5 |

Shared settings: effective batch 64, max length 2048, cosine schedule, 3% warmup, bf16,
gradient checkpointing, length-grouped batches. Each config was trained with a single seed.

```bash
# Local smoke test of the training loop (CPU/MPS, 0.5B model)
uv sync --extra train
uv run python -m training.train --experiment smoke --no-wandb --max-steps 4

# On Modal (needs `modal token new` plus `wandb-secret` / `huggingface-secret` Modal secrets)
uv run modal run training/modal_app.py --runs L2 --max-steps 20             # pilot
uv run modal run training/modal_app.py --runs L1,L2,L3,Q1,Q2,Q3,F1,F2,F3    # full grid, in parallel

# Anywhere with CUDA GPUs; the GPU count comes from torchrun
torchrun --nproc_per_node 2 -m training.train --experiment F2
```

Each run writes `summary.json` with tokens/sec, peak GPU memory, trainable params,
artifact size, losses, and estimated cost.
