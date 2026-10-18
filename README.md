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
