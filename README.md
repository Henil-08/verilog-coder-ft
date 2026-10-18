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
