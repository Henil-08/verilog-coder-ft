# Using the model in VS Code (Continue)

The fine-tuned model runs as a local server that [Continue](https://continue.dev) talks to.
Chat and edit work; tab autocomplete does not, because the model was trained on
spec-to-module chat examples, not fill-in-the-middle.

## 1. Start a server

**macOS (Apple Silicon), MLX:**
```bash
uv run --extra train --extra mlx python -m inference.convert --run <best 7B run> --name ft-7b --merge --mlx
uv run --extra mlx mlx_lm.server --model models/mlx/ft-7b-4bit --port 8080
```

**Windows (native) or anything without CUDA, Ollama:**
```bash
ollama create verilog-coder:7b -f models/gguf/ft-7b.Modelfile   # built by inference.convert --gguf
ollama serve
```

**Linux / Windows WSL2 with an NVIDIA GPU, vLLM:**
```bash
uv sync --extra vllm
uv run vllm serve henil08/verilog-coder-7b --port 8000 --max-model-len 4096
# 4-bit for 8-12 GB GPUs:
uv run vllm serve henil08/verilog-coder-7b-awq --port 8000 --max-model-len 4096
```

## 2. Point Continue at it

Copy the entries you need from [continue-config.yaml](continue-config.yaml) into
`~/.continue/config.yaml`, then pick the model in Continue's model dropdown.

## 3. Prompt it like it was trained

Results are best when the request reads like the training prompts: a short spec, plus
the module header if you already have one.

```text
Specification:
8-bit up/down counter with synchronous active-high reset and a load input.

Module interface:
```verilog
module counter(input clk, input rst, input load, input up, input [7:0] d, output reg [7:0] q);
```
```
