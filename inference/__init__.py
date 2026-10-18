"""Pick an inference backend for this machine and load a named model with it.

    from inference import load_backend
    backend = load_backend("ft-7b")              # auto: MLX on Apple Silicon, vLLM on CUDA, ...
    print(backend.generate("8-bit up counter with synchronous reset"))
"""

import importlib.util
import platform
import shutil
import sys
from pathlib import Path

import yaml

from inference.base import Backend, Generation

MODELS_PATH = Path(__file__).resolve().parents[1] / "configs" / "models.yaml"
BACKENDS = ("mlx", "vllm", "ollama", "transformers")

__all__ = ["BACKENDS", "Backend", "Generation", "detect_backend", "list_models", "load_backend"]


def _installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def _cuda_available() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return True  # driver present; vLLM brings its own torch


def detect_backend() -> str:
    """Apple Silicon -> MLX, CUDA + vLLM -> vLLM, running Ollama -> Ollama, else Transformers."""
    from inference.ollama_backend import ollama_available

    if sys.platform == "darwin" and platform.machine() == "arm64" and _installed("mlx_lm"):
        return "mlx"
    if sys.platform == "linux" and _installed("vllm") and _cuda_available():
        return "vllm"
    if ollama_available():
        return "ollama"
    return "transformers"


def list_models() -> dict[str, dict]:
    with open(MODELS_PATH) as f:
        return yaml.safe_load(f)["models"]


def _source(entry: dict, key: str) -> str:
    """Resolve a registry path relative to the repo root when it exists locally."""
    value = entry[key]
    local = MODELS_PATH.parents[1] / value
    return str(local) if local.exists() else value


def load_backend(model: str, backend: str = "auto", quantized: bool = False) -> Backend:
    """Load `model` (a configs/models.yaml name, Hub id, or local path) on `backend`.

    `quantized` picks the 4-bit build: MLX 4-bit, or AWQ on vLLM.
    """
    if backend == "auto":
        backend = detect_backend()
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend '{backend}', choose from {BACKENDS}")

    entry = list_models().get(model)
    if backend == "mlx":
        from inference.mlx_backend import MLXBackend

        key = "mlx_4bit" if quantized else "mlx"
        return MLXBackend(_source(entry, key) if entry else model)
    if backend == "vllm":
        from inference.vllm_backend import VLLMBackend

        key = "hf_awq" if quantized and entry and "hf_awq" in entry else "hf"
        return VLLMBackend(_source(entry, key) if entry else model)
    if backend == "ollama":
        from inference.ollama_backend import OllamaBackend

        return OllamaBackend(entry["ollama"] if entry else model)

    from inference.transformers_backend import TransformersBackend

    return TransformersBackend(_source(entry, "hf") if entry else model)
