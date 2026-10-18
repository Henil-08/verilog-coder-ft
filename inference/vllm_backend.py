"""CUDA backend via vLLM (Linux, or Windows through WSL2). Serves merged models,
LoRA adapters on their base, or AWQ 4-bit checkpoints."""

import time

from inference.base import DEFAULT_MAX_TOKENS, Backend, Generation


class VLLMBackend(Backend):
    name = "vllm"

    def __init__(self, model: str, adapter_path: str | None = None, max_model_len: int = 4096,
                 gpu_memory_utilization: float = 0.85):
        from vllm import LLM

        super().__init__(model)
        self._adapter = adapter_path
        # AWQ checkpoints carry their quantization config; vLLM picks it up automatically.
        self._llm = LLM(
            model=model,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_lora=adapter_path is not None,
            max_lora_rank=64,
        )

    def complete(self, messages: list[dict], max_tokens: int = DEFAULT_MAX_TOKENS,
                 temperature: float = 0.0) -> Generation:
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        lora = LoRARequest("adapter", 1, self._adapter) if self._adapter else None
        start = time.time()
        out = self._llm.chat([messages], SamplingParams(temperature=temperature, max_tokens=max_tokens),
                             lora_request=lora, use_tqdm=False)[0]
        seconds = time.time() - start
        completion = out.outputs[0]
        n = len(completion.token_ids)
        return Generation(
            text=completion.text,
            backend=self.name,
            model=self.model,
            prompt_tokens=len(out.prompt_token_ids),
            output_tokens=n,
            # Wall time includes prompt processing; with short prompts it is close to decode speed.
            tokens_per_second=round(n / seconds, 1) if seconds else None,
            finish_reason=completion.finish_reason,
        )
