"""Fallback backend with plain HF Transformers (CUDA, Apple MPS, or CPU). Slow, but runs anywhere."""

import time
from collections.abc import Iterator
from threading import Thread

from inference.base import DEFAULT_MAX_TOKENS, Backend, Generation


class TransformersBackend(Backend):
    name = "transformers"

    def __init__(self, model: str, adapter_path: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        super().__init__(model)
        if torch.cuda.is_available():
            self.device, dtype = "cuda", torch.bfloat16
        elif torch.backends.mps.is_available():
            self.device, dtype = "mps", torch.float16
        else:
            self.device, dtype = "cpu", torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(model)
        self._model = AutoModelForCausalLM.from_pretrained(model, dtype=dtype).to(self.device)
        if adapter_path:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, adapter_path)
        self._model.eval()

    def _inputs(self, messages):
        return self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.device)

    def _gen_kwargs(self, max_tokens, temperature):
        kwargs = {"max_new_tokens": max_tokens, "do_sample": temperature > 0}
        if temperature > 0:
            kwargs["temperature"] = temperature
        return kwargs

    def complete(self, messages: list[dict], max_tokens: int = DEFAULT_MAX_TOKENS,
                 temperature: float = 0.0) -> Generation:
        inputs = self._inputs(messages)
        prompt_len = inputs["input_ids"].shape[1]
        start = time.time()
        out = self._model.generate(**inputs, **self._gen_kwargs(max_tokens, temperature))
        seconds = time.time() - start
        new = out[0, prompt_len:]
        n = int(new.shape[0])
        return Generation(
            text=self._tokenizer.decode(new, skip_special_tokens=True),
            backend=self.name,
            model=self.model,
            prompt_tokens=prompt_len,
            output_tokens=n,
            tokens_per_second=round(n / seconds, 1) if seconds else None,
            finish_reason="length" if n >= max_tokens else "stop",
        )

    def stream(self, messages: list[dict], max_tokens: int = DEFAULT_MAX_TOKENS,
               temperature: float = 0.0) -> Iterator[str]:
        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)
        kwargs = self._inputs(messages) | self._gen_kwargs(max_tokens, temperature) | {"streamer": streamer}
        Thread(target=self._model.generate, kwargs=kwargs, daemon=True).start()
        yield from streamer
