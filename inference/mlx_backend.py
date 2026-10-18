"""Apple Silicon backend via mlx-lm (fp16 or 4-bit converted models)."""

from collections.abc import Iterator

from inference.base import DEFAULT_MAX_TOKENS, Backend, Generation


class MLXBackend(Backend):
    name = "mlx"

    def __init__(self, model: str, adapter_path: str | None = None):
        import mlx.core as mx
        from mlx_lm import load

        super().__init__(model)
        mx.reset_peak_memory()  # so peak_memory_gb reflects this model, not an earlier one
        self._model, self._tokenizer = load(model, adapter_path=adapter_path)

    def _prompt(self, messages: list[dict]) -> str:
        return self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _responses(self, messages, max_tokens, temperature):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        return stream_generate(
            self._model, self._tokenizer, self._prompt(messages),
            max_tokens=max_tokens, sampler=make_sampler(temp=temperature),
        )

    def complete(self, messages: list[dict], max_tokens: int = DEFAULT_MAX_TOKENS,
                 temperature: float = 0.0) -> Generation:
        text, last = [], None
        for response in self._responses(messages, max_tokens, temperature):
            text.append(response.text)
            last = response
        return Generation(
            text="".join(text),
            backend=self.name,
            model=self.model,
            prompt_tokens=last.prompt_tokens if last else None,
            output_tokens=last.generation_tokens if last else None,
            tokens_per_second=round(last.generation_tps, 1) if last else None,
            finish_reason=last.finish_reason if last else None,
            peak_memory_gb=round(last.peak_memory, 2) if last else None,
        )

    def stream(self, messages: list[dict], max_tokens: int = DEFAULT_MAX_TOKENS,
               temperature: float = 0.0) -> Iterator[str]:
        for response in self._responses(messages, max_tokens, temperature):
            yield response.text
