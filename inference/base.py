"""Backend-agnostic generation interface used by the CLI, Streamlit app, and benchmarks."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

from data.prompts import build_messages, extract_code

DEFAULT_MAX_TOKENS = 2048


@dataclass
class Generation:
    text: str
    backend: str
    model: str
    prompt_tokens: int | None = None
    output_tokens: int | None = None
    tokens_per_second: float | None = None  # decode speed, excluding prompt processing
    finish_reason: str | None = None

    @property
    def code(self) -> str:
        return extract_code(self.text)


class Backend(ABC):
    name: str = "base"

    def __init__(self, model: str):
        self.model = model

    @abstractmethod
    def complete(self, messages: list[dict], max_tokens: int = DEFAULT_MAX_TOKENS,
                 temperature: float = 0.0) -> Generation:
        """Run one chat completion."""

    def stream(self, messages: list[dict], max_tokens: int = DEFAULT_MAX_TOKENS,
               temperature: float = 0.0) -> Iterator[str]:
        """Yield text chunks; backends without native streaming yield once."""
        yield self.complete(messages, max_tokens, temperature).text

    def generate(self, prompt: str, interface: str | None = None, max_tokens: int = DEFAULT_MAX_TOKENS,
                 temperature: float = 0.0) -> str:
        """Spec (and optional module header) in, raw model response out."""
        return self.complete(build_messages(prompt, interface), max_tokens, temperature).text
