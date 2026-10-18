"""Backend for a local Ollama server (native Windows, macOS, Linux; CPU or GPU) serving a GGUF build."""

import json
import os
import urllib.request
from collections.abc import Iterator

from inference.base import DEFAULT_MAX_TOKENS, Backend, Generation

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def ollama_available(host: str = DEFAULT_HOST, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout):
            return True
    except OSError:
        return False


class OllamaBackend(Backend):
    name = "ollama"

    def __init__(self, model: str, host: str = DEFAULT_HOST):
        super().__init__(model)
        self.host = host.rstrip("/")

    def _request(self, messages, max_tokens, temperature, stream: bool):
        body = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        req = urllib.request.Request(f"{self.host}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return urllib.request.urlopen(req, timeout=600)

    def complete(self, messages: list[dict], max_tokens: int = DEFAULT_MAX_TOKENS,
                 temperature: float = 0.0) -> Generation:
        with self._request(messages, max_tokens, temperature, stream=False) as resp:
            data = json.loads(resp.read())
        n, ns = data.get("eval_count"), data.get("eval_duration")
        return Generation(
            text=data["message"]["content"],
            backend=self.name,
            model=self.model,
            prompt_tokens=data.get("prompt_eval_count"),
            output_tokens=n,
            tokens_per_second=round(n / ns * 1e9, 1) if n and ns else None,
            finish_reason=data.get("done_reason"),
        )

    def stream(self, messages: list[dict], max_tokens: int = DEFAULT_MAX_TOKENS,
               temperature: float = 0.0) -> Iterator[str]:
        with self._request(messages, max_tokens, temperature, stream=True) as resp:
            for line in resp:
                if line.strip():
                    yield json.loads(line)["message"]["content"]
