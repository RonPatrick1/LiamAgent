"""Thin client for talking to a local Ollama server's chat API."""

import requests

DEFAULT_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "mistral-small3.2:24b"


class OllamaClient:
    def __init__(self, model=DEFAULT_MODEL, url=DEFAULT_URL):
        self.model = model
        self.url = url

    def chat(self, messages, tools=None):
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
        try:
            resp = requests.post(self.url, json=payload, timeout=600)
            resp.raise_for_status()
            return resp.json()["message"]
        except requests.exceptions.RequestException as exc:
            # A network/timeout failure here shouldn't take down the whole
            # REPL — surface it as a plain assistant message (no tool_calls)
            # so the caller's loop treats it as a final answer and moves on.
            return {"role": "assistant", "content": f"[error] Could not reach Ollama: {exc}"}
