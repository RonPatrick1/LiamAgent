"""Thin client for talking to a local Ollama server's chat API."""

import requests

DEFAULT_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5:32b-instruct"


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
        resp = requests.post(self.url, json=payload, timeout=600)
        resp.raise_for_status()
        return resp.json()["message"]
