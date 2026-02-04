"""
ollama_client.py

Minimal client for calling a local Ollama server.
Ollama must be running on http://localhost:11434.
"""

from __future__ import annotations

import requests


def generate(prompt: str, model: str = "mistral", timeout: int = 300) -> str:
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["response"]
