from __future__ import annotations

import requests


class OllamaLLM:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        timeout: float = 180.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(
        self,
        *,
        instructions: str,
        user_prompt: str,
    ) -> str:
        response = requests.post(
            f"{self.base_url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": instructions,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                "stream": False,
                "think": False,
                "options": {
                    "temperature": 0.0,
                    "num_ctx": 4096,
                    "num_predict": 300,
                },
            },
            timeout=self.timeout,
        )

        response.raise_for_status()

        data = response.json()

        text = data.get("message", {}).get("content", "").strip()

        if not text:
            raise RuntimeError("Ollama returned an empty response.")

        return text
