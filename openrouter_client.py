import os
import requests
from typing import List, Dict, Any, Optional

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

class OpenRouterClient:
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        site_url: Optional[str] = None,
        app_name: Optional[str] = None,
        timeout: int = 60,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("Missing OPENROUTER_API_KEY env var.")
        self.site_url = site_url or os.environ.get("OPENROUTER_SITE_URL")
        self.app_name = app_name or os.environ.get("OPENROUTER_APP_NAME", "characterai-rag")
        self.timeout = timeout

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.app_name:
            headers["X-Title"] = self.app_name

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=self.timeout)
        if r.status_code != 200:
            raise RuntimeError(f"OpenRouter error {r.status_code}: {r.text}")

        data = r.json()
        return data["choices"][0]["message"]["content"]