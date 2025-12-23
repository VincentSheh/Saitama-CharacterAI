from __future__ import annotations

import os
import json
from typing import Optional


class LTMSummary:
    def __init__(self, path: str = "memory/artifacts/ltm_summary.json"):
        self.path = path
        self.user_summary: str = ""
        self.chat_summary: str = ""

    def load(self) -> None:
        if not os.path.exists(self.path):
            self.user_summary = ""
            self.chat_summary = ""
            return
        with open(self.path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.user_summary = data.get("user_summary", "") or ""
        self.chat_summary = data.get("chat_summary", "") or ""

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(
                {"user_summary": self.user_summary, "chat_summary": self.chat_summary},
                f,
                ensure_ascii=False,
                indent=2,
            )