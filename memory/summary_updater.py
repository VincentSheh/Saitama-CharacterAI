from __future__ import annotations

from openrouter_client import OpenRouterClient


class SummaryUpdater:
    def __init__(self, client: OpenRouterClient):
        self.client = client

    def update_user_summary(self, existing: str, user_text: str, assistant_text: str) -> str:
        system = (
            "You maintain a long-term USER SUMMARY.\n"
            "Update the summary using the new turn.\n"
            "Keep it short and stable.\n"
            "Only include durable preferences, goals, constraints, and ongoing projects.\n"
            "Do not include transient details.\n"
            "Return ONLY the updated summary text, no bullets unless needed.\n"
            "Max 120 words.\n"
        )
        user = (
            f"EXISTING_USER_SUMMARY:\n{existing if existing else '(empty)'}\n\n"
            f"NEW_TURN:\nUSER: {user_text}\nASSISTANT: {assistant_text}\n\n"
            "Write UPDATED_USER_SUMMARY:"
        )
        return self.client.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=220,
        ).strip()

    def update_chat_summary(self, existing: str, user_text: str, assistant_text: str) -> str:
        system = (
            "You maintain a long-term CHAT SUMMARY.\n"
            "Summarize what has happened so far in the conversation.\n"
            "Keep it short.\n"
            "Return ONLY the updated summary text.\n"
            "Max 120 words.\n"
        )
        user = (
            f"EXISTING_CHAT_SUMMARY:\n{existing if existing else '(empty)'}\n\n"
            f"NEW_TURN:\nUSER: {user_text}\nASSISTANT: {assistant_text}\n\n"
            "Write UPDATED_CHAT_SUMMARY:"
        )
        return self.client.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.0,
            max_tokens=220,
        ).strip()