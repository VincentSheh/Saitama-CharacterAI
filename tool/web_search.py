from __future__ import annotations

import os
from typing import List, Dict, Any, Optional

from tavily import TavilyClient


class WebSearchTool:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("TAVILY_API_KEY")
        if not self.api_key:
            raise RuntimeError("Missing TAVILY_API_KEY env var.")
        self.client = TavilyClient(api_key=self.api_key)

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Returns a list of {title, url, content}.
        content is already a short extracted summary snippet from Tavily.
        """
        res = self.client.search(
            query=query,
            max_results=k,
            search_depth="basic",
            include_answer=False,
            include_raw_content=False,
        )

        out: List[Dict[str, Any]] = []
        for item in res.get("results", [])[:k]:
            out.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "content": item.get("content", ""),
                }
            )
        return out