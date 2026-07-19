"""Concrete search backends for SearchAgent.

A search tool exposes `search(query) -> list[str]`, returning human-readable
result snippets. `DuckDuckGoNewsTool` queries DuckDuckGo News (free, no API key).
"""

from ddgs import DDGS
from ddgs.exceptions import DDGSException


class DuckDuckGoNewsTool:
    def __init__(self, max_results=5, region="wt-wt"):
        self.max_results = max_results
        self.region = region

    def search(self, query):
        try:
            with DDGS() as ddgs:
                results = ddgs.news(
                    query, region=self.region, max_results=self.max_results
                )
        except DDGSException:
            # No results, rate limiting, or upstream error: fail soft so the
            # supervisor can react instead of crashing.
            return []

        snippets = []
        for r in results:
            snippets.append(
                f"[{r.get('date', '')}] {r.get('title', '')} "
                f"({r.get('source', '')})\n{r.get('body', '')}\n{r.get('url', '')}"
            )
        return snippets
