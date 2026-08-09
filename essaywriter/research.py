"""Web search backing the research nodes."""

from typing import Iterable, List, Protocol

from tavily import TavilyClient


class SearchBackend(Protocol):
    """What the research nodes need from a search provider."""

    def gather(self, queries: Iterable[str]) -> List[str]:
        """Run each query and return the text snippets found."""
        ...


class TavilySearch:
    """Tavily-backed implementation of `SearchBackend`."""

    def __init__(self, api_key: str | None, results_per_query: int = 2) -> None:
        if not api_key:
            raise ValueError(
                "TAVILY_API_KEY is not set — the research nodes need a search key."
            )
        self._client = TavilyClient(api_key=api_key)
        self._results_per_query = results_per_query

    def gather(self, queries: Iterable[str]) -> List[str]:
        content: List[str] = []
        for query in queries:
            response = self._client.search(
                query=query, max_results=self._results_per_query
            )
            content.extend(result["content"] for result in response["results"])
        return content
