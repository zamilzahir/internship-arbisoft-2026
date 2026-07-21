"""
plugins/web_search.py
----------------------
A web-search skill/plugin. Supports two backends, chosen via the
SEARCH_PROVIDER env var:

  - "brave"   -> Brave Search API (https://api.search.brave.com)
  - "serpapi" -> SerpAPI Google results (https://serpapi.com)

Both return a normalized list of {title, url, snippet} dicts so the rest
of the agent doesn't care which provider is active.
"""

from __future__ import annotations
import os
import requests
from hooks import logged_tool

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
SERPAPI_ENDPOINT = "https://serpapi.com/search"


def _search_brave(query: str, num_results: int) -> list[dict]:
    api_key = os.environ.get("BRAVE_API_KEY")
    if not api_key:
        raise RuntimeError("BRAVE_API_KEY is not set in the environment/.env")

    resp = requests.get(
        BRAVE_ENDPOINT,
        headers={"Accept": "application/json", "X-Subscription-Token": api_key},
        params={"q": query, "count": num_results},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    results = []
    for item in data.get("web", {}).get("results", [])[:num_results]:
        results.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": item.get("description", ""),
        })
    return results


def _search_serpapi(query: str, num_results: int) -> list[dict]:
    api_key = os.environ.get("SERPAPI_API_KEY")
    if not api_key:
        raise RuntimeError("SERPAPI_API_KEY is not set in the environment/.env")

    resp = requests.get(
        SERPAPI_ENDPOINT,
        params={"q": query, "num": num_results, "api_key": api_key, "engine": "google"},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    results = []
    for item in data.get("organic_results", [])[:num_results]:
        results.append({
            "title": item.get("title", ""),
            "url": item.get("link", ""),
            "snippet": item.get("snippet", ""),
        })
    return results


@logged_tool
def web_search(query: str, num_results: int = 5) -> str:
    """
    Run a web search and return a formatted string of results
    (title / url / snippet per hit). This is the function the agent's
    tool-use loop calls.
    """
    provider = os.environ.get("SEARCH_PROVIDER", "brave").lower()
    if provider == "brave":
        results = _search_brave(query, num_results)
    elif provider == "serpapi":
        results = _search_serpapi(query, num_results)
    else:
        raise ValueError(f"Unknown SEARCH_PROVIDER: {provider!r} (use 'brave' or 'serpapi')")

    if not results:
        return f"No results found for query: {query!r}"

    lines = [f"Search results for {query!r}:"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}\n   {r['url']}\n   {r['snippet']}")
    return "\n".join(lines)
