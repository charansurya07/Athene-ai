"""
Web search tool used by the Searcher & Retriever Agent.

Wraps the Tavily API (falls back to Serper if configured instead). Both are
plain REST APIs, so this is a thin async httpx wrapper — no SDK required.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings

logger = logging.getLogger(__name__)

TAVILY_ENDPOINT = "https://api.tavily.com/search"
SERPER_ENDPOINT = "https://google.serper.dev/search"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, min=1, max=8))
async def web_search(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    """
    Run a single web search query and return a normalized list of results:
    [{"title", "url", "snippet", "published_at"}]
    """
    settings = get_settings()

    if settings.tavily_api_key:
        return await _search_tavily(query, max_results, settings.tavily_api_key)
    if settings.serper_api_key:
        return await _search_serper(query, max_results, settings.serper_api_key)

    logger.warning("No search provider configured (TAVILY_API_KEY / SERPER_API_KEY) — returning no results.")
    return []


async def _search_tavily(query: str, max_results: int, api_key: str) -> list[dict[str, Any]]:
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "advanced",
        "include_answer": False,
    }
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(TAVILY_ENDPOINT, json=payload)
        resp.raise_for_status()
        data = resp.json()

    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "snippet": r.get("content", "")[:500],
            "published_at": r.get("published_date"),
        }
        for r in data.get("results", [])
    ]


async def _search_serper(query: str, max_results: int, api_key: str) -> list[dict[str, Any]]:
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = {"q": query, "num": max_results}
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.post(SERPER_ENDPOINT, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    return [
        {
            "title": r.get("title", ""),
            "url": r.get("link", ""),
            "snippet": r.get("snippet", ""),
            "published_at": r.get("date"),
        }
        for r in data.get("organic", [])[:max_results]
    ]


async def web_search_many(queries: list[str], max_results_per_query: int = 4) -> list[dict[str, Any]]:
    """Run several queries concurrently (used for the Planner's sub-queries)."""
    import asyncio

    results_nested = await asyncio.gather(
        *(web_search(q, max_results_per_query) for q in queries),
        return_exceptions=True,
    )
    flattened: list[dict[str, Any]] = []
    for q, res in zip(queries, results_nested):
        if isinstance(res, Exception):
            logger.error("Search failed for query %r: %s", q, res)
            continue
        for item in res:
            item["source_query"] = q
            flattened.append(item)
    return flattened
