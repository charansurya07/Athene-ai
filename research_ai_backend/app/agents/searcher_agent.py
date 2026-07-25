"""
STAGE 3 — Searcher & Retriever Agent
Async information gathering (RAG).

Runs every Planner sub-query against the web search tool concurrently, and
(optionally) against a vector store of previously-ingested documents, then
hands the aggregated, de-duplicated results to the Verifier.
"""
from __future__ import annotations

import logging
from typing import Any

from app.models.domain import GraphState
from app.tools.search_tool import web_search_many

logger = logging.getLogger(__name__)


class SearcherAgent:
    """Stage 3 node: async web + vector retrieval."""

    name = "searcher"

    def __init__(self, vector_store: Any | None = None) -> None:
        # `vector_store` is injected by the orchestrator; kept optional so
        # this agent still works (web-only) when no vector DB is configured.
        self._vector_store = vector_store

    async def run(self, state: GraphState) -> dict[str, Any]:
        sub_queries = state.get("sub_queries") or [state.get("standardized_text", "")[:200]]

        web_results, vector_results = await self._gather(sub_queries)
        combined = _deduplicate_by_url(web_results + vector_results)

        logger.info(
            "Searcher retrieved %d web results and %d vector results (%d after de-dup)",
            len(web_results), len(vector_results), len(combined),
        )
        return {"search_results": combined}

    async def _gather(self, sub_queries: list[str]) -> tuple[list[dict], list[dict]]:
        web_results = await web_search_many(sub_queries)
        vector_results = await self._query_vector_store(sub_queries) if self._vector_store else []
        return web_results, vector_results

    async def _query_vector_store(self, sub_queries: list[str]) -> list[dict[str, Any]]:
        """
        Query a Chroma/Qdrant collection of previously-ingested documents.
        Returns results in the same shape as `web_search_many` so the
        Verifier can treat both sources uniformly.
        """
        results: list[dict[str, Any]] = []
        try:
            for query in sub_queries:
                hits = await self._vector_store.asimilarity_search(query, k=3)
                for hit in hits:
                    results.append(
                        {
                            "title": hit.metadata.get("title", "Indexed document"),
                            "url": hit.metadata.get("source", ""),
                            "snippet": hit.page_content[:500],
                            "published_at": hit.metadata.get("published_at"),
                            "source_query": query,
                        }
                    )
        except Exception:
            logger.exception("Vector store query failed — continuing with web results only")
        return results


def _deduplicate_by_url(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped = []
    for r in results:
        key = r.get("url") or r.get("title", "")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    return deduped
