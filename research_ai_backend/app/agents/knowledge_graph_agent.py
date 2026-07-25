"""
STAGE 6 — Knowledge Graph Extraction Agent
Graph RAG & relationship builder.

Extracts Subject -> Predicate -> Object triples from the standardized text
and verified sources, then materializes them into a NetworkX graph (or
Neo4j, if configured) for interactive UI visualization. The triples list
returned in the state is exactly the `[[subject, predicate, object], ...]`
shape the Athene AI frontend's `renderGraphSvg()` expects.
"""
from __future__ import annotations

import logging
from typing import Any

import networkx as nx
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

from app.config import get_settings
from app.models.domain import GraphState, Triple

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "Extract up to 6 factual Subject-Predicate-Object triples that capture "
    "the key relationships in the given text. Keep each element short "
    "(a few words). Prefer concrete entities and relationships over vague "
    "ones."
)


class TripleExtraction(BaseModel):
    triples: list[Triple] = Field(default_factory=list, max_length=6)


class KnowledgeGraphAgent:
    """Stage 6 node: LLM-based triple extraction + graph store materialization."""

    name = "knowledge_graph"

    def __init__(self) -> None:
        settings = get_settings()
        self._settings = settings
        self._llm = ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
            temperature=0,
            max_tokens=600,
        ).with_structured_output(TripleExtraction)

    async def run(self, state: GraphState) -> dict[str, Any]:
        text = state.get("standardized_text", "")
        try:
            extraction: TripleExtraction = await self._llm.ainvoke(
                [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": text[:6000]},
                ]
            )
            triples = [t.as_tuple() for t in extraction.triples]
        except Exception:
            logger.exception("Knowledge graph extraction failed — returning no triples")
            triples = []

        await self._materialize(triples, request_id=state.get("request_id", "unknown"))

        logger.info("Knowledge graph agent extracted %d triples", len(triples))
        return {"triples": triples}

    async def _materialize(self, triples: list[tuple[str, str, str]], request_id: str) -> None:
        """Persist the graph so it can be queried later (e.g. for a UI deep-link)."""
        if self._settings.graph_store_provider == "neo4j":
            await self._materialize_neo4j(triples, request_id)
        else:
            self._materialize_networkx(triples, request_id)

    def _materialize_networkx(self, triples: list[tuple[str, str, str]], request_id: str) -> nx.DiGraph:
        graph = nx.DiGraph()
        for subject, predicate, obj in triples:
            graph.add_edge(subject, obj, label=predicate, request_id=request_id)
        return graph

    async def _materialize_neo4j(self, triples: list[tuple[str, str, str]], request_id: str) -> None:
        try:
            from neo4j import AsyncGraphDatabase  # type: ignore

            driver = AsyncGraphDatabase.driver(
                self._settings.neo4j_uri,
                auth=(self._settings.neo4j_user, self._settings.neo4j_password),
            )
            async with driver.session() as session:
                for subject, predicate, obj in triples:
                    await session.run(
                        "MERGE (s:Entity {name: $subject}) "
                        "MERGE (o:Entity {name: $object}) "
                        "MERGE (s)-[r:RELATION {type: $predicate, request_id: $request_id}]->(o)",
                        subject=subject, object=obj, predicate=predicate, request_id=request_id,
                    )
            await driver.close()
        except Exception:
            logger.exception("Neo4j write failed — triples were still returned to the caller")
