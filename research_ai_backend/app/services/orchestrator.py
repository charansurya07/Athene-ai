"""
LangGraph workflow orchestrator.

Wires the 7 agents into a single stateful graph:

    ingestion -> planner -> searcher -> verifier -> recommendation
              -> knowledge_graph -> writer -> END

The Planner node runs here exactly like every other agent — it is only
hidden from the *frontend's* visible pipeline UI, not from the actual
backend execution.

Also exposes `TopicAnalysisPipeline`, a smaller, standalone pipeline behind
the "Research Analysis" feature (a single-topic lookup that doesn't need
file ingestion or the full 7-stage graph).
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from app.agents import (
    IngestionAgent,
    KnowledgeGraphAgent,
    PlannerAgent,
    RecommendationAgent,
    SearcherAgent,
    VerifierAgent,
    WriterAgent,
)
from app.config import get_settings
from app.models.domain import GraphState, Triple, UploadedFileRef
from app.tools.search_tool import web_search

logger = logging.getLogger(__name__)


class ResearchOrchestrator:
    """Builds the LangGraph graph once and reuses it for every request."""

    def __init__(self, vector_store: Any | None = None) -> None:
        self._ingestion = IngestionAgent()
        self._planner = PlannerAgent()
        self._searcher = SearcherAgent(vector_store=vector_store)
        self._verifier = VerifierAgent()
        self._recommendation = RecommendationAgent()
        self._knowledge_graph = KnowledgeGraphAgent()
        self._writer = WriterAgent()
        self._graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(GraphState)

        builder.add_node(self._ingestion.name, self._ingestion.run)
        builder.add_node(self._planner.name, self._planner.run)
        builder.add_node(self._searcher.name, self._searcher.run)
        builder.add_node(self._verifier.name, self._verifier.run)
        builder.add_node(self._recommendation.name, self._recommendation.run)
        builder.add_node(self._knowledge_graph.name, self._knowledge_graph.run)
        builder.add_node(self._writer.name, self._writer.run)

        builder.set_entry_point(self._ingestion.name)
        builder.add_edge(self._ingestion.name, self._planner.name)
        builder.add_edge(self._planner.name, self._searcher.name)
        builder.add_edge(self._searcher.name, self._verifier.name)
        builder.add_edge(self._verifier.name, self._recommendation.name)
        builder.add_edge(self._recommendation.name, self._knowledge_graph.name)
        builder.add_edge(self._knowledge_graph.name, self._writer.name)
        builder.add_edge(self._writer.name, END)

        return builder.compile()

    async def run(
        self,
        prompt: str | None,
        url: str | None,
        files: list[UploadedFileRef],
    ) -> GraphState:
        initial_state: GraphState = {
            "request_id": str(uuid.uuid4()),
            "raw_prompt": prompt or "",
            "raw_url": url,
            "raw_files": files,
            "errors": [],
        }
        logger.info(
            "Starting research run %s (prompt=%s, url=%s, %d files)",
            initial_state["request_id"], bool(prompt), bool(url), len(files),
        )
        final_state: GraphState = await self._graph.ainvoke(initial_state)
        return final_state


# --------------------------------------------------------------------------
# Standalone "Research Analysis" (topic lookup) pipeline
# --------------------------------------------------------------------------

_TOPIC_SYSTEM_PROMPT = (
    "You are the research analysis module of Athene AI. Given a topic and a "
    "set of web search snippets about it, produce: a 2-3 sentence overview; "
    "the topic's origin (where, when, and by whom it was created or first "
    "described); up to 4 key facts; an overall credibility score (0-100) "
    "for how well-supported your answer is by the snippets; and up to 4 "
    "Subject-Predicate-Object triples capturing its key relationships."
)


class TopicAnalysis(BaseModel):
    overview: str
    origin: str
    key_facts: list[str] = Field(default_factory=list, max_length=4)
    credibility: int = Field(ge=0, le=100)
    triples: list[Triple] = Field(default_factory=list, max_length=4)


class TopicAnalysisPipeline:
    """Lightweight single-topic version of the pipeline (no file ingestion)."""

    def __init__(self) -> None:
        settings = get_settings()
        self._llm = ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
            temperature=0.1,
            max_tokens=800,
        ).with_structured_output(TopicAnalysis)

    async def run(self, topic: str) -> TopicAnalysis:
        snippets = await web_search(topic, max_results=6)
        snippet_block = "\n".join(f"- {s['title']}: {s['snippet']}" for s in snippets) or "No search results found."

        try:
            return await self._llm.ainvoke(
                [
                    {"role": "system", "content": _TOPIC_SYSTEM_PROMPT},
                    {"role": "user", "content": f"TOPIC: {topic}\n\nSEARCH SNIPPETS:\n{snippet_block}"},
                ]
            )
        except Exception:
            logger.exception("Topic analysis failed for %r — returning a placeholder", topic)
            return TopicAnalysis(
                overview=f"Could not complete live analysis for '{topic}'.",
                origin="Not available.",
                key_facts=[],
                credibility=0,
                triples=[],
            )
