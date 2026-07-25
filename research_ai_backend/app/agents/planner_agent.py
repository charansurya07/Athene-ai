"""
STAGE 2 — Planner / Orchestrator Agent
Task decomposition & query formulation.

Breaks the standardized request text into a handful of discrete, parallel-
searchable sub-queries using an LLM with structured (Pydantic) output
parsing — no regex/string-splitting on the model's reply.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

from app.config import get_settings
from app.models.domain import GraphState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are the planning module of a multi-agent research system. Given a "
    "standardized research request, decompose it into 2-5 concise, "
    "independently-searchable sub-queries that together would let a "
    "researcher fully address the request. Avoid redundant sub-queries."
)


class SubQueryPlan(BaseModel):
    sub_queries: list[str] = Field(
        ..., min_length=1, max_length=5, description="Independent, searchable sub-queries"
    )


class PlannerAgent:
    """Stage 2 node: decomposes the request into sub-queries for the Searcher."""

    name = "planner"

    def __init__(self) -> None:
        settings = get_settings()
        self._llm = ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
            temperature=0,
            max_tokens=512,
        ).with_structured_output(SubQueryPlan)

    async def run(self, state: GraphState) -> dict[str, Any]:
        text = state.get("standardized_text", "")
        try:
            plan: SubQueryPlan = await self._llm.ainvoke(
                [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": text[:6000]},
                ]
            )
            sub_queries = plan.sub_queries
        except Exception:
            logger.exception("Planner LLM call failed — falling back to a single sub-query")
            sub_queries = [text[:200]] if text else []

        logger.info("Planner produced %d sub-queries", len(sub_queries))
        return {"sub_queries": sub_queries}
