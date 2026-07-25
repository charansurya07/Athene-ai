"""
STAGE 5 — Recommendation & Comparison Agent
Advisory & alternative synthesis.

Flags claims in the original request that look outdated or weakly
supported given what the Searcher/Verifier found, proposes a stronger
alternative for each, and produces the `input_score` vs `recommended_score`
comparison the frontend renders at the bottom of the Output tab.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, Field

from app.config import get_settings
from app.models.domain import GraphState, Recommendation

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are the recommendation module of a multi-agent research system. "
    "You will be given the user's original request plus verified source "
    "snippets. Identify up to 4 claims in the request that are outdated, "
    "unsupported, or could be improved, and for each propose a concrete, "
    "currently-supported alternative. Then give two overall 0-100 scores: "
    "`input_score` (how well-supported the request's original claims are) "
    "and `recommended_score` (how well-supported the proposed alternatives "
    "are). If nothing needs correcting, return an empty recommendations "
    "list and set both scores based on how solid the original request is."
)


class RecommendationOutput(BaseModel):
    recommendations: list[Recommendation] = Field(default_factory=list, max_length=4)
    input_score: int = Field(ge=0, le=100)
    recommended_score: int = Field(ge=0, le=100)


class RecommendationAgent:
    """Stage 5 node: builds the comparison matrix and the two headline scores."""

    name = "recommendation"

    def __init__(self) -> None:
        settings = get_settings()
        self._llm = ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
            temperature=0.2,
            max_tokens=900,
        ).with_structured_output(RecommendationOutput)

    async def run(self, state: GraphState) -> dict[str, Any]:
        context = self._build_context(state)
        try:
            output: RecommendationOutput = await self._llm.ainvoke(
                [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": context[:8000]},
                ]
            )
        except Exception:
            logger.exception("Recommendation LLM call failed — falling back to neutral scores")
            confidence = state.get("confidence_score", 50.0)
            return {
                "recommendations": [],
                "input_score": confidence,
                "recommended_score": confidence,
            }

        logger.info(
            "Recommendation agent: %d suggestions, input=%d recommended=%d",
            len(output.recommendations), output.input_score, output.recommended_score,
        )
        return {
            "recommendations": output.recommendations,
            "input_score": float(output.input_score),
            "recommended_score": float(output.recommended_score),
        }

    def _build_context(self, state: GraphState) -> str:
        request_text = state.get("standardized_text", "")
        sources = state.get("sources", [])
        snippets = "\n".join(
            f"- ({s.credibility:.0f}% credible, {s.published_at or 'undated'}) {s.title}: {s.snippet or ''}"
            for s in sources[:8]
        )
        return f"ORIGINAL REQUEST:\n{request_text}\n\nVERIFIED SOURCE SNIPPETS:\n{snippets}"
