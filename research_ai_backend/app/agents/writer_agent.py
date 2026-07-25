"""
STAGE 7 — Writer & Synthesizer Agent
Output assembly & citation formatting.

Final node in the graph: pulls together the standardized request, verified
sources, recommendations and extracted triples into the short executive
report string the frontend's Output tab renders.
"""
from __future__ import annotations

import logging
from typing import Any

from langchain_anthropic import ChatAnthropic

from app.config import get_settings
from app.models.domain import GraphState

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are the writer module of a multi-agent research system. Write a "
    "concise executive report (2-4 short paragraphs, plain prose, no "
    "markdown headers) that answers the user's original request using the "
    "verified sources and recommendations provided. Reference sources by "
    "name inline where relevant rather than using footnote markers."
)


class WriterAgent:
    """Stage 7 node: assembles the final human-readable report."""

    name = "writer"

    def __init__(self) -> None:
        settings = get_settings()
        self._llm = ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
            temperature=0.3,
            max_tokens=700,
        )

    async def run(self, state: GraphState) -> dict[str, Any]:
        context = self._build_context(state)
        try:
            response = await self._llm.ainvoke(
                [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": context[:9000]},
                ]
            )
            report = response.content if isinstance(response.content, str) else str(response.content)
        except Exception:
            logger.exception("Writer LLM call failed — falling back to a templated summary")
            report = self._fallback_report(state)

        logger.info("Writer agent produced a %d-character report", len(report))
        return {"report": report.strip()}

    def _build_context(self, state: GraphState) -> str:
        sources = state.get("sources", [])
        recs = state.get("recommendations", [])
        source_lines = "\n".join(f"- {s.title} ({s.url}) — {s.credibility:.0f}% credible" for s in sources[:6])
        rec_lines = "\n".join(f"- {r.claim} -> {r.alternative}" for r in recs)
        return (
            f"ORIGINAL REQUEST:\n{state.get('standardized_text', '')}\n\n"
            f"OVERALL CONFIDENCE: {state.get('confidence_score', 0)}%\n\n"
            f"SOURCES:\n{source_lines or 'none'}\n\n"
            f"RECOMMENDATIONS:\n{rec_lines or 'none'}"
        )

    def _fallback_report(self, state: GraphState) -> str:
        confidence = state.get("confidence_score", 0)
        n_sources = len(state.get("sources", []))
        return (
            f"Athene AI gathered {n_sources} source(s) for this request and reached an overall "
            f"confidence of {confidence:.0f}%. The live writing step could not be reached — "
            f"the raw sources, recommendations and knowledge graph in the other tabs are still "
            f"based on live retrieval and are safe to use."
        )
