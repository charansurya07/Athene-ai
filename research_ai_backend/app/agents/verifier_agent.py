"""
STAGE 4 — Time & Credibility Verifier Agent
Fact-checking & confidence scoring.

Applies a time-decay penalty to facts older than 2 years and a per-domain
trust weight, then rolls everything up into a single 0-100 confidence
score plus a per-source credibility score the frontend renders as bars.
"""
from __future__ import annotations

import logging
import re
from datetime import date, datetime
from typing import Any
from urllib.parse import urlparse

from app.models.domain import GraphState, SourceRef

logger = logging.getLogger(__name__)

_DECAY_CUTOFF_YEARS = 2
_DECAY_PENALTY = 25  # points knocked off a source older than the cutoff

# Simple, editable domain trust table — extend as needed. Unknown domains
# default to `_DEFAULT_TRUST`.
_DOMAIN_TRUST: dict[str, int] = {
    "wikipedia.org": 78,
    "nature.com": 96,
    "sciencedirect.com": 92,
    "arxiv.org": 88,
    "gov": 90,
    "edu": 85,
    "reuters.com": 90,
    "apnews.com": 90,
    "bbc.com": 87,
    "nytimes.com": 82,
}
_DEFAULT_TRUST = 60


class VerifierAgent:
    """Stage 4 node: scores every retrieved source and rolls up overall confidence."""

    name = "verifier"

    async def run(self, state: GraphState) -> dict[str, Any]:
        raw_results = state.get("search_results", [])
        if not raw_results:
            logger.info("Verifier received no search results — confidence defaults to 0")
            return {"verified_facts": [], "confidence_score": 0.0, "sources": []}

        scored_sources: list[SourceRef] = []
        for result in raw_results:
            credibility = self._score_source(result)
            scored_sources.append(
                SourceRef(
                    title=result.get("title") or result.get("url", "Untitled source"),
                    url=result.get("url", ""),
                    credibility=credibility,
                    published_at=result.get("published_at"),
                    snippet=result.get("snippet"),
                )
            )

        scored_sources.sort(key=lambda s: s.credibility, reverse=True)
        overall_confidence = round(sum(s.credibility for s in scored_sources) / len(scored_sources), 1)

        logger.info(
            "Verifier scored %d sources — overall confidence %.1f%%",
            len(scored_sources), overall_confidence,
        )
        return {
            "verified_facts": [s.model_dump() for s in scored_sources],
            "confidence_score": overall_confidence,
            "sources": scored_sources,
        }

    def _score_source(self, result: dict[str, Any]) -> float:
        trust = self._domain_trust(result.get("url", ""))
        penalty = self._time_decay_penalty(result.get("published_at"))
        return max(0.0, min(100.0, trust - penalty))

    def _domain_trust(self, url: str) -> int:
        if not url:
            return _DEFAULT_TRUST
        host = urlparse(url).netloc.lower().removeprefix("www.")
        for known_domain, trust in _DOMAIN_TRUST.items():
            if host.endswith(known_domain):
                return trust
        if host.endswith(".gov"):
            return _DOMAIN_TRUST["gov"]
        if host.endswith(".edu"):
            return _DOMAIN_TRUST["edu"]
        return _DEFAULT_TRUST

    def _time_decay_penalty(self, published_at: str | None) -> int:
        published_date = _parse_date(published_at)
        if published_date is None:
            return 0  # unknown date — no penalty, but also no freshness bonus
        age_years = (date.today() - published_date).days / 365.25
        return _DECAY_PENALTY if age_years > _DECAY_CUTOFF_YEARS else 0


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    value = value.strip()
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ", "%B %d, %Y", "%Y"):
        try:
            return datetime.strptime(value[: len(fmt) if fmt != "%Y" else 4], fmt).date()
        except ValueError:
            continue
    match = re.search(r"(\d{4})", value)
    if match:
        return date(int(match.group(1)), 1, 1)
    return None
