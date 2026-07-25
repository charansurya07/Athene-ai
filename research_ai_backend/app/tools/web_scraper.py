"""
Web scraper tool.

Used by the Ingestion Agent when the request includes a URL, and by the
Searcher Agent when a search result needs its full body (not just the
snippet). Uses httpx + BeautifulSoup for static pages; Playwright can be
swapped in behind the same `fetch_page_text()` signature for JS-heavy pages
without touching any calling agent.
"""
from __future__ import annotations

import logging

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_USER_AGENT = "AtheneAI-ResearchBot/1.0 (+https://athene.ai/bot)"


async def fetch_page_text(url: str, max_chars: int = 8000) -> str:
    """Fetch a URL and return its main readable text, truncated to max_chars."""
    try:
        async with httpx.AsyncClient(
            timeout=15, headers={"User-Agent": _USER_AGENT}, follow_redirects=True
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        return ""

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()

    text = " ".join(soup.get_text(separator=" ").split())
    return text[:max_chars]


async def fetch_page_title(url: str) -> str:
    """Best-effort <title> lookup, used when labeling a source card."""
    try:
        async with httpx.AsyncClient(
            timeout=10, headers={"User-Agent": _USER_AGENT}, follow_redirects=True
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
    except httpx.HTTPError:
        return url

    soup = BeautifulSoup(resp.text, "html.parser")
    return soup.title.string.strip() if soup.title and soup.title.string else url


# NOTE: For JavaScript-rendered pages, swap the implementation above for
# Playwright, e.g.:
#
#   from playwright.async_api import async_playwright
#   async def fetch_page_text_js(url: str) -> str:
#       async with async_playwright() as p:
#           browser = await p.chromium.launch()
#           page = await browser.new_page()
#           await page.goto(url, wait_until="networkidle")
#           text = await page.inner_text("body")
#           await browser.close()
#           return text
#
# Kept out of the default path to avoid requiring a Chromium download for
# every deployment; wire it in behind a feature flag if needed.
