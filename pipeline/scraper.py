"""
Page scraper — fetches HTML and extracts clean body text.

Active functions:
  - fetch_html     — async HTTP GET, returns (final_url, raw_html)
  - extract_body_text — strips nav/header/footer, returns {title, meta_description, text}

Archived functions (used by the old orchestrator, no longer needed):
  - extract_nav_links   — replaced by DomainCrawler
  - extract_footer_data — replaced by DomainCrawler + enrichment
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup, NavigableString, Tag

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTTP defaults
# ---------------------------------------------------------------------------
HTTP_TIMEOUT = 30
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

JUNK_TAGS = ["script", "style", "noscript", "iframe", "svg", "canvas", "video", "audio"]
MIN_LINE_LENGTH = 3

# Lines that are common UI noise — stripped from every page's body text
NOISE_LINES = frozenset([
    "skip to content",
    "toggle content",
    "previous",
    "next",
    "search",
    "close",
    "menu",
    "toggle navigation",
    "open menu",
    "close menu",
    "back to top",
])

# Patterns that indicate UI/CMS noise (matched as substrings, case-insensitive)
NOISE_PATTERNS = [
    "facebook-f",
    "instagram linkedin",
    "rank math",
    "wordpress seo",
    "search engine optimization by",
    "click to chat",
    "holithemes.com",
    "#inner-wrap",
    "#wrapper",
    "© 20",
    "all rights reserved",
    "terms and conditions",
    "privacy policy",
]


# ---------------------------------------------------------------------------
# 1. Fetch raw HTML
# ---------------------------------------------------------------------------


async def fetch_html(url: str) -> tuple[str, str]:
    """Fetch a page and return (final_url, raw_html)."""
    async with httpx.AsyncClient(
        timeout=HTTP_TIMEOUT, verify=False,
        follow_redirects=True, headers=HTTP_HEADERS,
    ) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return str(resp.url), resp.text


# ---------------------------------------------------------------------------
# 2. Extract full page body text
# ---------------------------------------------------------------------------


def extract_body_text(html: str, base_url: str = "") -> Dict[str, str]:
    """
    Extract title, meta description, and ONLY the unique page content.
    Strips nav, header, footer, and menu elements. Also extracts main images to text.
    """
    soup = BeautifulSoup(html, "html.parser")

    # Remove junk tags
    for tag_name in JUNK_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Title (grab before removing elements)
    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    # Meta description
    meta_desc = ""
    meta_tag = soup.find("meta", attrs={"name": re.compile(r"description", re.I)})
    if meta_tag and isinstance(meta_tag, Tag):
        meta_desc = (meta_tag.get("content", "") or "").strip()  # type: ignore[arg-type]

    # ── Strip repeated site-wide elements ──
    for tag in soup.find_all("nav"):
        tag.decompose()
    for tag in soup.find_all("header"):
        tag.decompose()
    for tag in soup.find_all("footer"):
        tag.decompose()

    # Remove only top-level site wrappers with EXACT id matches
    for exact_id in ["site-header", "site-footer", "masthead", "colophon"]:
        tag = soup.find(id=exact_id)
        if tag:
            tag.decompose()

    # Extract ONLY the remaining page-specific content
    body = soup.find("body")
    body_text = _extract_all_visible_text(body, base_url) if body else _extract_all_visible_text(soup, base_url)

    return {
        "title": title,
        "meta_description": meta_desc,
        "text": body_text,
    }


# ---------------------------------------------------------------------------
# Text extraction — full DOM walk
# ---------------------------------------------------------------------------

BLOCK_TAGS = frozenset([
    "p", "div", "section", "article", "main", "aside",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "tr", "dt", "dd",
    "blockquote", "figcaption", "caption",
    "header", "footer", "nav",
    "br", "hr", "table", "thead", "tbody", "tfoot",
    "ul", "ol", "dl", "form", "fieldset",
    "pre", "code", "address",
])


def _extract_all_visible_text(element: Tag, base_url: str) -> str:
    """Walk DOM tree, extract ALL visible text with structural line breaks.
    Filters out UI noise lines (Skip to content, social labels, CMS comments, etc.)."""
    lines: List[str] = []
    _walk_tree(element, lines, base_url)

    seen: str = ""
    clean: List[str] = []
    for line in lines:
        s = line.strip()
        if not s or len(s) < MIN_LINE_LENGTH or s == seen:
            continue
        if s.lower() in NOISE_LINES:
            continue
        lower = s.lower()
        if any(pat in lower for pat in NOISE_PATTERNS):
            continue
        if s.startswith("html ") or s.startswith("<!--") or s.startswith("/"):
            continue
        seen = s
        clean.append(s)

    return "\n".join(clean)


def _walk_tree(node, lines: List[str], base_url: str) -> None:
    if isinstance(node, NavigableString):
        text = " ".join(str(node).split()).strip()
        if text:
            if lines and lines[-1] and not lines[-1].endswith("\n"):
                lines[-1] += " " + text
            else:
                lines.append(text)
        return

    if not isinstance(node, Tag):
        return

    style = node.get("style", "")
    if isinstance(style, str) and ("display:none" in style.replace(" ", "") or "visibility:hidden" in style.replace(" ", "")):
        return

    tag_name = (node.name or "").lower()
    if tag_name == "img":
        src = node.get("src")
        if src and not isinstance(src, list):
            alt = node.get("alt", "").strip() if getattr(node, "get", None) else ""
            if not ("1x1" in src or "pixel" in src or "track" in src):
                abs_url = urljoin(base_url, src) if base_url else src
                img_text = f"[Image: {alt}]({abs_url})"
                if lines and lines[-1] and not lines[-1].endswith("\n"):
                    lines[-1] += " " + img_text
                else:
                    lines.append(img_text)
        return

    is_block = tag_name in BLOCK_TAGS

    if is_block and lines and lines[-1]:
        lines.append("")

    for child in node.children:
        _walk_tree(child, lines, base_url)

    if is_block and lines and lines[-1]:
        lines.append("")


# ---------------------------------------------------------------------------
# ARCHIVED — Unused functions from old orchestrator pipeline
# ---------------------------------------------------------------------------
# The following functions were part of the old /ingest flow and are no longer
# used. They are kept here for reference but should not be imported.
#
# def extract_nav_links(html: str, page_url: str) -> List[Dict[str, str]]:
#     """Extract all navigation links from <nav>, <header>, and menu elements."""
#     ...
#
# def extract_footer_data(html: str, page_url: str) -> Dict[str, Any]:
#     """Extract contact info, address, phone, email, and social links from footer."""
#     ...
#
# See archive/orchestrator.py for the full implementation context.
