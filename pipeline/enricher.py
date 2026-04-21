from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from google import genai

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Structuring prompt
# ---------------------------------------------------------------------------

STRUCTURING_PROMPT = """\
You are a data structuring engine for a website chatbot.
Analyze the following website page content and extract ALL useful information.

Return a JSON object with this exact schema:
{{
  "page_title": "the page title",
  "page_type": "product | service | blog | faq | policy | about | contact | general",
  "summary": "a 2-3 sentence summary of the page",
  "main_topics": ["topic1", "topic2"],
  "key_points": ["detailed point 1", "detailed point 2", "..."],
  "services_or_products": ["item 1", "item 2"],
  "faqs": [{{"question": "Q?", "answer": "A."}}]
}}

IMPORTANT RULES:
- Extract EVERY piece of useful information as key_points. Be thorough.
- Each key_point should be a complete, self-contained sentence.
- Include company details, product names, features, contact info, addresses, etc.
- If the page mentions services or products, list them in services_or_products.
- If the page has Q&A style content, extract them as faqs.
- Only return valid JSON. No explanations outside JSON.

Page Title: {page_title}

Content:
---
{page_text}
---
"""


def get_gemini_client(api_key: str | None = None) -> genai.Client:
    """
    Instantiate a Gemini client. The api_key can be supplied directly or via
    the GOOGLE_API_KEY environment variable, which the client will pick up.
    """
    if api_key:
        return genai.Client(api_key=api_key)
    return genai.Client()


# ---------------------------------------------------------------------------
# Structuring with Gemini
# ---------------------------------------------------------------------------


def structure_page_content(
    client: genai.Client,
    page_title: str,
    page_text: str,
    model: str = "gemini-2.5-flash",
) -> Dict[str, Any]:
    """
    Call Gemini to turn raw page text into structured JSON.
    Falls back to raw text data if Gemini fails.
    """
    try:
        prompt = STRUCTURING_PROMPT.format(
            page_title=page_title,
            page_text=page_text[:20000],
        )

        result = client.models.generate_content(
            model=model,
            contents=[prompt],
            config={"response_mime_type": "application/json"},
        )

        data = result.parsed  # type: ignore[attr-defined]
        if data is None:
            # Try extracting JSON from the text response
            text_resp = getattr(result, "text", "") or ""
            data = _try_parse_json(text_resp)

        data = data or {}

    except Exception as e:
        logger.warning(f"Gemini structuring failed for '{page_title}': {e}")
        data = {}

    # Ensure all fields exist
    data.setdefault("page_title", page_title)
    data.setdefault("page_type", "general")
    data.setdefault("summary", "")
    data.setdefault("main_topics", [])
    data.setdefault("key_points", [])
    data.setdefault("services_or_products", [])
    data.setdefault("faqs", [])
    data["_raw_text"] = page_text  # always preserve raw text

    return data


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

MAX_CHUNK_CHARS = 2000   # ~500 tokens
OVERLAP_CHARS = 200      # overlap between consecutive raw-text chunks


def chunk_structured_page(
    page_url: str, structured: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Turn a structured page into semantic chunks suitable for embeddings.

    Strategy:
    1. Create a chunk from the summary + main_topics.
    2. Create a chunk from key_points.
    3. Create individual chunks per FAQ.
    4. Create individual chunks per service/product.
    5. FALLBACK: if no chunks were created, split raw text into overlapping windows.
    """
    chunks: List[Dict[str, Any]] = []
    page_type = structured.get("page_type", "general")
    page_title = structured.get("page_title", "")
    base_meta = {"page_title": page_title, "page_type": page_type}

    # 1. Summary chunk
    summary = (structured.get("summary") or "").strip()
    topics = structured.get("main_topics") or []
    if summary:
        summary_text = f"Page: {page_title}\nSummary: {summary}"
        if topics:
            summary_text += "\nTopics: " + ", ".join(topics)
        chunks.append(_make_chunk(page_url, page_type, summary_text, base_meta))

    # 2. Key points chunk (may split if too long)
    key_points = structured.get("key_points") or []
    if key_points:
        kp_text = "\n".join(f"• {kp}" for kp in key_points)
        if len(kp_text) > MAX_CHUNK_CHARS:
            # Split key points into multiple chunks
            current_batch: List[str] = []
            current_len = 0
            for kp in key_points:
                line = f"• {kp}"
                if current_len + len(line) > MAX_CHUNK_CHARS and current_batch:
                    chunks.append(_make_chunk(
                        page_url, page_type,
                        f"Key information from {page_title}:\n" + "\n".join(current_batch),
                        base_meta,
                    ))
                    current_batch = []
                    current_len = 0
                current_batch.append(line)
                current_len += len(line) + 1
            if current_batch:
                chunks.append(_make_chunk(
                    page_url, page_type,
                    f"Key information from {page_title}:\n" + "\n".join(current_batch),
                    base_meta,
                ))
        else:
            chunks.append(_make_chunk(
                page_url, page_type,
                f"Key information from {page_title}:\n{kp_text}",
                base_meta,
            ))

    # 3. Services / products
    services = structured.get("services_or_products") or []
    if services:
        svc_text = f"Services/Products from {page_title}:\n" + "\n".join(f"• {s}" for s in services)
        chunks.append(_make_chunk(page_url, "service", svc_text, base_meta))

    # 4. FAQs
    for faq in structured.get("faqs") or []:
        q = (faq.get("question") or "").strip()
        a = (faq.get("answer") or "").strip()
        if q and a:
            chunks.append(_make_chunk(
                page_url, "faq",
                f"Q: {q}\nA: {a}",
                {**base_meta, "faq": True},
            ))

    # 5. NO FALLBACK — we only use enriched data now.
    # If no chunks were created, this page contributes nothing to the RAG index.
    if not chunks:
        logger.warning(f"⚠️ No structured chunks extracted for: {page_url}. Skipping RAG index for this page.")

    return chunks


def _make_chunk(
    source: str, chunk_type: str, content: str, metadata: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "source": source,
        "type": chunk_type,
        "content": content,
        "metadata": metadata,
    }


def _split_raw_text(
    page_url: str,
    page_type: str,
    text: str,
    base_meta: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Split raw text into overlapping windows for embedding."""
    chunks: List[Dict[str, Any]] = []
    start = 0
    part = 1
    while start < len(text):
        end = start + MAX_CHUNK_CHARS
        chunk_text = text[start:end]
        if chunk_text.strip():
            chunks.append(_make_chunk(
                page_url,
                page_type,
                chunk_text.strip(),
                {**base_meta, "chunk_part": part, "note": "raw_text_chunk"},
            ))
            part += 1
        start = end - OVERLAP_CHARS  # overlap
    return chunks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _try_parse_json(text: str) -> Dict[str, Any] | None:
    """Try to parse JSON from text, even if wrapped in markdown code fences."""
    text = text.strip()
    # Strip code fences
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
        text = text.strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
