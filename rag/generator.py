from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from pipeline.enricher import get_gemini_client as get_struct_client

logger = logging.getLogger(__name__)

GEMINI_MODEL_GENERATION = os.getenv("GEMINI_MODEL_GENERATION", "gemini-2.5-flash")


def build_rag_prompt(chunks: List[str], question: str) -> str:
    context = "\n\n---\n\n".join(chunks)
    return f"""\
You are a helpful website assistant for a company. Your goal is to provide clear, perfectly formatted, and highly readable information.

### CONTEXT FROM WEBSITE:
---
{context}
---

### USER QUESTION:
{question}

### RULES FOR RESPONSE:
1.  **Strictly Use Context**: Answer ONLY using the provided metadata and context. Never invent details.
2.  **Visualization & Structure**:
    - Use **Beautiful Markdown**. Use bolding for key terms and professional titles.
    - Use **Clear Lists**: Every item must be on a NEW LINE with a bullet point. NEVER group multiple items on the same line.
    - If there are many items, group them under bolded sub-headings for clarity.
3.  **Contact Information Logic**:
    - If the user asks specifically for **"all contact details"** or **"how to contact"**, provide all locations, phones, and emails found.
    - For **"general"** queries or requests for **"address/location"**, prioritize the **Registered Office Address**, **Primary Emails**, and **Primary Phone Numbers**. Do NOT list every single branch or promotion office unless the user explicitly asks for "all offices".
4.  **Tone**: Professional, friendly, and helpful.
5.  **Fallback**: If the answer isn't in the context, say "I couldn't find that information on the website."

Answer:"""


def generate_answer(question: str, context_chunks: List[str]) -> str:
    """Call Gemini to generate an answer based on context chunks."""
    prompt = build_rag_prompt(context_chunks, question)
    gen_client = get_struct_client()

    try:
        result = gen_client.models.generate_content(
            model=GEMINI_MODEL_GENERATION,
            contents=[prompt],
        )
        return result.text or "Sorry, I couldn't generate a response."  # type: ignore[attr-defined]
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return "Sorry, I encountered an error while generating the response."
