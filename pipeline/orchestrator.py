import asyncio
import logging
from typing import Any, Callable, Coroutine, Dict, List, Optional
from urllib.parse import urlparse

from pipeline.scraper import fetch_html, extract_body_text
from pipeline.enricher import get_gemini_client as get_struct_client, structure_page_content
from pipeline.data_store import ScrapedDataStore
from rag.retriever import process_and_index, reset_store

logger = logging.getLogger(__name__)

# Type alias for async progress callbacks
ProgressCallback = Optional[Callable[[str, int, int], Coroutine[Any, Any, None]]]


async def run_ingestion_pipeline(
    domain: str,
    urls: List[str],
    model: str = "gemini-2.5-flash",
    progress_cb: ProgressCallback = None,
) -> Dict[str, Any]:
    """
    Step-by-step pipeline: Scrape -> Enrich -> Index.
    If progress_cb is provided, it is called as:
        await progress_cb(phase, current_item, total_items)
    phase is one of: 'scraping', 'enriching', 'indexing'
    """
    store = ScrapedDataStore(domain)
    reset_store(domain)

    total = len(urls)

    # PHASE A: SCRAPING
    logger.info(f"🕷️ PHASE A: SCRAPING {total} pages for {domain}")
    pages_scraped = 0
    pages_failed = 0

    if progress_cb:
        await progress_cb("scraping", 0, total)

    for i, url in enumerate(urls, 1):
        try:
            logger.info(f"  [{i}/{total}] Scraping: {url}")
            page_url, page_html = await fetch_html(url)
            page_data = extract_body_text(page_html, page_url)
            page_data["url"] = page_url

            slug = urlparse(page_url).path.strip("/").replace("/", "_") or "index"
            page_data["slug"] = slug

            if page_data["text"].strip():
                store.save_page(slug, page_data)
                pages_scraped += 1
            else:
                logger.warning(f"    ⚠️ Empty content, skipping")
        except Exception as e:
            logger.warning(f"    ❌ Failed to scrape {url}: {e}")
            pages_failed += 1

        if progress_cb:
            await progress_cb("scraping", i, total)

        await asyncio.sleep(0.3)

    # PHASE B: ENRICHMENT
    logger.info("🧠 PHASE B: ENRICHMENT")
    all_pages = store.load_all_pages()
    struct_client = get_struct_client()
    enrich_total = len(all_pages)

    if progress_cb:
        await progress_cb("enriching", 0, enrich_total)

    pages_enriched = 0
    for i, page in enumerate(all_pages, 1):
        page_text = page.get("text", "")
        if not page_text.strip():
            continue
        try:
            structured = structure_page_content(
                client=struct_client,
                page_title=page.get("title") or page.get("label") or "Untitled",
                page_text=page_text,
                model=model,
            )
            slug = page.get("slug", "") or "page"
            store.save_enriched(slug, {
                "url": page.get("url"),
                "title": page.get("title"),
                "structured": structured,
            })
            pages_enriched += 1
        except Exception as e:
            logger.warning(f"    ❌ Failed to enrich page: {e}")
            continue

        if progress_cb:
            await progress_cb("enriching", i, enrich_total)

        if i < enrich_total:
            await asyncio.sleep(1)

    # PHASE C: INDEXING
    logger.info("📑 PHASE C: INDEXING")
    enriched_data = store.load_all_enriched()
    index_total = len(enriched_data)

    if progress_cb:
        await progress_cb("indexing", 0, index_total)

    indexed_count = await process_and_index(domain, enriched_data)

    if progress_cb:
        await progress_cb("indexing", index_total, index_total)

    return {
        "pages_scraped": pages_scraped,
        "pages_enriched": pages_enriched,
        "pages_failed": pages_failed,
        "chunks_indexed": indexed_count,
    }
