from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv("SCRAPED_DATA_DIR", "data/scraped"))


class ScrapedDataStore:
    """
    File-based store for orchestrated scraping.

    Layout:
      data/scraped/{domain}/
      ├── index.json            ← landing page
      ├── nav_links.json        ← header navigation links
      ├── footer_contact.json   ← footer / contact data
      ├── pages/                ← raw scraped page content
      │   ├── about-us.json
      │   └── ...
      └── enriched/             ← Gemini-structured content
          ├── about-us.json
          └── ...
    """

    def __init__(self, domain: str, data_dir: Path | None = None):
        self.domain = domain
        safe = domain.replace(":", "_").replace("/", "_")
        self.root = (data_dir or DATA_DIR) / safe
        self.pages_dir = self.root / "pages"
        self.enriched_dir = self.root / "enriched"

    def _ensure_dirs(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.pages_dir.mkdir(parents=True, exist_ok=True)
        self.enriched_dir.mkdir(parents=True, exist_ok=True)

    def clear(self) -> None:
        """Remove all existing scraped data for this domain."""
        if self.root.exists():
            import shutil
            shutil.rmtree(self.root)
        self._ensure_dirs()

    # ── Single-file writers ──

    def save_index(self, data: Dict[str, Any]) -> Path:
        self._ensure_dirs()
        return self._write(self.root / "index.json", data)

    def save_nav_links(self, links: List[Dict[str, str]]) -> Path:
        self._ensure_dirs()
        return self._write(self.root / "nav_links.json", {"links": links, "count": len(links)})

    def save_footer(self, data: Dict[str, Any]) -> Path:
        self._ensure_dirs()
        return self._write(self.root / "footer_contact.json", data)

    def save_crawl_results(self, data: Dict[str, Any]) -> Path:
        """Save the DomainCrawler output (internal/external/media/failed)."""
        self._ensure_dirs()
        return self._write(self.root / "crawl_results.json", data)


    def save_page(self, slug: str, data: Dict[str, Any]) -> Path:
        self._ensure_dirs()
        safe_slug = slug.strip("/").replace("/", "_") or "index"
        return self._write(self.pages_dir / f"{safe_slug}.json", data)

    def save_enriched(self, slug: str, data: Dict[str, Any]) -> Path:
        """Save Gemini-structured/enriched content for a page."""
        self._ensure_dirs()
        safe_slug = slug.strip("/").replace("/", "_") or "index"
        return self._write(self.enriched_dir / f"{safe_slug}.json", data)

    # ── Readers ──

    def load_nav_links(self) -> List[Dict[str, str]]:
        path = self.root / "nav_links.json"
        if not path.exists():
            return []
        return self._read(path).get("links", [])

    def load_index(self) -> Dict[str, Any]:
        path = self.root / "index.json"
        return self._read(path) if path.exists() else {}

    def load_footer(self) -> Dict[str, Any]:
        path = self.root / "footer_contact.json"
        return self._read(path) if path.exists() else {}

    def load_all_pages(self) -> List[Dict[str, Any]]:
        """Load index + all pages + footer into a flat list for processing."""
        pages: List[Dict[str, Any]] = []

        # Index page
        idx = self.load_index()
        if idx:
            pages.append(idx)

        # Footer as a special page
        footer = self.load_footer()
        if footer and footer.get("text"):
            pages.append({
                "url": f"https://{self.domain}/",
                "slug": "footer-contact",
                "title": "Contact & Footer Information",
                "text": footer.get("text", ""),
                "meta_description": "",
                "contact": footer.get("contact", {}),
            })

        # All scraped pages
        if self.pages_dir.exists():
            for fp in sorted(self.pages_dir.glob("*.json")):
                pages.append(self._read(fp))

        return pages

    def load_all_enriched(self) -> List[Dict[str, Any]]:
        """Load all Gemini-enriched data from the enriched/ directory."""
        enriched: List[Dict[str, Any]] = []
        if self.enriched_dir.exists():
            for fp in sorted(self.enriched_dir.glob("*.json")):
                enriched.append(self._read(fp))
        return enriched

    # ── Internal ──

    def _write(self, path: Path, data: Any) -> Path:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved: {path}")
        return path

    def _read(self, path: Path) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
