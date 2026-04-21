"""
Domain crawler — discovers all internal, external, and media links
by recursively following <a> tags within the same brand domain.
"""
import requests
import time
import tldextract
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin, urldefrag
from collections import deque


class DomainCrawler:
    MEDIA_EXTENSIONS = {
        ".jpg", ".jpeg", ".png", ".gif", ".svg", ".webp",
        ".mp4", ".avi", ".mov", ".wmv", ".flv", ".mkv",
        ".mp3", ".wav",
        ".pdf", ".doc", ".docx", ".xls", ".xlsx",
        ".zip", ".rar", ".7z",
        ".ico"
    }

    def __init__(self, start_url, options="all", max_pages=500, delay=0):
        self.start_url = self.normalize(start_url)

        # Extract brand domain using tldextract
        parsed = urlparse(self.start_url)
        extracted = tldextract.extract(parsed.netloc)
        self.brand_domain = extracted.domain

        self.options = options
        self.max_pages = max_pages
        self.delay = delay

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (ProductionDomainCrawler/1.0)"
        })

        self.visited = set()
        self.internal_pages = set()
        self.external_links = set()
        self.media_links = set()
        self.failed_links = set()

        self.queue = deque([(self.start_url, 0)])

    # -----------------------------
    # Internal Detection (Cross-TLD Safe)
    # -----------------------------
    def is_internal(self, url):
        parsed = urlparse(url)
        extracted = tldextract.extract(parsed.netloc)
        return extracted.domain == self.brand_domain

    # -----------------------------
    # URL Normalization
    # -----------------------------
    def normalize(self, url):
        url = urldefrag(url)[0]
        parsed = urlparse(url)

        scheme = parsed.scheme or "https"
        netloc = parsed.netloc.lower()
        path = parsed.path.rstrip("/")

        return f"{scheme}://{netloc}{path}"

    # -----------------------------
    # Media Check
    # -----------------------------
    def is_media(self, url):
        path = urlparse(url).path.lower()
        return any(path.endswith(ext) for ext in self.MEDIA_EXTENSIONS)

    # -----------------------------
    # Extract Links
    # -----------------------------
    def extract_links(self, url):
        try:
            response = self.session.get(url, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            links = []

            for tag in soup.find_all("a", href=True):
                href = tag["href"].strip()

                if href.startswith(("mailto:", "tel:", "javascript:", "#")):
                    continue

                absolute = urljoin(url, href)
                normalized = self.normalize(absolute)

                links.append(normalized)

            return links

        except Exception:
            self.failed_links.add(url)
            return []

    # -----------------------------
    # Core Crawl
    # -----------------------------
    def crawl(self):
        print(f"\n🚀 Starting Crawl: {self.start_url}")
        print(f"🏷 Brand Domain: {self.brand_domain}\n")

        while self.queue and len(self.visited) < self.max_pages:
            current_url, depth = self.queue.popleft()

            if current_url in self.visited:
                continue

            self.visited.add(current_url)

            if self.is_media(current_url):
                self.media_links.add(current_url)
                continue

            if not self.is_internal(current_url):
                self.external_links.add(current_url)
                continue

            self.internal_pages.add(current_url)
            print(f"🔍 Crawling ({len(self.visited)}): {current_url}")

            links = self.extract_links(current_url)
            time.sleep(self.delay)

            for link in links:
                if link in self.visited:
                    continue

                if self.is_media(link):
                    self.media_links.add(link)
                elif self.is_internal(link):
                    self.queue.append((link, depth + 1))
                else:
                    self.external_links.add(link)

        print("\n✅ Crawl Finished")
        print(f"Internal Pages: {len(self.internal_pages)}")
        print(f"External Links: {len(self.external_links)}")
        print(f"Media Links: {len(self.media_links)}")
        print(f"Failed Pages: {len(self.failed_links)}")

        return self.export_results()

    # -----------------------------
    # Export Results
    # -----------------------------
    def export_results(self):
        return {
            "domain": self.brand_domain,
            "internal": sorted(self.internal_pages),
            "external": sorted(self.external_links),
            "media": sorted(self.media_links),
            "failed": sorted(self.failed_links),
        }
