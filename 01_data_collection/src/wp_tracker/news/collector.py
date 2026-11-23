"""News HTTP access: the sitemap lists every article, article pages give content.

One sitemap request replaces walking ~1 250 listing pages. Only articles that
are new to the snapshot are opened, and every deletion candidate is confirmed
by requesting its URL, so a sitemap glitch can never delete records.
"""
import logging
import random
import time
from urllib.parse import urlsplit
import xml.etree.ElementTree as ET

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import SITEMAP_URL
from .parsing import HOST, canonical_url, parse_article
from .schema import article_row, news_number

log = logging.getLogger(__name__)
SITEMAP_NS = "{http://www.sitemaps.org/schemas/sitemap/0.9}"


class CollectionIncomplete(RuntimeError):
    pass


class Client:
    """Polite pacing, retries for transient server errors, news host only."""
    def __init__(self, delay=1.0):
        self.delay = delay
        self.session = requests.Session()
        retry = Retry(total=4, connect=4, read=3, backoff_factor=1.5,
                      status_forcelist=[429, 500, 502, 503, 504],
                      allowed_methods=["GET"], respect_retry_after_header=True)
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session.headers.update({
            "User-Agent": "UNEC-News-ResearchCollector/1.0 (public academic archive; polite rate)",
            "Accept-Language": "az,en;q=0.6"})

    def close(self):
        self.session.close()

    def get(self, url, *, timeout=(12, 35)):
        """Raw response without following redirects; raises on network failure."""
        if urlsplit(url).hostname != HOST:
            raise CollectionIncomplete("Refusing non-news request")
        time.sleep(random.uniform(self.delay, self.delay * 1.4))
        response = self.session.get(url, timeout=timeout, allow_redirects=False)
        if not response.encoding or response.encoding.lower() == "iso-8859-1":
            response.encoding = "utf-8"
        return response


def sitemap_articles(client, known_urls=None) -> dict[str, str]:
    """Map article number to canonical URL for every article in the sitemap.

    Follows a sitemap index if the site splits it. An article number listed
    under several category URLs keeps the already known URL, otherwise the
    first in sorted order, so identities stay stable.
    """
    known_urls = known_urls or {}
    pending, seen_maps, candidates = [SITEMAP_URL], set(), {}
    while pending:
        url = pending.pop()
        if url in seen_maps:
            continue
        seen_maps.add(url)
        try:
            response = client.get(url, timeout=(12, 120))
            response.raise_for_status()
            root = ET.fromstring(response.content)
        except Exception as exc:
            raise CollectionIncomplete(f"News sitemap unavailable ({url}): {type(exc).__name__}: {exc}") from exc
        if root.tag == SITEMAP_NS + "sitemapindex":
            pending.extend(loc.text.strip() for loc in root.iter(SITEMAP_NS + "loc") if loc.text)
            continue
        if root.tag != SITEMAP_NS + "urlset":
            raise CollectionIncomplete(f"Unexpected sitemap document: {root.tag}")
        for loc in root.iter(SITEMAP_NS + "loc"):
            article = canonical_url((loc.text or "").strip())
            if article:
                candidates.setdefault(news_number(article), set()).add(article)
    articles = {}
    for number, urls in candidates.items():
        known = known_urls.get(number)
        articles[number] = known if known in urls else sorted(urls)[0]
    if not articles:
        raise CollectionIncomplete("News sitemap contains no articles; refusing to treat the site as empty")
    return articles


def fetch_article(client, url) -> dict:
    """Scrape one new article into a record; raises CollectionIncomplete if unusable."""
    try:
        response = client.get(url)
        if response.status_code != 200 or "html" not in response.headers.get("Content-Type", "").lower():
            raise ValueError(f"HTTP {response.status_code}")
        published, title, content = parse_article(BeautifulSoup(response.text, "html.parser"))
    except Exception as exc:
        raise CollectionIncomplete(f"News article unavailable {url}: {type(exc).__name__}: {exc}") from exc
    if not published or not title.strip() or not content.strip():
        raise CollectionIncomplete(f"News article {url} lacks a date, title or content")
    return article_row(url, title, content, published)


def confirm_presence(client, url) -> str:
    """'deleted' only for HTTP 404/410; 'present' for a parseable article; else 'unknown'."""
    try:
        response = client.get(url)
    except Exception as exc:
        log.warning("[NEWS CHECK] %s unreachable: %s", url, exc)
        return "unknown"
    if response.status_code in (404, 410):
        return "deleted"
    if response.status_code == 200 and "html" in response.headers.get("Content-Type", "").lower():
        try:
            _, title, content = parse_article(BeautifulSoup(response.text, "html.parser"))
            if title.strip() and content.strip():
                return "present"
        except Exception:
            pass
    return "unknown"
