"""Exhaustive WordPress REST API scans; incomplete scans are rejected."""
from __future__ import annotations

import logging
import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import Settings

LOG = logging.getLogger(__name__)
SOURCES = ("pages", "posts")


class WordPressClient:
    def __init__(self, settings: Settings, session: requests.Session | None = None):
        self.settings = settings
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": settings.user_agent, "Accept": "application/json"})
        retry = Retry(
            total=settings.retries,
            connect=settings.retries,
            read=settings.retries,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            respect_retry_after_header=True,
        )
        if session is None:
            self.session.mount("https://", HTTPAdapter(max_retries=retry))

    def fetch_all(self, source: str) -> list[dict[str, Any]]:
        if source not in SOURCES:
            raise ValueError(f"Unsupported source: {source}")
        url = f"{self.settings.base_url}/wp-json/wp/v2/{source}"
        items: list[dict[str, Any]] = []
        ids: set[int] = set()
        total_pages: int | None = None
        expected_total: int | None = None
        page = 1
        while True:
            if page > self.settings.max_pages_per_source:
                raise RuntimeError(f"Scan exceeded max_pages_per_source for {source}")
            response = self.session.get(
                url,
                params={
                    "page": page,
                    "per_page": self.settings.per_page,
                    "status": "publish",
                    "orderby": "id",
                    "order": "asc",
                },
                timeout=self.settings.timeout_seconds,
            )
            response.raise_for_status()
            if total_pages is None:
                try:
                    total_pages = int(response.headers["X-WP-TotalPages"])
                    expected_total = int(response.headers["X-WP-Total"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise RuntimeError(f"WordPress pagination headers missing for {source}") from exc
                if total_pages < 1 or expected_total < 0:
                    raise RuntimeError(f"Invalid WordPress pagination headers for {source}")
                if total_pages > self.settings.max_pages_per_source:
                    raise RuntimeError(f"Too many WordPress pages for {source}: {total_pages}")
            batch = response.json()
            if not isinstance(batch, list):
                raise RuntimeError(f"Unexpected WordPress response for {source}, page {page}")
            for item in batch:
                if not isinstance(item, dict) or type(item.get("id")) is not int or item["id"] <= 0:
                    raise RuntimeError(f"Invalid WordPress ID in {source}, page {page}")
                wp_id = item["id"]
                if wp_id in ids:
                    raise RuntimeError(f"Duplicate WordPress ID {wp_id} in {source}; retry later")
                ids.add(wp_id)
                items.append(item)
            LOG.info("Fetched %s page %s/%s: %s items", source, page, total_pages, len(batch))
            if page >= total_pages:
                break
            page += 1
            if self.settings.delay_seconds:
                time.sleep(self.settings.delay_seconds)
        if len(items) != expected_total:
            raise RuntimeError(
                f"Incomplete {source} scan: expected {expected_total} unique items, "
                f"received {len(items)}. No changes will be persisted."
            )
        LOG.info("Complete scan for %s: %s published records", source, len(items))
        return items
