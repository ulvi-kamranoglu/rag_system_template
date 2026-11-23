"""Extraction ported from unec_news_scraper.py; selectors and date rules retained.

Uses the dependency-free BeautifulSoup html.parser backend instead of lxml.
"""
from datetime import datetime
import re
from urllib.parse import urljoin, urlsplit, urlunsplit, parse_qs
from bs4 import BeautifulSoup

BASE_URL = "https://news.unec.edu.az/xeber"

HOST = "news.unec.edu.az"

SOURCE = "website:news.unec.edu.az"

STEP = 8

COLUMNS = ["id", "source_type", "title", "content", "url", "date",
           "modified", "source", "language", "search_text", "metadata"]

ARTICLE_RE = re.compile(r"^/xeber/\d+[^/]*/\d+[^/]*/*$", re.IGNORECASE)

MONTHS = {
    "yan": 1, "yanvar": 1, "jan": 1, "january": 1,
    "fev": 2, "fevral": 2, "feb": 2, "february": 2,
    "mar": 3, "mart": 3, "march": 3,
    "apr": 4, "aprel": 4, "april": 4,
    "may": 5, "mayıs": 5,
    "iyn": 6, "iyun": 6, "jun": 6, "june": 6,
    "iyl": 7, "iyul": 7, "jul": 7, "july": 7,
    "avq": 8, "avqust": 8, "aug": 8, "august": 8,
    "sen": 9, "sentyabr": 9, "sep": 9, "sept": 9, "september": 9,
    "okt": 10, "oktyabr": 10, "oct": 10, "october": 10,
    "noy": 11, "noyabr": 11, "nov": 11, "november": 11,
    "dek": 12, "dekabr": 12, "dec": 12, "december": 12,
}

def normalize_space(text: str) -> str:
    return re.sub(r"[ \t\xa0]+", " ", text or "").strip()

def clean_text(text: str) -> str:
    lines = [normalize_space(s) for s in (text or "").splitlines()]
    return "\n".join(re.sub(r"\s+([.,!?;:])", r"\1", line)
                     for line in lines if line and line not in ("A", "A+", "A-"))

def canonical_url(href: str, base: str = BASE_URL) -> str | None:
    if not isinstance(href, str) or any(char.isspace() for char in href):
        return None
    try:
        u = urlsplit(urljoin(base, href))
    except ValueError:
        return None
    if (u.scheme not in {"http", "https"} or u.username or u.password or
            u.netloc.lower() != HOST or not ARTICLE_RE.fullmatch(u.path)):
        return None
    # Query parameters are not part of a canonical article URL.
    return urlunsplit(("https", HOST, u.path.rstrip("/"), "", ""))

def parse_date(raw: str | None) -> str | None:
    if not raw:
        return None
    text = normalize_space(str(raw)).replace("İ", "i").replace("I", "i")
    m = re.search(r"\b(20\d{2}|19\d{2})[-/.](\d{1,2})[-/.](\d{1,2})(?!\d)", text)
    if m:
        try:
            return datetime(int(m[1]), int(m[2]), int(m[3])).date().isoformat()
        except ValueError:
            pass
    m = re.search(r"\b(\d{1,2})[./-](\d{1,2})[./-]((?:19|20)\d{2})\b", text)
    if m:
        try:
            return datetime(int(m[3]), int(m[2]), int(m[1])).date().isoformat()
        except ValueError:
            pass
    m = re.search(r"\b(\d{1,2})\s+([a-zA-ZəğıöüşçƏĞIİÖÜŞÇ]+)\.?\s+((?:19|20)\d{2})\b", text)
    if m:
        month = MONTHS.get(m[2].lower())
        if month:
            try:
                return datetime(int(m[3]), month, int(m[1])).date().isoformat()
            except ValueError:
                pass
    return None

def listing_links(soup: BeautifulSoup) -> list[tuple[str, str]]:
    # K2 article listing occupies #itemListPrimary / .itemList; avoid site-wide
    # modules (Top news, announcements, etc.) when the primary area is present.
    scope = (soup.select_one(".t3-content .blog") or
             soup.select_one(".t3-content .itemList") or
             soup.select_one("#itemListPrimary") or
             soup.select_one("#k2Container .itemList") or
             soup.select_one(".t3-content") or
             soup.select_one("main") or soup)
    links = scope.select(
        ".catItemTitle a[href], .item-title a[href], "
        "article h2 a[href], article h3 a[href], "
        ".page-header h2 a[href], .blog h2 a[href], "
        ".itemList h2 a[href], .itemList h3 a[href]"
    )
    if not links:
        links = scope.select("a[href]")
    result = []
    seen = set()
    for a in links:
        url = canonical_url(a.get("href", ""))
        if url and url not in seen:
            seen.add(url)
            result.append((url, normalize_space(a.get_text(" ", strip=True))))
    return result

def last_offset(soup: BeautifulSoup) -> int:
    offsets = []
    # Pagination links only (avoid unrelated query parameters).
    scopes = soup.select(".k2Pagination, .pagination, ul.pagination, .page-navigation")
    scope = scopes if scopes else [soup]
    for node in scope:
        for a in node.select('a[href*="start="]'):
            u = urlsplit(urljoin(BASE_URL, a.get("href", "")))
            if u.hostname != HOST or u.path.rstrip("/") != "/xeber":
                continue
            for value in parse_qs(u.query).get("start", []):
                if value.isdecimal():
                    offsets.append(int(value))
    return max(offsets, default=0)

def text_in(node) -> str:
    fragment = BeautifulSoup(str(node), "html.parser")
    for bad in fragment.select(
        "script, style, noscript, svg, form, iframe, "
        ".itemImageBlock, .itemLinks, .itemToolbar, .itemRatingBlock, "
        ".itemSocialSharing, .itemAuthorBlock, .itemAttachmentsBlock, "
        ".itemComments, .addthis_toolbox, .share-buttons, .k2FeedIcon, "
        ".itemExtraFields, .itemTagsBlock, .itemNavigation, .itemRelated"
    ):
        bad.decompose()
    # Keep paragraphs, line breaks, and bullets, while avoiding newline after
    # every inline <b> / <a> / <span> element.
    for br in fragment.select("br"):
        br.replace_with("\n")
    for tag in fragment.select("p, li, h2, h3, h4, tr, blockquote"):
        tag.insert_after("\n")
    return clean_text(fragment.get_text(" ", strip=False))

def article_sections(soup: BeautifulSoup):
    """Return content nodes of the main article, excluding adjacent sidebars.

    The current Joomla template uses section.article-content.clearfix.
    Legacy pages may instead use K2 intro/full text or other article blocks.
    """
    for css in [
        ".t3-content article section.article-content",
        "article section.article-content",
        ".item-page .article-content",
        ".article-content",
    ]:
        node = soup.select_one(css)
        if node:
            return [node]
    intro = soup.select_one("#k2Container .itemIntroText, .itemIntroText")
    full = soup.select_one("#k2Container .itemFullText, .itemFullText")
    if intro or full:
        # Some older templates display the same introduction in both nodes.
        if intro and full and normalize_space(full.get_text(" ", strip=True)).startswith(
                normalize_space(intro.get_text(" ", strip=True))):
            return [full]
        return [n for n in (intro, full) if n]
    for css in [
        '[itemprop="articleBody"]', ".com-content-article__body",
        ".itemBody", ".article-body", ".entry-content", ".post-content",
        "article .content",
    ]:
        node = soup.select_one(css)
        if node:
            return [node]
    return []

def article_title_and_content(soup: BeautifulSoup) -> tuple[str, str]:
    """Move every <strong> from the article text to title (not to content).

    For articles without <strong>, use the page's actual H1 as a fallback title.
    Avoid grabbing site navigation, image carousel, suggestions or sidebars.
    """
    sections = article_sections(soup)
    titles: list[str] = []
    parts: list[str] = []
    seen_titles: set[str] = set()
    for section in sections:
        # Work on a copy; never mutate the soup used for publication-date parsing.
        fragment = BeautifulSoup(str(section), "html.parser")
        for bad in fragment.select(
            "script, style, noscript, svg, form, iframe, "
            ".itemImageBlock, .itemLinks, .itemToolbar, .itemRatingBlock, "
            ".itemSocialSharing, .itemAuthorBlock, .itemAttachmentsBlock, "
            ".itemComments, .addthis_toolbox, .share-buttons, .k2FeedIcon, "
            ".itemExtraFields, .itemTagsBlock, .itemNavigation, .itemRelated, "
            ".article-tools, .social-sharing, .sharethis, .jwplayer"
        ):
            bad.decompose()
        for strong in list(fragment.select("strong")):
            title_piece = clean_text(strong.get_text(" ", strip=True)).replace("\n", " ")
            if title_piece and title_piece not in seen_titles:
                titles.append(title_piece)
                seen_titles.add(title_piece)
            strong.decompose()
        body = text_in(fragment)
        if body:
            parts.append(body)
    title = " ".join(titles).strip()
    if not title:
        for css in [
            ".t3-content article h1", ".item-page .page-header h1",
            ".itemHeader .itemTitle", "article h1", ".itemTitle", "h1",
        ]:
            header = soup.select_one(css)
            if header:
                title = normalize_space(header.get_text(" ", strip=True))
                if title:
                    break
    return title, "\n".join(parts).strip()

def article_body(soup: BeautifulSoup) -> str:
    """Backwards-compatible helper: content without <strong> introductory text."""
    return article_title_and_content(soup)[1]

def article_date(soup: BeautifulSoup) -> str | None:
    for css, attribute in [
        ('meta[property="article:published_time"]', "content"),
        ('meta[name="datePublished"]', "content"),
        ('meta[itemprop="datePublished"]', "content"),
        ('time[datetime]', "datetime"),
        ("#k2Container .itemDateCreated", None),
        (".itemDateCreated", None),
        (".itemHeader .itemDate", None),
        ("article .article-info .published", None),
        ("article .article-info .created", None),
        ("article .article-info time", None),
        ("article .published", None),
        ("article .created", None),
        ("article .article-meta", None),
        ("article .article-aside", None),
        (".t3-content .published", None),
        (".t3-content .article-info", None),
        ('meta[name="date"]', "content"),
    ]:
        node = soup.select_one(css)
        if node:
            raw = node.get(attribute) if attribute else node.get_text(" ", strip=True)
            parsed = parse_date(raw)
            if parsed:
                return parsed
    # Search only main article header, not body/sidebar (dates inside body are misleading).
    for css in [
        "#k2Container .itemHeader", ".itemHeader", "article header",
        ".item-page .article-info", ".t3-content .article-info",
        ".t3-content .page-header",
    ]:
        node = soup.select_one(css)
        if node:
            parsed = parse_date(node.get_text(" ", strip=True))
            if parsed:
                return parsed
    return None

def parse_article(soup: BeautifulSoup) -> tuple[str | None, str, str]:
    title, content = article_title_and_content(soup)
    return article_date(soup), title, content
