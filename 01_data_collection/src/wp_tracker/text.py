"""Small, consistent text projection for change detection (not canonicalization)."""
from __future__ import annotations

import html
import re
import unicodedata

from bs4 import BeautifulSoup, NavigableString, Tag

BLOCKS = frozenset({
    "p", "div", "section", "article", "h1", "h2", "h3", "h4", "h5", "h6",
    "li", "ul", "ol", "table", "tr", "blockquote", "pre", "header", "footer",
})
HIDDEN = ("script", "style", "noscript", "iframe", "svg", "template")


def normalize_text(value: object, preserve_paragraphs: bool = True) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFC", html.unescape(str(value)))
    text = text.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")
    if not preserve_paragraphs:
        return re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def html_to_text(value: object, preserve_paragraphs: bool = True) -> str:
    """Match the previous v2 text behavior for more reliable baseline imports."""
    raw = "" if value is None else str(value)
    # Plain text, including standalone URLs, should not be parsed as HTML.
    if "<" not in raw:
        return normalize_text(raw, preserve_paragraphs)
    soup = BeautifulSoup(raw, "html.parser")
    for element in soup.find_all(HIDDEN):
        element.decompose()
    chunks: list[str] = []

    def visit(node: Tag | NavigableString) -> None:
        if isinstance(node, NavigableString):
            chunks.append(str(node))
        elif isinstance(node, Tag):
            if node.name == "br":
                chunks.append("\n" if preserve_paragraphs else " ")
                return
            boundary = node.name in BLOCKS
            if boundary:
                chunks.append("\n" if preserve_paragraphs else " ")
            for child in node.children:
                visit(child)
            if boundary:
                chunks.append("\n" if preserve_paragraphs else " ")

    for child in soup.contents:
        visit(child)
    return normalize_text("".join(chunks), preserve_paragraphs)


def comparable(value: object) -> str:
    """Ignore markup/whitespace-only changes, preserve text and case changes."""
    return normalize_text(html_to_text(value, preserve_paragraphs=False), preserve_paragraphs=False)


MAX_SHORT_CONTENT_CHARS = 5
_URL_TOKEN = re.compile(r"(?:https?://|www\.)[^\s<>]+", flags=re.IGNORECASE)


def initial_cleaning_rejection(content: str) -> str | None:
    """Classify unusable new-record content after HTML-to-text conversion.

    The count excludes whitespace, while paragraphs and the original normalized
    content remain unchanged. The caller must never remove these WordPress IDs
    from the full snapshot used for deletion detection.
    """
    text = normalize_text(content)
    if not text:
        return "empty_content"
    if all(_URL_TOKEN.fullmatch(token) for token in text.split()):
        return "url_only_content"
    if len(re.sub(r"\s+", "", text)) <= MAX_SHORT_CONTENT_CHARS:
        return "short_content"
    return None
