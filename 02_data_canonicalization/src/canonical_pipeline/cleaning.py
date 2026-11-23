"""Conservative text extraction retaining paragraphs and list structure."""
import re
import unicodedata
import html as html_module
import warnings
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from ftfy import fix_text

BLOCK_TAGS = {"p", "div", "section", "article", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "ul", "ol", "li", "tr", "table"}
DROP_TAGS = {"script", "style", "noscript", "svg", "iframe", "form"}


def clean_text(value: object) -> str:
    if value is None:
        return ""
    html = fix_text(str(value), unescape_html=False)
    # Decode encoded markup before parsing so its scripts cannot become text.
    for _ in range(3):
        if not re.search(r"&(?:amp;)*(?:lt|#0*60|#x0*3c);\s*/?[a-zA-Z]", html, re.I):
            break
        html = html_module.unescape(html)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MarkupResemblesLocatorWarning)
        soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(DROP_TAGS):
        tag.decompose()
    for tag in soup.find_all("br"):
        tag.replace_with("\n")
    for tag in soup.find_all("td"):
        tag.append("\t")
    for tag in soup.find_all("li"):
        tag.insert(0, "- ")
    for tag in soup.find_all(BLOCK_TAGS):
        tag.insert_before("\n")
        tag.insert_after("\n")
    text = soup.get_text()
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u200b", " ")
    text = "".join(" " if unicodedata.category(ch) == "Zs" else
                   "\n" if unicodedata.category(ch) in {"Zl", "Zp"} else ch for ch in text)
    text = "".join(ch for ch in text if ch in "\n\t" or
                   unicodedata.category(ch) not in {"Cc", "Cf", "Cs"})
    text = unicodedata.normalize("NFC", text).replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def content_rejection(content: str, min_content_chars: int = 6) -> str | None:
    if not content.strip():
        return "Empty content after cleaning"
    tokens = content.split()
    if tokens and all(re.fullmatch(r"(?:https?://|www\.)\S+", token, re.I) for token in tokens):
        return "URL-only content after cleaning"
    if not any(ch.isalnum() for ch in content):
        return "Content without alphanumeric information"
    if len(re.sub(r"\s", "", content)) < min_content_chars:
        return f"Content shorter than {min_content_chars} non-whitespace characters"
    return None
