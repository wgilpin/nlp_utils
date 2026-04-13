"""HTML parsing utilities."""

import logging
from datetime import date

import markdownify as md
from bs4 import BeautifulSoup
from dateutil import parser as dateutil_parser

logger = logging.getLogger(__name__)

_NOISE_TAGS: frozenset[str] = frozenset(
    {"nav", "aside", "header", "footer", "script", "style", "iframe", "noscript"}
)


def extract_article(html: str) -> str:
    """Return markdown of <article>, falling back to <main> then <body>.

    Strips noise elements (nav, aside, header, footer, script, style, etc.)
    before converting the remaining HTML to markdown.
    """
    soup = BeautifulSoup(html, "lxml")
    tag = soup.find("article") or soup.find("main") or soup.find("body") or soup
    chosen = tag.name if hasattr(tag, "name") else "document"
    logger.debug("Extracting text from <%s>", chosen)
    for tag_name in _NOISE_TAGS:
        for noise in tag.find_all(tag_name):
            noise.decompose()
    return md.markdownify(str(tag), heading_style="ATX")


def extract_clean_html(html: str) -> str:
    """Strip noise tags and return clean body text.

    Removes: nav, aside, header, footer, script, style, iframe, noscript.
    """
    soup = BeautifulSoup(html, "lxml")
    for tag_name in _NOISE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()
    logger.debug("Removed noise tags: %s", ", ".join(_NOISE_TAGS))
    body = soup.find("body") or soup
    return body.get_text(separator="\n", strip=True)


def extract_publication_date(html: str) -> date | None:
    """Extract the publication date from an HTML page.

    Checks (in order):
    1. ``data-testid="storyPublishDate"`` span (Medium)
    2. ``<meta property="article:published_time">``
    3. ``<time datetime=...>``

    Returns a :class:`datetime.date` or ``None`` if nothing is found / parseable.
    """
    soup = BeautifulSoup(html, "lxml")

    candidates: list[str] = []

    el = soup.find(attrs={"data-testid": "storyPublishDate"})
    if el:
        candidates.append(el.get_text(strip=True))

    meta = soup.find("meta", property="article:published_time")
    if meta and meta.get("content"):
        candidates.append(meta["content"])

    time_el = soup.find("time", datetime=True)
    if time_el:
        candidates.append(time_el["datetime"])

    for raw in candidates:
        try:
            return dateutil_parser.parse(raw, fuzzy=True).date()
        except (ValueError, OverflowError):
            logger.debug("Could not parse date candidate %r", raw)

    return None


def html_to_markdown(html: str) -> str:
    """Convert HTML to Markdown using markdownify.

    Uses ATX-style headings (# H1, ## H2, …). Script and style elements
    are removed entirely before conversion. Note: quality depends on HTML
    structure; JavaScript-rendered or table-heavy pages may produce noisy output.
    """
    soup = BeautifulSoup(html, "lxml")
    for tag_name in ("script", "style"):
        for tag in soup.find_all(tag_name):
            tag.decompose()
    return md.markdownify(str(soup), heading_style="ATX")
