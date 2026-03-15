"""HTML parsing utilities."""

import logging

import markdownify as md
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_NOISE_TAGS: frozenset[str] = frozenset(
    {"nav", "aside", "header", "footer", "script", "style", "iframe", "noscript"}
)


def extract_article(html: str) -> str:
    """Return text of <article>, falling back to <main> then <body>.

    Strips all content if no suitable element is found.
    """
    soup = BeautifulSoup(html, "lxml")
    tag = soup.find("article") or soup.find("main") or soup.find("body") or soup
    chosen = tag.name if hasattr(tag, "name") else "document"
    logger.debug("Extracting text from <%s>", chosen)
    return tag.get_text(separator="\n", strip=True)


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
