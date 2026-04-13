"""PDF and URL fetching utilities — fetch and extract text from PDFs and URLs."""

import base64
import logging
import re
from datetime import date
from io import BytesIO

import httpx
from pypdf import PdfReader

logger = logging.getLogger(__name__)

_USER_AGENT = "Mozilla/5.0 (compatible; nlp-utils/1.0)"


def extract_pdf_text_from_bytes(data: bytes) -> str:
    """Extract text from raw PDF bytes via pypdf.

    Raises:
        ValueError: if no text could be extracted (image-only PDF).
    """
    reader = PdfReader(BytesIO(data))
    page_count = len(reader.pages)
    logger.debug("PDF has %d pages", page_count)

    pages: list[str] = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        if not page_text:
            logger.warning("PDF page %d/%d returned empty text", i + 1, page_count)
        pages.append(page_text)

    text = "\n\n".join(pages)
    if not text.strip():
        raise ValueError(
            "No text extracted from PDF bytes. "
            "The file may be image-only; use fetch_pdf_text_llm instead."
        )
    return text


async def fetch_url_text(url: str, timeout: float = 30.0) -> str:
    """Fetch a URL and return clean extracted text.

    Uses httpx + extract_clean_html from nlp_utils.html.

    Raises:
        httpx.HTTPError: on network or HTTP failure.
        ValueError: if extracted text is empty after cleaning.
    """
    from nlp_utils.html import extract_clean_html

    logger.info("Fetching URL url=%s", url)
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
        response = await client.get(url, headers={"User-Agent": _USER_AGENT})
    response.raise_for_status()

    text = extract_clean_html(response.text)
    if not text.strip():
        raise ValueError(f"No text extracted from URL: {url}")
    logger.info("Fetched URL url=%s chars=%d", url, len(text))
    return text


async def fetch_url_text_with_metadata(
    url: str, timeout: float = 30.0
) -> tuple[str, date | None]:
    """Fetch a URL and return ``(clean_text, publication_date)``.

    Like :func:`fetch_url_text` but also extracts a publication date from the
    raw HTML before stripping tags (so date markup is available).

    Raises:
        httpx.HTTPError: on network or HTTP failure.
        ValueError: if extracted text is empty after cleaning.
    """
    from nlp_utils.html import extract_clean_html, extract_publication_date

    logger.info("Fetching URL (with metadata) url=%s", url)
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
        response = await client.get(url, headers={"User-Agent": _USER_AGENT})
    response.raise_for_status()

    raw_html = response.text
    pub_date = extract_publication_date(raw_html)
    text = extract_clean_html(raw_html)
    if not text.strip():
        raise ValueError(f"No text extracted from URL: {url}")
    logger.info("Fetched URL url=%s chars=%d pub_date=%s", url, len(text), pub_date)
    return text, pub_date


async def fetch_pdf_text(url: str, timeout: float = 30.0) -> str:
    """Fetch a PDF from *url* and return its full text content via pypdf.

    Raises:
        httpx.HTTPError: on network or HTTP failure.
        ValueError: if no text could be extracted from any page.
    """
    logger.info("Fetching PDF url=%s", url)
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
        response = await client.get(url, headers={"User-Agent": _USER_AGENT})
    response.raise_for_status()

    reader = PdfReader(BytesIO(response.content))
    page_count = len(reader.pages)
    logger.debug("PDF has %d pages", page_count)

    pages: list[str] = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        if not page_text:
            logger.warning("PDF page %d/%d returned empty text", i + 1, page_count)
        else:
            logger.debug("PDF page %d/%d: %d chars", i + 1, page_count, len(page_text))
        pages.append(page_text)

    text = "\n\n".join(pages).replace("\x00", "")
    if not text.strip():
        raise ValueError(
            f"No text extracted from PDF at {url}. "
            "The file may be image-only; use fetch_pdf_text_llm instead."
        )
    return text


async def fetch_pdf_text_llm(
    url: str,
    model: str = "claude-opus-4-6",
    prompt: str = "Extract all text from this PDF, preserving its structure.",
    timeout: float = 30.0,
) -> str:
    """Fetch a PDF from *url* and use a vision-capable LLM to extract text.

    Uses the native PDF document block format supported by models with built-in
    PDF understanding (Claude 3.5+, Gemini 1.5+). Other models will raise an
    error from the provider.

    Args:
        url: URL of the PDF to fetch.
        model: LiteLLM model string (e.g. "claude-opus-4-6", "gemini/gemini-1.5-pro").
        prompt: Instruction sent alongside the PDF.
        timeout: HTTP request timeout in seconds.

    Raises:
        httpx.HTTPError: on network or HTTP failure.
    """
    import litellm

    logger.info("Fetching PDF for LLM parsing url=%s model=%s", url, model)
    async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
        response = await client.get(url, headers={"User-Agent": _USER_AGENT})
    response.raise_for_status()

    b64_data = base64.b64encode(response.content).decode()
    logger.debug("PDF encoded to base64 (%d bytes raw)", len(response.content))

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": b64_data,
                    },
                },
            ],
        }
    ]

    llm_response = await litellm.acompletion(model=model, messages=messages)
    result: str = llm_response.choices[0].message.content
    logger.info("LLM extracted %d chars from PDF at url=%s", len(result), url)
    return result


_ARXIV_ID_RE = re.compile(r"arxiv\.org/(?:abs|pdf|html)/(\d{4}\.\d{4,5}(?:v\d+)?)")


async def fetch_arxiv_text(url: str, timeout: float = 30.0) -> str:
    """Fetch an arXiv paper and return its full text.

    Accepts any arXiv URL form: abs page, pdf link, or pdf with .pdf extension.

    Args:
        url: An arXiv URL (e.g. https://arxiv.org/abs/2301.00001).
        timeout: HTTP request timeout in seconds.

    Raises:
        ValueError: if *url* is not a recognisable arXiv URL.
        httpx.HTTPError: on network or HTTP failure.
    """
    match = _ARXIV_ID_RE.search(url)
    if not match:
        raise ValueError(f"Not a valid arXiv URL: {url!r}")

    paper_id = match.group(1)
    pdf_url = f"https://arxiv.org/pdf/{paper_id}"
    logger.info("Fetching arXiv paper id=%s url=%s", paper_id, pdf_url)
    return await fetch_pdf_text(pdf_url, timeout=timeout)
