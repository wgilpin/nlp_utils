"""PDF fetching utilities — fetch and extract text from a PDF URL."""

import base64
import logging
from io import BytesIO

import httpx
from pypdf import PdfReader

logger = logging.getLogger(__name__)

_USER_AGENT = "Mozilla/5.0 (compatible; nlp-utils/1.0)"


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

    text = "\n\n".join(pages)
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
