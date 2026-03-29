"""NLP utilities library for common NLP routines."""

__version__ = "0.1.0"

from nlp_utils.chunker import chunk_sentences
from nlp_utils.fetcher import (
    extract_pdf_text_from_bytes,
    fetch_pdf_text,
    fetch_pdf_text_llm,
    fetch_url_text,
)
from nlp_utils.html import extract_article, extract_clean_html, html_to_markdown
from nlp_utils.llm import to_markdown_llm
from nlp_utils.youtube import fetch_youtube_transcript

__all__ = [
    "extract_pdf_text_from_bytes",
    "fetch_pdf_text",
    "fetch_pdf_text_llm",
    "fetch_url_text",
    "fetch_youtube_transcript",
    "extract_article",
    "extract_clean_html",
    "html_to_markdown",
    "to_markdown_llm",
    "chunk_sentences",
]
