"""NLP utilities library for common NLP routines."""

__version__ = "0.1.0"

from nlp_utils.chunker import chunk_sentences
from nlp_utils.fetcher import (
    extract_pdf_text_from_bytes,
    fetch_arxiv_text,
    fetch_pdf_text,
    fetch_pdf_text_llm,
    fetch_url_text,
    fetch_url_text_with_metadata,
)
from nlp_utils.html import extract_article, extract_clean_html, extract_publication_date, html_to_markdown
from nlp_utils.llm import to_markdown_llm
from nlp_utils.youtube import fetch_youtube_transcript

__all__ = [
    "extract_pdf_text_from_bytes",
    "fetch_arxiv_text",
    "fetch_pdf_text",
    "fetch_pdf_text_llm",
    "fetch_url_text",
    "fetch_url_text_with_metadata",
    "fetch_youtube_transcript",
    "extract_article",
    "extract_clean_html",
    "extract_publication_date",
    "html_to_markdown",
    "to_markdown_llm",
    "chunk_sentences",
]
