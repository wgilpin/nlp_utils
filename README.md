# nlp-utils

A utility library for common NLP routines: PDF and HTML extraction, markdown conversion, and recursive text chunking.

## Installation

```bash
uv add nlp-utils
```

Or for a local editable install (run from the repo root, where `pyproject.toml` lives):

```bash
uv pip install -e .
```

## API Reference

### PDF Fetching

#### `fetch_pdf_text(url, timeout=30.0) -> str`

Fetch a PDF from a URL and extract its text using [pypdf](https://pypdf.readthedocs.io/).

```python
import asyncio
from nlp_utils import fetch_pdf_text

text = asyncio.run(fetch_pdf_text("https://example.com/paper.pdf"))
```

Raises `ValueError` if no text can be extracted (e.g. image-only PDFs — use `fetch_pdf_text_llm` instead).

---

#### `fetch_pdf_text_llm(url, model="claude-opus-4-6", prompt=..., timeout=30.0) -> str`

Fetch a PDF from a URL and extract its text using a vision-capable LLM via [LiteLLM](https://docs.litellm.ai/).

Uses the native PDF document block format — requires a model with built-in PDF understanding (Claude 3.5+, Gemini 1.5+).

```python
import asyncio
from nlp_utils import fetch_pdf_text_llm

text = asyncio.run(fetch_pdf_text_llm(
    "https://example.com/scanned.pdf",
    model="gemini-2.5-flash",
))
```

---

#### `fetch_arxiv_text(url, timeout=30.0) -> str`

Fetch an arXiv paper and return its full text. Accepts any arXiv URL form: abstract page, PDF link, or PDF link with `.pdf` extension.

```python
import asyncio
from nlp_utils import fetch_arxiv_text

text = asyncio.run(fetch_arxiv_text("https://arxiv.org/abs/2301.00001"))
```

Raises `ValueError` for non-arXiv URLs. Raises `ValueError` if no text can be extracted (image-only PDF — use `fetch_pdf_text_llm` on the PDF URL instead).

---

### HTML Parsing

#### `extract_article(html) -> str`

Extract the main article text from an HTML string. Targets `<article>`, falling back to `<main>` then `<body>`.

```python
from nlp_utils import extract_article

text = extract_article(html)
```

---

#### `extract_clean_html(html) -> str`

Extract body text from HTML with noise elements removed. Strips: `nav`, `aside`, `header`, `footer`, `script`, `style`, `iframe`, `noscript`.

```python
from nlp_utils import extract_clean_html

text = extract_clean_html(html)
```

---

### Markdown Conversion

#### `html_to_markdown(html) -> str`

Convert HTML to Markdown using [markdownify](https://github.com/matthewwithanm/python-markdownify). Uses ATX-style headings. Script and style elements are removed before conversion.

```python
from nlp_utils import html_to_markdown

md = html_to_markdown(html)
```

---

#### `to_markdown_llm(text, model="claude-opus-4-6", prompt=...) -> str`

Convert plain text to well-structured Markdown using a LiteLLM model.

```python
import asyncio
from nlp_utils import to_markdown_llm

md = asyncio.run(to_markdown_llm(raw_text, model="claude-opus-4-6"))
```

---

### Text Chunking

#### `chunk_sentences(text, chunk_size=1000, chunk_overlap=0) -> list[str]`

Split text into chunks of at most `chunk_size` characters. Splits recursively at paragraph → sentence → word → character boundaries until all chunks fit.

```python
from nlp_utils import chunk_sentences

chunks = chunk_sentences(text, chunk_size=512, chunk_overlap=50)
```

- `chunk_size`: maximum characters per chunk
- `chunk_overlap`: characters from the end of each chunk carried into the next (must be less than `chunk_size`)

Requires the NLTK `punkt_tab` tokenizer, which is downloaded automatically on first use.

---

## Typical Workflow

```python
import asyncio
import httpx
from nlp_utils import extract_clean_html, html_to_markdown, chunk_sentences

# Fetch a web page
response = httpx.get("https://example.com/article")

# Parse and clean
text = extract_clean_html(response.text)

# Convert to Markdown
md = html_to_markdown(response.text)

# Chunk for embedding / retrieval
chunks = chunk_sentences(text, chunk_size=512, chunk_overlap=64)
```

---

## Development

Requires Python 3.13+.

```bash
uv sync
uv run pytest
```
