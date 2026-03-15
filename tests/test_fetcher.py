"""Tests for PDF fetching utilities (network calls mocked)."""

import base64
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pypdf import PdfWriter

from nlp_utils.fetcher import fetch_pdf_text, fetch_pdf_text_llm


def _make_pdf_bytes(pages: list[str]) -> bytes:
    """Build a minimal valid PDF with the given text on each page."""
    writer = PdfWriter()
    for _ in pages:
        writer.add_blank_page(width=612, height=792)
    buf = BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _make_mock_response(content: bytes, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.content = content
    resp.status_code = status_code
    resp.raise_for_status = MagicMock()
    return resp


class TestFetchPdfText:
    @pytest.mark.asyncio
    async def test_returns_extracted_text(self, mocker):
        pdf_bytes = _make_pdf_bytes(["page1"])
        mock_resp = _make_mock_response(pdf_bytes)

        mock_get = AsyncMock(return_value=mock_resp)
        mocker.patch("nlp_utils.fetcher.httpx.AsyncClient")
        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = mock_get
        mocker.patch("nlp_utils.fetcher.httpx.AsyncClient", return_value=mock_client)

        # Patch PdfReader to return controlled text
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Hello from page one"
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]
        with patch("nlp_utils.fetcher.PdfReader", return_value=mock_reader):
            result = await fetch_pdf_text("https://example.com/doc.pdf")

        assert "Hello from page one" in result

    @pytest.mark.asyncio
    async def test_raises_on_empty_pdf(self, mocker):
        pdf_bytes = _make_pdf_bytes([""])
        mock_resp = _make_mock_response(pdf_bytes)

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)
        mocker.patch("nlp_utils.fetcher.httpx.AsyncClient", return_value=mock_client)

        mock_page = MagicMock()
        mock_page.extract_text.return_value = ""
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page]
        with patch("nlp_utils.fetcher.PdfReader", return_value=mock_reader):
            with pytest.raises(ValueError, match="No text extracted"):
                await fetch_pdf_text("https://example.com/empty.pdf")

    @pytest.mark.asyncio
    async def test_multiple_pages_joined(self, mocker):
        mock_resp = _make_mock_response(b"fakepdf")

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)
        mocker.patch("nlp_utils.fetcher.httpx.AsyncClient", return_value=mock_client)

        pages = [MagicMock(), MagicMock()]
        pages[0].extract_text.return_value = "Page one text"
        pages[1].extract_text.return_value = "Page two text"
        mock_reader = MagicMock()
        mock_reader.pages = pages
        with patch("nlp_utils.fetcher.PdfReader", return_value=mock_reader):
            result = await fetch_pdf_text("https://example.com/multi.pdf")

        assert "Page one text" in result
        assert "Page two text" in result


class TestFetchPdfTextLlm:
    @pytest.mark.asyncio
    async def test_calls_litellm_with_document_block(self, mocker):
        pdf_content = b"%PDF-fake-content"
        mock_resp = _make_mock_response(pdf_content)

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)
        mocker.patch("nlp_utils.fetcher.httpx.AsyncClient", return_value=mock_client)

        llm_result = MagicMock()
        llm_result.choices[0].message.content = "Extracted by LLM"
        mock_acompletion = AsyncMock(return_value=llm_result)
        mocker.patch("litellm.acompletion", mock_acompletion)

        result = await fetch_pdf_text_llm(
            "https://example.com/doc.pdf",
            model="claude-opus-4-6",
        )

        assert result == "Extracted by LLM"
        call_kwargs = mock_acompletion.call_args
        messages = call_kwargs.kwargs["messages"]
        content_blocks = messages[0]["content"]
        doc_block = next(b for b in content_blocks if b.get("type") == "document")
        assert doc_block["source"]["media_type"] == "application/pdf"
        assert doc_block["source"]["data"] == base64.b64encode(pdf_content).decode()

    @pytest.mark.asyncio
    async def test_returns_llm_response(self, mocker):
        mock_resp = _make_mock_response(b"pdf-bytes")

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)
        mocker.patch("nlp_utils.fetcher.httpx.AsyncClient", return_value=mock_client)

        llm_result = MagicMock()
        llm_result.choices[0].message.content = "LLM output text"
        mocker.patch("litellm.acompletion", AsyncMock(return_value=llm_result))

        result = await fetch_pdf_text_llm("https://example.com/doc.pdf")
        assert result == "LLM output text"
