"""Tests for LLM-based markdown conversion (LLM calls mocked)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from nlp_utils.llm import to_markdown_llm


class TestToMarkdownLlm:
    @pytest.mark.asyncio
    async def test_returns_llm_response(self, mocker):
        llm_result = MagicMock()
        llm_result.choices[0].message.content = "# Markdown Output\n\nSome content."
        mocker.patch("litellm.acompletion", AsyncMock(return_value=llm_result))

        result = await to_markdown_llm("plain text input")
        assert result == "# Markdown Output\n\nSome content."

    @pytest.mark.asyncio
    async def test_passes_text_in_message(self, mocker):
        llm_result = MagicMock()
        llm_result.choices[0].message.content = "output"
        mock_acompletion = AsyncMock(return_value=llm_result)
        mocker.patch("litellm.acompletion", mock_acompletion)

        await to_markdown_llm("my input text", model="claude-opus-4-6")

        call_kwargs = mock_acompletion.call_args.kwargs
        assert call_kwargs["model"] == "claude-opus-4-6"
        user_content = call_kwargs["messages"][0]["content"]
        assert "my input text" in user_content

    @pytest.mark.asyncio
    async def test_strips_whitespace_from_response(self, mocker):
        llm_result = MagicMock()
        llm_result.choices[0].message.content = "  \n# Output\n  "
        mocker.patch("litellm.acompletion", AsyncMock(return_value=llm_result))

        result = await to_markdown_llm("text")
        assert result == "# Output"

    @pytest.mark.asyncio
    async def test_custom_prompt_used(self, mocker):
        llm_result = MagicMock()
        llm_result.choices[0].message.content = "done"
        mock_acompletion = AsyncMock(return_value=llm_result)
        mocker.patch("litellm.acompletion", mock_acompletion)

        await to_markdown_llm("text", prompt="Custom instruction.")
        user_content = mock_acompletion.call_args.kwargs["messages"][0]["content"]
        assert "Custom instruction." in user_content
