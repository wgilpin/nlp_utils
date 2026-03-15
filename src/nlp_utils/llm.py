"""LLM-based text utilities."""

import logging

logger = logging.getLogger(__name__)


async def to_markdown_llm(
    text: str,
    model: str = "claude-opus-4-6",
    prompt: str = "Convert the following text to well-structured Markdown.",
) -> str:
    """Convert *text* to Markdown using a LiteLLM model.

    Args:
        text: Plain text (or lightly structured text) to convert.
        model: LiteLLM model string, e.g. "claude-opus-4-6".
        prompt: System-level instruction prepended before the text.

    Returns:
        Markdown-formatted string as returned by the model.
    """
    import litellm

    messages = [{"role": "user", "content": f"{prompt}\n\n{text}"}]
    logger.info("Converting text to Markdown via LLM model=%s (%d chars)", model, len(text))
    response = await litellm.acompletion(model=model, messages=messages)
    result: str = response.choices[0].message.content.strip()
    logger.debug("LLM returned %d chars of Markdown", len(result))
    return result
