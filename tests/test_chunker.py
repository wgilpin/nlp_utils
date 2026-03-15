"""Tests for chunk_sentences."""

import pytest

from nlp_utils.chunker import chunk_sentences


def test_empty_string():
    assert chunk_sentences("") == []


def test_whitespace_only():
    assert chunk_sentences("   \n\n   ") == []


def test_single_chunk_fits():
    text = "Hello world."
    result = chunk_sentences(text, chunk_size=100)
    assert result == ["Hello world."]


def test_respects_chunk_size():
    text = "Hello world. " * 100
    result = chunk_sentences(text, chunk_size=200)
    for chunk in result:
        assert len(chunk) <= 200


def test_all_text_preserved():
    """No content should be lost across chunks."""
    text = "The quick brown fox. " * 50
    result = chunk_sentences(text, chunk_size=100)
    # Every word from the original should appear somewhere in the chunks
    combined = " ".join(result)
    assert "quick brown fox" in combined


def test_overlap_less_than_chunk_size():
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_sentences("Some text.", chunk_size=100, chunk_overlap=100)


def test_overlap_carries_tail():
    """With overlap, the end of one chunk should appear at the start of the next."""
    # Use a long repeating text so we definitely get multiple chunks
    text = "Alpha beta gamma delta epsilon. " * 30
    result = chunk_sentences(text, chunk_size=100, chunk_overlap=20)
    assert len(result) > 1
    # The overlap tail of chunk N should appear at the start of chunk N+1
    for prev, curr in zip(result, result[1:]):
        tail = prev[-20:]
        assert tail in curr


def test_paragraph_splitting():
    para1 = "First paragraph content here."
    para2 = "Second paragraph content here."
    text = f"{para1}\n\n{para2}"
    result = chunk_sentences(text, chunk_size=1000)
    assert len(result) == 1
    assert para1 in result[0]
    assert para2 in result[0]


def test_sentence_level_split():
    # Two sentences that together exceed chunk_size
    s1 = "A" * 60 + "."
    s2 = "B" * 60 + "."
    text = f"{s1} {s2}"
    result = chunk_sentences(text, chunk_size=70)
    assert len(result) >= 2
    for chunk in result:
        assert len(chunk) <= 70


def test_word_level_split():
    # One very long "sentence" (no sentence boundary) made of many words
    words = " ".join(["word"] * 200)
    result = chunk_sentences(words, chunk_size=50)
    for chunk in result:
        assert len(chunk) <= 50


def test_char_level_split():
    # A single unbreakable token longer than chunk_size
    text = "x" * 300
    result = chunk_sentences(text, chunk_size=100)
    for chunk in result:
        assert len(chunk) <= 100
