"""Recursive sentence chunker."""

import logging

import nltk

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)


def _split_atoms(text: str, level: int) -> list[str]:
    if level == 0:
        return [p.strip() for p in text.split("\n\n") if p.strip()]
    if level == 1:
        return nltk.tokenize.sent_tokenize(text)
    if level == 2:
        return text.split()
    # level 3+: handled by caller (character slices)
    return [text]


def _pack(
    atoms: list[str],
    chunk_size: int,
    chunk_overlap: int,
    level: int,
) -> list[str]:
    chunks: list[str] = []
    current_parts: list[str] = []
    current_len: int = 0

    for atom in atoms:
        atom_len = len(atom)

        if atom_len > chunk_size:
            # Flush current buffer first
            if current_parts:
                chunks.append("\n\n".join(current_parts) if level == 0 else " ".join(current_parts))
                current_parts, current_len = [], 0

            if level >= 2:
                # Character-level slicing for pathologically long tokens
                for i in range(0, atom_len, chunk_size):
                    chunks.append(atom[i : i + chunk_size])
            else:
                sub_chunks = _pack(
                    _split_atoms(atom, level + 1),
                    chunk_size,
                    chunk_overlap,
                    level + 1,
                )
                chunks.extend(sub_chunks)
            continue

        sep = "\n\n" if level == 0 else " "
        sep_len = len(sep) if current_parts else 0

        if current_len + sep_len + atom_len <= chunk_size:
            current_parts.append(atom)
            current_len += sep_len + atom_len
        else:
            if current_parts:
                joined = (sep).join(current_parts)
                chunks.append(joined)
                # Overlap: carry tail of previous chunk text into next
                if chunk_overlap > 0:
                    tail = joined[-chunk_overlap:]
                    current_parts = [tail, atom]
                    current_len = len(tail) + len(sep) + atom_len
                else:
                    current_parts = [atom]
                    current_len = atom_len
            else:
                current_parts = [atom]
                current_len = atom_len

    if current_parts:
        sep = "\n\n" if level == 0 else " "
        chunks.append(sep.join(current_parts))

    return chunks


def chunk_sentences(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 0,
) -> list[str]:
    """Split *text* into chunks of at most *chunk_size* characters.

    Splits recursively: paragraphs → sentences → words → characters.
    *chunk_overlap* characters from the end of each chunk are prepended to
    the next chunk. Raises ValueError if chunk_overlap >= chunk_size.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
        )
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []
    return _pack(paragraphs, chunk_size, chunk_overlap, level=0)
