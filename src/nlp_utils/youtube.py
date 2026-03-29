"""YouTube transcript extraction utilities."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

_YOUTUBE_ID_RE = re.compile(
    r"(?:v=|youtu\.be/|embed/|shorts/)([A-Za-z0-9_-]{11})"
)


def _extract_video_id(url_or_id: str) -> str:
    """Extract the 11-character YouTube video ID from a URL or bare ID."""
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url_or_id):
        return url_or_id
    match = _YOUTUBE_ID_RE.search(url_or_id)
    if match:
        return match.group(1)
    raise ValueError(f"Cannot extract YouTube video ID from: {url_or_id!r}")


async def fetch_youtube_transcript(url_or_id: str) -> tuple[str, str]:
    """Fetch transcript for a YouTube video.

    Args:
        url_or_id: Full YouTube URL or bare 11-character video ID.

    Returns:
        (title, transcript_text) — title falls back to video_id if unavailable.

    Raises:
        ValueError: if no transcript is available or the video cannot be found.
    """
    from youtube_transcript_api import (  # type: ignore[import-untyped]
        NoTranscriptFound,
        TranscriptsDisabled,
        YouTubeTranscriptApi,
    )

    video_id = _extract_video_id(url_or_id)
    logger.info("Fetching YouTube transcript video_id=%s", video_id)

    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        # Prefer manually created, fall back to auto-generated
        try:
            transcript = transcript_list.find_manually_created_transcript(
                ["en", "en-US", "en-GB"]
            )
        except Exception:
            transcript = transcript_list.find_generated_transcript(["en", "en-US", "en-GB"])

        snippets = transcript.fetch()
        text = " ".join(s.get("text", "") for s in snippets)
    except TranscriptsDisabled as exc:
        raise ValueError(f"Transcripts are disabled for video {video_id}") from exc
    except NoTranscriptFound as exc:
        raise ValueError(f"No transcript available for video {video_id}") from exc
    except Exception as exc:
        raise ValueError(f"Failed to fetch transcript for video {video_id}: {exc}") from exc

    if not text.strip():
        raise ValueError(f"Empty transcript for video {video_id}")

    # Try to get a title via yt-dlp (best-effort; fall back to video ID)
    title = _fetch_title(video_id)
    logger.info(
        "YouTube transcript fetched video_id=%s title=%r chars=%d", video_id, title, len(text)
    )
    return title, text


def _fetch_title(video_id: str) -> str:
    """Return the video title using yt-dlp metadata; falls back to video_id."""
    try:
        import yt_dlp  # type: ignore[import-untyped]

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "skip_download": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(
                f"https://www.youtube.com/watch?v={video_id}", download=False
            )
            return str(info.get("title", video_id))
    except Exception:
        logger.debug("Could not fetch title for video_id=%s, using ID as fallback", video_id)
        return video_id
