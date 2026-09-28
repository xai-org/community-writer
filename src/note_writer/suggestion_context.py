import random
import re

from data_models.writer_data_models import PostWithContext


MAX_SUGGESTION_CHARS = 500

MAX_SUGGESTED_SOURCES = 5
MAX_FREE_TEXT_SUGGESTIONS = 5

_WRITER_WRAPPER = "\n{sources_section}\n\n{details_section}\n\n"

_WRITER_SOURCES_PRESENT = """\
There were {total_requests} users who suggested a post on X as containing potentially useful information for writing this Community Note. \
The list below contains up to 5 of the top suggested posts, with the number preceding each post indicating how many users suggested it.
{source_list}

The suggested posts may be wrong or irrelevant, you are not obligated to use the suggested posts, and you should always do your own research.
Unless the post is a trustworthy and authoritative representation of first-hand information (e.g. a user stating their own opinion), \
then it is best to treat the post as a lead and cite an authoritative source instead."""

_WRITER_NO_SOURCES = (
    "No posts were suggested as potential sources for writing this Community Note."
)

_REJECTOR_WRAPPER = """
The following context was gathered from the users who requested a Community Note on this post. \
It may be wrong, irrelevant, or incomplete; you are not obligated to use it and should always do your own research.

{sources_section}

{details_section}
"""

_REJECTOR_SOURCES_PRESENT = """\
There were {total_requests} users who suggested a post on X as containing potentially useful information for this Community Note. \
The list below contains up to 5 of the top suggested posts, with the number preceding each post indicating how many users suggested it.
{source_list}
Unless a suggested post is a trustworthy and authoritative representation of first-hand information (e.g. a user stating their own opinion), \
it is best to treat the post as a lead and verify against an authoritative source."""

_REJECTOR_NO_SOURCES = (
    "No posts were suggested as potential sources for this Community Note."
)

_DETAILS_PRESENT = """\
Users also provided the following free-form suggestions about why this note may be needed (up to 5 shown):
{detail_list}"""

_NO_DETAILS = (
    "No free-form suggestions about why this note may be needed were provided."
)


def _normalize_suggestion_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "")
    text = "".join(ch for ch in text if ch.isprintable())
    text = text.replace('"', "'")
    text = re.sub(r"`{3,}", "`", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = text.lstrip("# ").strip()
    return f'"{text}"'


def sample_free_text_suggestions(
    post_with_context: PostWithContext,
    attempt_id: int,
) -> list[str]:
    suggestions = [
        _normalize_suggestion_text(s)
        for s in post_with_context.note_request_suggestions
    ]
    suggestions = [
        s for s in suggestions if s != '""' and len(s) <= MAX_SUGGESTION_CHARS
    ]
    if len(suggestions) <= MAX_FREE_TEXT_SUGGESTIONS:
        return suggestions
    rng = random.Random(f"{post_with_context.post.post_id}:{attempt_id}")
    return rng.sample(suggestions, MAX_FREE_TEXT_SUGGESTIONS)


def _build_block(
    post_with_context: PostWithContext,
    attempt_id: int,
    wrapper: str,
    sources_present: str,
    no_sources: str,
) -> str:
    sources = post_with_context.suggested_sources[:MAX_SUGGESTED_SOURCES]
    details = sample_free_text_suggestions(post_with_context, attempt_id)
    if not sources and not details:
        return ""

    if sources:
        sources_section = sources_present.format(
            total_requests=sum(s.count for s in post_with_context.suggested_sources),
            source_list="\n".join(f"{s.count}: {s.link}" for s in sources),
        )
    else:
        sources_section = no_sources

    if details:
        details_section = _DETAILS_PRESENT.format(
            detail_list="\n".join(f"- {d}" for d in details)
        )
    else:
        details_section = _NO_DETAILS

    return wrapper.format(
        sources_section=sources_section, details_section=details_section
    )


def build_writer_suggestion_block(
    post_with_context: PostWithContext,
    attempt_id: int,
) -> str:
    return _build_block(
        post_with_context,
        attempt_id,
        _WRITER_WRAPPER,
        _WRITER_SOURCES_PRESENT,
        _WRITER_NO_SOURCES,
    )


def build_rejector_suggestion_block(
    post_with_context: PostWithContext,
    attempt_id: int,
) -> str:
    return _build_block(
        post_with_context,
        attempt_id,
        _REJECTOR_WRAPPER,
        _REJECTOR_SOURCES_PRESENT,
        _REJECTOR_NO_SOURCES,
    )
